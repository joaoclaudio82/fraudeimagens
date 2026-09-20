"""Persistência das análises, fila de revisão humana e feedback rotulado (SQLite).

Princípios (LGPD):
- a imagem não é armazenada; ficam hashes, features, indicadores e o relatório;
- o texto do OCR pode conter dados pessoais (nome, endereço): guarde apenas se
  ``store_ocr_text`` estiver ligado e dentro do prazo de retenção;
- cada análise tem ``expires_at`` e ``purge_expired`` remove o que venceu;
- a revisão humana registra quem decidiu, o quê e por quê, e vira rótulo para
  recalibrar o score (``labeled_rows``).
"""

from __future__ import annotations

import json
import sqlite3
from functools import wraps
from threading import RLock
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .hashing import HashStore

REVIEW_STATUSES = ("pending", "confirmed_fraud", "legitimate", "inconclusive")
LABELS = {"confirmed_fraud": 1, "legitimate": 0}
DECISION_ORDER = {"BAIXO RISCO": 0, "ATENÇÃO": 1, "REVISAR": 2}


class ReviewConflict(ValueError):
    """O parecer foi alterado desde a leitura do cliente."""


def synchronized(method):
    """Não intercalar operações na conexão SQLite compartilhada entre threads."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapped


def _now() -> datetime:
    return datetime.now(timezone.utc)


def strip_report(report: dict[str, Any], store_ocr_text: bool = True) -> dict[str, Any]:
    """Remove imagens (não serializáveis) e, se pedido, o texto do OCR e as palavras."""
    cleaned = {key: value for key, value in report.items() if not key.startswith("_")}
    if not store_ocr_text:
        cleaned["ocr_text"] = ""
        signals = dict(cleaned.get("signals") or {})
        ocr = dict(signals.get("ocr") or {})
        ocr["text"] = ""
        ocr["words"] = []
        signals["ocr"] = ocr
        cleaned["signals"] = signals
    return cleaned


class AnalysisStore:
    def __init__(self, path: str | Path = ":memory:", retention_days: int = 90, store_ocr_text: bool = True,
                 hash_store: HashStore | None = None) -> None:
        self._lock = RLock()
        self.path = str(path)
        self.retention_days = retention_days
        self.store_ocr_text = store_ocr_text
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                filename TEXT,
                sha256 TEXT,
                reference TEXT,
                score INTEGER NOT NULL,
                decision TEXT NOT NULL,
                model TEXT,
                report_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at);
            CREATE INDEX IF NOT EXISTS idx_analyses_reference ON analyses(reference);
            CREATE TABLE IF NOT EXISTS reviews (
                analysis_id INTEGER PRIMARY KEY REFERENCES analyses(id) ON DELETE CASCADE,
                status TEXT NOT NULL,
                reviewer TEXT,
                note TEXT,
                reviewed_at TEXT
            );
        """)
        self._conn.commit()
        self._migrate_reviews()
        if hash_store is not None:
            self.hashes = hash_store
        else:
            self.hashes = HashStore(self.path) if self.path != ":memory:" else HashStore()

    def _migrate_reviews(self) -> None:
        # A migração guarda apenas o último parecer legado disponível.
        # Pareceres sobrescritos antes desta versão não podem ser reconstruídos.
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(reviews)")}
            if "version" not in columns:
                self._conn.execute("ALTER TABLE reviews ADD COLUMN version INTEGER NOT NULL DEFAULT 0")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS review_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    reviewer TEXT,
                    note TEXT,
                    reviewed_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    UNIQUE(analysis_id, version)
                )
            """)
            self._conn.execute("UPDATE reviews SET version = 1 WHERE reviewed_at IS NOT NULL AND version = 0")
            self._conn.execute("""
                INSERT OR IGNORE INTO review_history
                    (analysis_id, version, status, reviewer, note, reviewed_at, source)
                SELECT analysis_id, version, status, reviewer, note, reviewed_at, 'legacy_snapshot'
                FROM reviews WHERE reviewed_at IS NOT NULL
            """)

    # ------------------------------------------------------------------ escrita
    @synchronized
    def save(self, report: dict[str, Any], reference: str | None = None, register_hashes: bool = True) -> int:
        created = _now()
        expires = created + timedelta(days=self.retention_days)
        cleaned = strip_report(report, self.store_ocr_text)
        cursor = self._conn.execute(
            "INSERT INTO analyses (created_at, expires_at, filename, sha256, reference, score, decision, model, report_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (created.isoformat(), expires.isoformat(), cleaned.get("filename"), cleaned.get("sha256"), reference,
             int(cleaned["risk_score"]), cleaned["decision"], cleaned.get("score_model"),
             json.dumps(cleaned, ensure_ascii=False, default=str)))
        analysis_id = int(cursor.lastrowid)
        self._conn.execute("INSERT INTO reviews (analysis_id, status) VALUES (?, 'pending')", (analysis_id,))
        self._conn.commit()
        if register_hashes and cleaned.get("sha256"):
            phash = (cleaned.get("features") or {}).get("duplicates.phash") or ""
            self.hashes.add(cleaned["sha256"], cleaned.get("perceptual_hash") or "", phash,
                            cleaned.get("filename") or "", reference)
        return analysis_id

    @synchronized
    def review(self, analysis_id: int, status: str, reviewer: str = "", note: str = "",
               expected_version: int | None = None) -> None:
        if status not in REVIEW_STATUSES:
            raise ValueError(f"status inválido: {status}")
        if expected_version is not None and (type(expected_version) is not int or expected_version < 0):
            raise ValueError("expected_version deve ser um inteiro não negativo")
        with self._conn:
            # Serializa leitura/atualização também entre conexões e processos.
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute("SELECT version FROM reviews WHERE analysis_id = ?", (analysis_id,)).fetchone()
            if row is None:
                raise KeyError(analysis_id)
            if expected_version is not None and row["version"] != expected_version:
                raise ReviewConflict("parecer atualizado por outro revisor; recarregue a análise")
            version = row["version"] + 1
            reviewed_at = _now().isoformat()
            self._conn.execute(
                "UPDATE reviews SET status=?, reviewer=?, note=?, reviewed_at=?, version=? WHERE analysis_id=?",
                (status, reviewer, note, reviewed_at, version, analysis_id))
            self._conn.execute(
                "INSERT INTO review_history (analysis_id, version, status, reviewer, note, reviewed_at, source) "
                "VALUES (?, ?, ?, ?, ?, ?, 'review')",
                (analysis_id, version, status, reviewer, note, reviewed_at))

    @synchronized
    def review_history(self, analysis_id: int, limit: int = 50, after_version: int = 0) -> list[dict[str, Any]]:
        """Eventos do mais antigo ao mais recente, paginados pela versão."""
        if not 1 <= limit <= 500 or after_version < 0:
            raise ValueError("paginação inválida")
        if self.get(analysis_id) is None:
            raise KeyError(analysis_id)
        return [dict(row) for row in self._conn.execute(
            "SELECT version, status, reviewer, note, reviewed_at, source FROM review_history "
            "WHERE analysis_id = ? AND version > ? ORDER BY version LIMIT ?",
            (analysis_id, after_version, limit))]

    @synchronized
    def purge_expired(self, now: datetime | None = None) -> int:
        moment = (now or _now()).isoformat()
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            # Subconsultas evitam o limite de parâmetros em expurgos grandes.
            for table in ("review_history", "reviews"):
                self._conn.execute(
                    f"DELETE FROM {table} WHERE analysis_id IN (SELECT id FROM analyses WHERE expires_at <= ?)",
                    (moment,))
            removed = self._conn.execute("DELETE FROM analyses WHERE expires_at <= ?", (moment,)).rowcount
        return removed

    # ------------------------------------------------------------------ leitura
    @synchronized
    def get(self, analysis_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT a.*, r.status, r.reviewer, r.note, r.reviewed_at, r.version AS review_version FROM analyses a "
            "LEFT JOIN reviews r ON r.analysis_id = a.id WHERE a.id = ?", (analysis_id,)).fetchone()
        return self._hydrate(row) if row else None

    @synchronized
    def list(self, limit: int = 50, decision: str | None = None, reference: str | None = None) -> list[dict[str, Any]]:
        query = ("SELECT a.id, a.created_at, a.expires_at, a.filename, a.sha256, a.reference, a.score, a.decision, "
                 "a.model, r.status, r.reviewer, r.reviewed_at, r.version AS review_version FROM analyses a LEFT JOIN reviews r ON r.analysis_id = a.id")
        clauses, params = [], []
        if decision:
            clauses.append("a.decision = ?")
            params.append(decision)
        if reference:
            clauses.append("a.reference = ?")
            params.append(reference)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY a.id DESC LIMIT ?"
        params.append(limit)
        return [dict(row) for row in self._conn.execute(query, params).fetchall()]

    @synchronized
    def queue(self, limit: int = 50, min_decision: str = "ATENÇÃO") -> list[dict[str, Any]]:
        """Análises pendentes de revisão, das mais arriscadas para as menos."""
        if limit <= 0:
            return []
        if min_decision not in DECISION_ORDER:
            raise ValueError(f"decisão inválida: {min_decision}")
        floor = DECISION_ORDER[min_decision]
        decisions = [name for name, rank in DECISION_ORDER.items() if rank >= floor]
        marks = ",".join("?" for _ in decisions)
        rows = self._conn.execute(
            "SELECT a.id, a.created_at, a.filename, a.reference, a.score, a.decision, a.model, a.report_json, r.version AS review_version "
            "FROM analyses a JOIN reviews r ON r.analysis_id = a.id WHERE r.status = 'pending' "
            f"AND a.decision IN ({marks}) ORDER BY a.score DESC, a.id ASC LIMIT ?",
            (*decisions, limit)).fetchall()
        items = []
        for row in rows:
            report = json.loads(row["report_json"])
            items.append({
                "id": row["id"], "created_at": row["created_at"], "filename": row["filename"],
                "review_version": row["review_version"],
                "reference": row["reference"], "score": row["score"], "decision": row["decision"], "model": row["model"],
                "findings": [{"code": f["code"], "label": f["label"], "points": f["points"], "severity": f["severity"]}
                             for f in report.get("findings", [])],
            })
        return items

    @synchronized
    def labeled_rows(self) -> list[dict[str, Any]]:
        """Decisões humanas viram linhas de treino no mesmo formato do harness."""
        rows = self._conn.execute(
            "SELECT a.id, a.filename, a.reference, a.report_json, r.status FROM analyses a "
            "JOIN reviews r ON r.analysis_id = a.id WHERE r.status IN ('confirmed_fraud', 'legitimate') ORDER BY a.id").fetchall()
        out = []
        for row in rows:
            report = json.loads(row["report_json"])
            out.append({"file": row["filename"], "analysis_id": row["id"], "reference": row["reference"],
                        "label": LABELS[row["status"]], "kind": row["status"], "origin": "review",
                        "score": report.get("risk_score"), "decision": report.get("decision"),
                        "codes": [f["code"] for f in report.get("findings", [])],
                        "features": report.get("features", {})})
        return out

    @synchronized
    def stats(self) -> dict[str, Any]:
        total = int(self._conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0])
        by_decision = {row[0]: row[1] for row in self._conn.execute(
            "SELECT decision, COUNT(*) FROM analyses GROUP BY decision")}
        by_status = {row[0]: row[1] for row in self._conn.execute(
            "SELECT status, COUNT(*) FROM reviews GROUP BY status")}
        reviewed = sum(v for k, v in by_status.items() if k != "pending")
        confirmed = by_status.get("confirmed_fraud", 0)
        flagged_reviewed = int(self._conn.execute(
            "SELECT COUNT(*) FROM analyses a JOIN reviews r ON r.analysis_id = a.id "
            "WHERE a.decision = 'REVISAR' AND r.status IN ('confirmed_fraud', 'legitimate')").fetchone()[0])
        flagged_legit = int(self._conn.execute(
            "SELECT COUNT(*) FROM analyses a JOIN reviews r ON r.analysis_id = a.id "
            "WHERE a.decision = 'REVISAR' AND r.status = 'legitimate'").fetchone()[0])
        return {
            "analyses": total, "by_decision": by_decision, "by_review_status": by_status,
            "reviewed": reviewed, "confirmed_fraud": confirmed,
            # Taxa de reversão: quanto do que o sistema mandou revisar o humano considerou legítimo.
            "reversal_rate": round(flagged_legit / flagged_reviewed, 4) if flagged_reviewed else None,
            "hashes": self.hashes.count(), "retention_days": self.retention_days,
        }

    @synchronized
    def close(self) -> None:
        self._conn.close()
        self.hashes.close()

    @staticmethod
    def _hydrate(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["report"] = json.loads(item.pop("report_json"))
        return item
