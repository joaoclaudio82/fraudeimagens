"""Hashes perceptuais, busca por distância de Hamming e índice persistente.

- dHash (64 bits, gradiente): rápido; mantido por compatibilidade e como informação.
- pHash (256 bits, DCT 16x16): em documentos, o hash de 64 bits captura só o
  layout (papel sobre fundo) e dois comprovantes diferentes do mesmo modelo ficam
  a 2-6 bits; com 256 bits a mesma foto reenviada por WhatsApp fica a <=6 bits e
  comprovantes diferentes a >=26. É o hash usado para decidir duplicidade.
- BK-tree: busca por raio de Hamming sem comparar com todos os registros.
- HashStore: SQLite com os hashes e o SHA-256, para achar reenvio do mesmo
  arquivo (exato) ou da mesma foto (aproximado) entre sessões e usuários.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

import cv2
import numpy as np
from PIL import Image

PHASH_SIZE = 16  # 16x16 coeficientes DCT = 256 bits


def dhash(image: Image.Image, size: int = 8) -> str:
    gray = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    return _bits_to_hex(bits.flatten())


def phash(image: Image.Image, hash_size: int = PHASH_SIZE, highfreq_factor: int = 4) -> str:
    side = hash_size * highfreq_factor
    gray = image.convert("L").resize((side, side), Image.Resampling.LANCZOS)
    values = np.asarray(gray, dtype=np.float32)
    dct = cv2.dct(values)[:hash_size, :hash_size]
    flat = dct.flatten()
    median = np.median(flat[1:])  # ignora o termo DC, que só carrega o brilho médio
    return _bits_to_hex(dct > median)


def _bits_to_hex(bits: np.ndarray) -> str:
    flat = np.asarray(bits).flatten()
    value = 0
    for bit in flat:
        value = (value << 1) | int(bool(bit))
    return f"{value:0{max(1, len(flat) // 4)}x}"


def hamming_distance(hash_a: str, hash_b: str) -> int:
    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()


class BKTree:
    """Árvore de Burkhard-Keller sobre a distância de Hamming de hashes hexadecimais."""

    def __init__(self) -> None:
        self._root: list[Any] | None = None  # [hash_int, payloads, {distance: child}]
        self.size = 0

    def add(self, hash_hex: str, payload: Any) -> None:
        value = int(hash_hex, 16)
        self.size += 1
        if self._root is None:
            self._root = [value, [payload], {}]
            return
        node = self._root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(payload)
                return
            child = node[2].get(distance)
            if child is None:
                node[2][distance] = [value, [payload], {}]
                return
            node = child

    def search(self, hash_hex: str, max_distance: int) -> list[tuple[int, Any]]:
        if self._root is None:
            return []
        value = int(hash_hex, 16)
        results: list[tuple[int, Any]] = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= max_distance:
                results.extend((distance, payload) for payload in node[1])
            for child_distance, child in node[2].items():
                if abs(child_distance - distance) <= max_distance:
                    stack.append(child)
        results.sort(key=lambda item: item[0])
        return results


class HashIndex(Protocol):
    def find_exact(self, sha256: str) -> list[dict[str, Any]]: ...

    def find_near(self, dhash_hex: str, phash_hex: str, max_dhash: int, max_phash: int) -> dict[str, Any] | None: ...


class ListHashIndex:
    """Compatibilidade com o histórico de sessão antigo: lista de dHash em hexadecimal."""

    def __init__(self, hashes: Iterable[str]) -> None:
        self.hashes = [h for h in hashes if isinstance(h, str) and len(h) == 16]

    def find_exact(self, sha256: str) -> list[dict[str, Any]]:
        return []

    def find_near(self, dhash_hex: str, phash_hex: str, max_dhash: int, max_phash: int) -> dict[str, Any] | None:
        if not self.hashes:
            return None
        best = min(self.hashes, key=lambda candidate: hamming_distance(dhash_hex, candidate))
        distance = hamming_distance(dhash_hex, best)
        return {"dhash_distance": distance, "phash_distance": None, "distance": distance,
                "within": distance <= max_dhash, "match": {"dhash": best}}


class HashStore:
    """Índice persistente em SQLite com busca aproximada em memória (BK-tree por pHash)."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS hashes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sha256 TEXT NOT NULL,
                dhash TEXT NOT NULL,
                phash TEXT NOT NULL,
                filename TEXT,
                reference TEXT,
                created_at TEXT NOT NULL
            )""")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_hashes_sha256 ON hashes(sha256)")
        self._conn.commit()
        self._tree: BKTree | None = None

    def add(self, sha256: str, dhash_hex: str, phash_hex: str, filename: str = "", reference: str | None = None) -> int:
        created_at = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            "INSERT INTO hashes (sha256, dhash, phash, filename, reference, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (sha256, dhash_hex, phash_hex, filename, reference, created_at))
        self._conn.commit()
        row_id = int(cursor.lastrowid)
        if self._tree is not None:
            self._tree.add(phash_hex, self._row_payload(row_id, sha256, dhash_hex, phash_hex, filename, reference, created_at))
        return row_id

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM hashes").fetchone()[0])

    def find_exact(self, sha256: str) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM hashes WHERE sha256 = ? ORDER BY id", (sha256,)).fetchall()
        return [dict(row) for row in rows]

    def find_near(self, dhash_hex: str, phash_hex: str, max_dhash: int, max_phash: int) -> dict[str, Any] | None:
        tree = self._ensure_tree()
        if tree.size == 0:
            return None
        # Raio maior que o limiar para também devolver o "mais próximo" quando nada está dentro.
        candidates = tree.search(phash_hex, max(max_phash * 2, 32))
        best: dict[str, Any] | None = None
        for phash_distance, payload in candidates:
            if len(payload["phash"]) != len(phash_hex):
                continue  # registro de outra versão do hash
            if best is None or phash_distance < best["distance"]:
                dhash_distance = hamming_distance(dhash_hex, payload["dhash"])
                best = {"phash_distance": phash_distance, "dhash_distance": dhash_distance, "distance": phash_distance,
                        "within": phash_distance <= max_phash, "match": payload}
        return best

    def all(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._conn.execute("SELECT * FROM hashes ORDER BY id").fetchall()]

    def close(self) -> None:
        self._conn.close()

    def _ensure_tree(self) -> BKTree:
        if self._tree is None:
            tree = BKTree()
            for row in self.all():
                tree.add(row["phash"], row)
            self._tree = tree
        return self._tree

    @staticmethod
    def _row_payload(row_id: int, sha256: str, dhash_hex: str, phash_hex: str, filename: str,
                     reference: str | None, created_at: str) -> dict[str, Any]:
        return {"id": row_id, "sha256": sha256, "dhash": dhash_hex, "phash": phash_hex,
                "filename": filename, "reference": reference, "created_at": created_at}


def as_index(known_hashes: Any) -> HashIndex | None:
    if known_hashes is None:
        return None
    if hasattr(known_hashes, "find_near") and hasattr(known_hashes, "find_exact"):
        return known_hashes
    return ListHashIndex(list(known_hashes))
