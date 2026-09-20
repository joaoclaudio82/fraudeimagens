"""API HTTP para integrar a triagem ao sistema operacional.

    uvicorn fraud_detector.api:app --host 0.0.0.0 --port 8000

Variáveis de ambiente:
    IMAGEGUARD_DB            caminho do SQLite (padrão data/imageguard.sqlite)
    IMAGEGUARD_RETENTION_DAYS  retenção das análises (padrão 90)
    IMAGEGUARD_STORE_OCR_TEXT  "0" para não guardar o texto do OCR
    IMAGEGUARD_SCORING_MODEL   caminho de um modelo JSON treinado (opcional)
    IMAGEGUARD_SCORING_MODE    "logistic" ou "max" (padrão max)
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from . import ANALYSIS_VERSION, AnalysisConfig, analyze_image
from .context import ImageLimitError
from .storage import DECISION_ORDER, REVIEW_STATUSES, AnalysisStore, ReviewConflict

EXPECTED_FIELDS = ("pedido", "data", "destinatario", "valor", "codigo")


class ReviewRequest(BaseModel):
    status: str = Field(description="confirmed_fraud | legitimate | inconclusive | pending")
    reviewer: str = ""
    note: str = ""
    expected_version: int | None = Field(default=None, ge=0, strict=True)


def config_from_env() -> AnalysisConfig:
    model = os.environ.get("IMAGEGUARD_SCORING_MODEL") or None
    mode = os.environ.get("IMAGEGUARD_SCORING_MODE", "max")
    return AnalysisConfig(scoring_model=model, scoring_mode=mode,
                          max_upload_bytes=int(os.environ.get("IMAGEGUARD_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024))),
                          max_image_pixels=int(os.environ.get("IMAGEGUARD_MAX_IMAGE_PIXELS", "20000000")))


def store_from_env() -> AnalysisStore:
    path = os.environ.get("IMAGEGUARD_DB", "data/imageguard.sqlite")
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return AnalysisStore(path, retention_days=int(os.environ.get("IMAGEGUARD_RETENTION_DAYS", "90")),
                         store_ocr_text=os.environ.get("IMAGEGUARD_STORE_OCR_TEXT", "1") != "0")


def _png_base64(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def ocr_available() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def create_app(store: AnalysisStore | None = None, config: AnalysisConfig | None = None) -> FastAPI:
    app = FastAPI(title="ImageGuard", version=ANALYSIS_VERSION,
                  description="Triagem explicável de risco em imagens de comprovantes. O score prioriza revisão humana; não comprova fraude.")
    state = {"store": store, "config": config}

    def get_store() -> AnalysisStore:
        if state["store"] is None:
            state["store"] = store_from_env()
        return state["store"]

    def get_config() -> AnalysisConfig:
        if state["config"] is None:
            state["config"] = config_from_env()
        return state["config"]

    @app.get("/health")
    def health() -> dict[str, Any]:
        cfg = get_config()
        return {"status": "ok", "analysis_version": ANALYSIS_VERSION, "ocr_available": ocr_available(),
                "scoring_model": cfg.scoring_model, "scoring_mode": cfg.scoring_mode, **get_store().stats()}

    @app.post("/analyze")
    async def analyze(
        file: UploadFile = File(...),
        expected: str | None = Form(None, description="JSON com os campos esperados"),
        pedido: str | None = Form(None), data: str | None = Form(None), destinatario: str | None = Form(None),
        valor: str | None = Form(None), codigo: str | None = Form(None),
        reference: str | None = Form(None, description="identificador do pedido/entrega no sistema"),
        persist: bool = Form(True, description="guardar a análise e os hashes para comparações futuras"),
        include_images: bool = Form(False, description="devolver ELA, ghost e copy-move em PNG base64"),
    ) -> dict[str, Any]:
        cfg = get_config()
        try:
            payload = await file.read(cfg.max_upload_bytes + 1)
        finally:
            await file.close()
        if len(payload) > cfg.max_upload_bytes:
            raise HTTPException(413, "arquivo excede o limite de bytes")
        if not payload:
            raise HTTPException(400, "arquivo vazio")
        fields: dict[str, str] = {}
        if expected:
            try:
                values = json.loads(expected)
                if not isinstance(values, dict):
                    raise ValueError("esperado um objeto JSON")
                for key, value in values.items():
                    if key not in EXPECTED_FIELDS:
                        raise ValueError(f"campo desconhecido: {key}")
                    if value is None:
                        continue
                    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                        raise ValueError(f"valor inválido para {key}")
                    if isinstance(value, float) and not math.isfinite(value):
                        raise ValueError(f"valor não finito para {key}")
                    if str(value).strip():
                        fields[key] = str(value)
            except (ValueError, AttributeError) as exc:
                raise HTTPException(400, f"expected inválido: {exc}") from exc
        for name, value in zip(EXPECTED_FIELDS, (pedido, data, destinatario, valor, codigo)):
            if value:
                fields[name] = value
        storage = get_store()
        try:
            report = analyze_image(payload, file.filename or "upload", fields, storage.hashes, cfg)
        except ImageLimitError as exc:
            raise HTTPException(413, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(422, f"não foi possível analisar a imagem: {type(exc).__name__}: {exc}") from exc
        images = report.pop("_images", {})
        report.pop("_ela_image", None)
        if persist:
            report["analysis_id"] = storage.save(report, reference or fields.get("pedido"))
        if include_images:
            report["images"] = {name: _png_base64(image) for name, image in images.items()}
        return report

    @app.get("/analyses")
    def list_analyses(limit: int = Query(50, ge=1, le=500), decision: str | None = None, reference: str | None = None) -> list[dict[str, Any]]:
        if decision is not None and decision not in DECISION_ORDER:
            raise HTTPException(422, "decision inválida")
        return get_store().list(limit=limit, decision=decision, reference=reference)

    @app.get("/analyses/{analysis_id}")
    def get_analysis(analysis_id: int) -> dict[str, Any]:
        item = get_store().get(analysis_id)
        if item is None:
            raise HTTPException(404, "análise não encontrada")
        return item

    @app.get("/review-queue")
    def review_queue(limit: int = Query(50, ge=1, le=500), min_decision: str = "ATENÇÃO") -> list[dict[str, Any]]:
        if min_decision not in DECISION_ORDER:
            raise HTTPException(422, "min_decision inválida")
        return get_store().queue(limit=limit, min_decision=min_decision)

    @app.post("/analyses/{analysis_id}/review")
    def review(analysis_id: int, body: ReviewRequest) -> dict[str, Any]:
        if body.status not in REVIEW_STATUSES:
            raise HTTPException(400, f"status deve ser um de {REVIEW_STATUSES}")
        try:
            get_store().review(analysis_id, body.status, body.reviewer, body.note, body.expected_version)
        except ReviewConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except KeyError:
            raise HTTPException(404, "análise não encontrada") from None
        return {"analysis_id": analysis_id, "status": body.status}

    @app.get("/analyses/{analysis_id}/reviews")
    def review_history(analysis_id: int, limit: int = Query(50, ge=1, le=500),
                       after_version: int = Query(0, ge=0)) -> list[dict[str, Any]]:
        try:
            return get_store().review_history(analysis_id, limit, after_version)
        except KeyError:
            raise HTTPException(404, "análise não encontrada") from None

    @app.get("/export/labels")
    def export_labels() -> list[dict[str, Any]]:
        return get_store().labeled_rows()

    @app.post("/maintenance/purge")
    def purge() -> dict[str, int]:
        return {"removed": get_store().purge_expired()}

    return app


app = create_app()
