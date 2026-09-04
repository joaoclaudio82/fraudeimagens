"""Extração estruturada e checagem semântica com um modelo de visão e linguagem (Claude).

Para fotos ruins, o Tesseract erra muito; um modelo de visão lê o comprovante como
um humano leria e devolve os campos em JSON, além de sinais que só a semântica
captura: assinatura presente, tipo de documento, indícios visíveis de edição.

Este sinal é desligado por padrão (``vlm_enabled=False``): a imagem sai do
ambiente para a API da Anthropic, o que exige base legal e contrato adequados
(LGPD), e cada chamada tem custo. As credenciais vêm do ambiente (ANTHROPIC_API_KEY
ou perfil do ``ant auth login``); nada é gravado aqui.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from ..config import AnalysisConfig
from ..context import ImageContext
from ..textnorm import match_field
from .base import SignalResult, finding

FIELD_ALIASES = {
    "pedido": "order_id", "order": "order_id", "numero": "order_id",
    "data": "date", "date": "date",
    "destinatario": "recipient", "destinatário": "recipient", "nome": "recipient", "recipient": "recipient",
    "valor": "value", "total": "value", "value": "value",
    "codigo": "code", "código": "code", "code": "code", "referencia": "code", "rastreio": "code",
}

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string", "description": "ex.: comprovante de entrega, nota fiscal, captura de tela, outro"},
        "order_id": {"type": ["string", "null"]},
        "date": {"type": ["string", "null"], "description": "data principal do documento, como impressa"},
        "recipient": {"type": ["string", "null"]},
        "value": {"type": ["string", "null"], "description": "valor principal, como impresso"},
        "code": {"type": ["string", "null"], "description": "código, referência ou rastreio"},
        "has_signature": {"type": "boolean"},
        "has_stamp": {"type": "boolean"},
        "legibility": {"type": "number", "description": "0 a 1"},
        "edit_signs": {"type": "array", "items": {"type": "string"},
                       "description": "indícios visíveis de edição: fonte diferente, alinhamento, fundo remendado, texto sobreposto"},
        "notes": {"type": "string"},
    },
    "required": ["document_type", "order_id", "date", "recipient", "value", "code", "has_signature", "has_stamp",
                 "legibility", "edit_signs", "notes"],
    "additionalProperties": False,
}

PROMPT = (
    "Você está auditando a imagem de um comprovante de entrega. Leia o documento e preencha o JSON pedido. "
    "Transcreva os campos exatamente como impressos (não corrija, não normalize). Se um campo não estiver visível, "
    "use null. Em edit_signs, liste apenas indícios visíveis na própria imagem, como fonte ou tamanho de letra "
    "diferente em um trecho, texto desalinhado, fundo remendado, borrão localizado ou texto sobreposto; se não houver, "
    "devolva lista vazia. Não afirme que houve fraude: descreva o que vê."
)


def _media_type(fmt: str) -> str:
    return "image/png" if (fmt or "").upper() == "PNG" else "image/jpeg"


def default_client() -> Any:
    import anthropic

    return anthropic.Anthropic()


class VlmSignal:
    name = "vlm"

    def __init__(self, client: Any = None) -> None:
        self._client = client

    def client(self) -> Any:
        if self._client is None:
            self._client = default_client()
        return self._client

    def extract(self, ctx: ImageContext, config: AnalysisConfig) -> tuple[dict[str, Any] | None, str | None]:
        data = ctx.data if _media_type(ctx.format) in ("image/jpeg", "image/png") else None
        if data is None:
            return None, "formato não suportado"
        response = self.client().messages.create(
            model=config.vlm_model,
            max_tokens=config.vlm_max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": _media_type(ctx.format),
                                                  "data": base64.standard_b64encode(data).decode("ascii")}},
                    {"type": "text", "text": PROMPT},
                ],
            }],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        )
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            return None, f"modelo recusou a análise ({getattr(details, 'category', None) or 'sem categoria'})"
        text = next((block.text for block in response.content if getattr(block, "type", "") == "text"), "")
        try:
            return json.loads(text), None
        except json.JSONDecodeError:
            return None, "resposta do modelo não é JSON válido"

    def run(self, ctx: ImageContext, config: AnalysisConfig) -> SignalResult:
        result = SignalResult(self.name)
        result.features["vlm_enabled"] = bool(config.vlm_enabled)
        if not config.vlm_enabled:
            result.details["note"] = "desligado (vlm_enabled=False)"
            return result

        try:
            extracted, warning = self.extract(ctx, config)
        except Exception as exc:  # rede, credenciais, cota: registra e segue sem o sinal
            extracted, warning = None, f"{type(exc).__name__}: {exc}"
        result.details["warning"] = warning
        result.features["vlm_available"] = extracted is not None
        if extracted is None:
            result.add(finding("vlm_unavailable", "Extração por modelo de visão indisponível", 0, warning or "", "baixo"))
            return result

        result.details["extracted"] = extracted
        result.features["vlm_legibility"] = round(float(extracted.get("legibility") or 0.0), 3)
        result.features["vlm_has_signature"] = bool(extracted.get("has_signature"))
        result.features["vlm_edit_signs"] = len(extracted.get("edit_signs") or [])
        ctx.shared["vlm_extracted"] = extracted

        matched = conflict = 0
        details: dict[str, Any] = {}
        for field, expected_value in ctx.expected.items():
            key = FIELD_ALIASES.get(field.strip().casefold())
            if not key or not expected_value:
                continue
            seen = extracted.get(key)
            if not seen:
                details[field] = {"matched": False, "seen": None}
                continue
            check = match_field(field, expected_value, str(seen), config.ocr_fuzzy_threshold, config.ocr_code_threshold)
            details[field] = {"matched": check.matched, "seen": seen, "score": check.score, "kind": check.kind}
            if check.matched:
                matched += 1
            else:
                conflict += 1
                result.add(finding(f"vlm_field_{field}", f"Campo divergente na leitura visual: {field}", 18,
                                   f"Esperado '{expected_value}'; o documento mostra '{seen}'.", "alto"))
        result.details["field_details"] = details
        result.features["vlm_fields_matched"] = matched
        result.features["vlm_fields_conflict"] = conflict

        signs = [str(sign) for sign in (extracted.get("edit_signs") or []) if str(sign).strip()]
        if signs:
            result.add(finding("vlm_edit_signs", "Indícios visuais de edição descritos pelo modelo", 12,
                               "; ".join(signs[:4]), "médio"))
        if config.vlm_require_signature and not extracted.get("has_signature"):
            result.add(finding("vlm_no_signature", "Assinatura não identificada no comprovante", 5,
                               "O modelo não encontrou assinatura visível.", "baixo"))
        return result
