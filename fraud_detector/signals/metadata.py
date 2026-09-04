"""Metadados do arquivo: EXIF (IFD0, sub-IFD Exif e GPS) e contêiner JPEG.

Cuidados que evitam falsos positivos comuns:
- iPhone grava a versão do iOS em Software ("17.5.1") e Samsung grava o firmware
  ("S918BXXU3CWL1"); só editores conhecidos pontuam alto, o resto é informativo.
- Sem EXIF não prova edição: WhatsApp, capturas de tela e exportações removem tudo.
  Vale pouco, mas registra que o arquivo não é o original da câmera.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from PIL import ExifTags, Image

from ..config import AnalysisConfig
from ..context import ImageContext
from ..jpegutil import jpeg_summary
from ..textnorm import infer_kind, parse_date
from .base import SignalResult, finding

EXIF_IFD = 0x8769
GPS_IFD = 0x8825

FIRMWARE_PATTERNS = (
    re.compile(r"^\d+(\.\d+)*$"),                       # iOS: "17.5.1"
    re.compile(r"^[A-Z]\d{3}[A-Z0-9]{5,}$"),            # Samsung: "S918BXXU3CWL1"
    re.compile(r"^(hdr\+|google|motorola|xiaomi|miui|hyperos|oneplus|oxygen|huawei|honor|emui|hmd|nokia|"
               r"lge?|sony|asus|realme|oppo|coloros|vivo|funtouch|tecno|infinix|zte|tcl|alcatel|"
               r"positivo|multilaser|camera|câmera|android)(?![a-z])", re.IGNORECASE),
    re.compile(r"^ver(sion|\.)?\s*\d", re.IGNORECASE),
    re.compile(r"^[A-Z]{2,4}\d{2,}[A-Z0-9._-]*$"),      # firmwares de câmera: "NX3000-1.05", "DSC-RX100"
)


def read_exif(image: Image.Image) -> dict[str, str]:
    """Tags do IFD0 mais as do sub-IFD Exif (DateTimeOriginal etc.) e GPS, por nome."""
    result: dict[str, str] = {}
    try:
        exif = image.getexif()
    except Exception:
        return result

    def store(items: Any, names: dict[int, str]) -> None:
        for key, value in items:
            name = names.get(key, str(key))
            if isinstance(value, bytes):
                value = f"<{len(value)} bytes>"
            result[name] = str(value)

    try:
        store(exif.items(), ExifTags.TAGS)
    except Exception:
        pass
    for ifd, names in ((EXIF_IFD, ExifTags.TAGS), (GPS_IFD, ExifTags.GPSTAGS)):
        try:
            store(exif.get_ifd(ifd).items(), names)
        except Exception:
            continue
    return result


def parse_exif_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value[:19], fmt)
        except ValueError:
            continue
    return None


def classify_software(software: str, config: AnalysisConfig) -> str:
    """'editor' (lista conhecida), 'firmware' (câmera/sistema) ou 'unknown'."""
    text = software.strip()
    if not text:
        return "none"
    lowered = text.casefold()
    if any(editor in lowered for editor in config.known_editors):
        return "editor"
    if any(pattern.search(text) for pattern in FIRMWARE_PATTERNS):
        return "firmware"
    return "unknown"


def expected_date(expected: dict[str, str]):
    for field, value in expected.items():
        if not value:
            continue
        if infer_kind(field, value) == "date":
            parsed = parse_date(value)
            if parsed:
                return parsed
    return None


class MetadataSignal:
    name = "metadata"

    def run(self, ctx: ImageContext, config: AnalysisConfig) -> SignalResult:
        result = SignalResult(self.name)
        exif = read_exif(ctx.image)
        container = jpeg_summary(ctx.image)
        result.details["exif"] = exif
        result.details["jpeg"] = container
        result.features["exif_tags"] = len(exif)
        result.features["has_camera_info"] = bool(exif.get("Make") or exif.get("Model"))
        result.features["has_gps"] = any(key.startswith("GPS") for key in exif)
        result.features["jpeg_quality"] = container.get("estimated_quality") or 0
        result.features["jpeg_standard_tables"] = container.get("standard_tables") is not False
        result.features["jpeg_subsampling"] = container.get("subsampling") or "n/a"

        software = exif.get("Software", "")
        kind = classify_software(software, config)
        result.features["software_kind"] = kind
        result.features["has_software_tag"] = bool(software)
        if kind == "editor":
            result.add(finding("editing_software", "Editor de imagem registrado nos metadados", 25,
                               f"EXIF Software: {software}", "alto"))
        elif kind == "unknown":
            result.add(finding("unknown_software", "Software não reconhecido nos metadados", 6,
                               f"EXIF Software: {software}. Não consta como firmware de câmera nem como editor conhecido.",
                               "baixo"))

        is_jpeg = bool(container.get("is_jpeg"))
        if is_jpeg and not exif:
            result.add(finding("no_exif", "Arquivo sem metadados EXIF", 5,
                               "Comum após WhatsApp, captura de tela ou exportação por editor; "
                               "não é o arquivo original da câmera.", "baixo"))
        if is_jpeg and container.get("standard_tables") is False:
            result.add(finding("nonstandard_quantization", "Tabelas de quantização JPEG fora do padrão", 4,
                               f"Qualidade estimada {container.get('estimated_quality')}; o último software a salvar "
                               "o arquivo usa tabelas próprias (alguns editores e apps).", "baixo"))

        original = parse_exif_datetime(exif.get("DateTimeOriginal"))
        modified = parse_exif_datetime(exif.get("DateTime"))
        result.features["has_capture_datetime"] = original is not None
        if original:
            result.details["capture_datetime"] = original.isoformat()
            target = expected_date(ctx.expected)
            if target:
                gap = abs((original.date() - target).days)
                result.features["capture_gap_days"] = gap
                if gap > config.exif_date_tolerance_days:
                    result.add(finding("exif_date_mismatch", "Data de captura diferente da data esperada", 20,
                                       f"EXIF DateTimeOriginal {original:%d/%m/%Y %H:%M}; esperado {target:%d/%m/%Y} "
                                       f"({gap} dia(s) de diferença).", "médio"))
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            if (original - now).days >= 1:
                result.add(finding("exif_date_future", "Data de captura no futuro", 10,
                                   f"EXIF DateTimeOriginal {original.isoformat()} é posterior à análise.", "baixo"))
            if modified:
                delta_minutes = (modified - original).total_seconds() / 60
                result.features["modified_after_capture_minutes"] = round(delta_minutes, 1)
                if delta_minutes > config.exif_modified_gap_minutes:
                    result.add(finding("exif_modified_later", "Arquivo regravado depois da captura", 10,
                                       f"DateTime {modified:%d/%m/%Y %H:%M} é {delta_minutes:.0f} min posterior a "
                                       f"DateTimeOriginal {original:%d/%m/%Y %H:%M}; editores atualizam essa tag ao salvar.",
                                       "baixo"))
        return result
