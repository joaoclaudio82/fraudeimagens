"""Metadados EXIF do arquivo."""

from __future__ import annotations

from PIL import ExifTags, Image

from ..config import AnalysisConfig
from ..context import ImageContext
from .base import SignalResult, finding


def read_exif(image: Image.Image) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        for key, value in image.getexif().items():
            name = ExifTags.TAGS.get(key, str(key))
            if isinstance(value, bytes):
                value = f"<{len(value)} bytes>"
            result[name] = str(value)
    except Exception:
        pass
    return result


class MetadataSignal:
    name = "metadata"

    def run(self, ctx: ImageContext, config: AnalysisConfig) -> SignalResult:
        result = SignalResult(self.name)
        exif = read_exif(ctx.image)
        result.details["exif"] = exif
        result.features["exif_tags"] = len(exif)

        software = exif.get("Software", "")
        result.features["has_software_tag"] = bool(software)
        if software:
            result.add(finding("editing_software", "Software registrado nos metadados", 18,
                               f"EXIF Software: {software}", "médio"))
        return result
