"""Registro dos sinais padrão. A ordem importa quando um sinal reutiliza dados de outro."""

from __future__ import annotations

from ..config import AnalysisConfig
from .base import Signal, SignalResult, finding, region_box
from .copy_move import CopyMoveSignal
from .duplicates import DuplicateSignal
from .metadata import MetadataSignal
from .ocr import OcrSignal
from .quality import QualitySignal
from .recompression import RecompressionSignal


def default_signals(config: AnalysisConfig | None = None) -> list[Signal]:
    return [
        QualitySignal(),
        MetadataSignal(),
        RecompressionSignal(),
        CopyMoveSignal(),
        DuplicateSignal(),
        OcrSignal(),
    ]


__all__ = [
    "Signal", "SignalResult", "finding", "region_box", "default_signals",
    "QualitySignal", "MetadataSignal", "RecompressionSignal", "CopyMoveSignal", "DuplicateSignal", "OcrSignal",
]
