"""Contexto compartilhado entre os sinais: a imagem decodificada uma única vez."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass
class ImageContext:
    data: bytes
    filename: str
    image: Image.Image
    rgb: Image.Image
    np_rgb: np.ndarray
    gray: np.ndarray
    expected: dict[str, str] = field(default_factory=dict)
    known_hashes: Any = None
    # Espaço para um sinal publicar resultados intermediários que outro sinal reutiliza
    # (ex.: palavras e caixas do OCR reaproveitadas pela checagem tipográfica).
    shared: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.rgb.size[0]

    @property
    def height(self) -> int:
        return self.rgb.size[1]

    @property
    def format(self) -> str:
        return self.image.format or "desconhecido"

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        filename: str = "image",
        expected: dict[str, str] | None = None,
        known_hashes: Any = None,
    ) -> "ImageContext":
        image = Image.open(io.BytesIO(data))
        image.load()
        rgb = image.convert("RGB")
        np_rgb = np.asarray(rgb)
        gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
        return cls(
            data=data,
            filename=filename,
            image=image,
            rgb=rgb,
            np_rgb=np_rgb,
            gray=gray,
            expected={k: str(v) for k, v in (expected or {}).items() if v is not None},
            known_hashes=known_hashes,
        )
