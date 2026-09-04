"""Integridade (SHA-256) e duplicidade aproximada (hash perceptual)."""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

from ..config import AnalysisConfig
from ..context import ImageContext
from .base import SignalResult, finding


def perceptual_hash(image: Image.Image, size: int = 8) -> str:
    """dHash: compara pixels vizinhos em uma miniatura em tons de cinza."""
    gray = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    return f"{int(''.join('1' if bit else '0' for bit in bits.flat), 2):016x}"


def hamming_distance(hash_a: str, hash_b: str) -> int:
    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()


class DuplicateSignal:
    name = "duplicates"

    def run(self, ctx: ImageContext, config: AnalysisConfig) -> SignalResult:
        result = SignalResult(self.name)
        sha256 = hashlib.sha256(ctx.data).hexdigest()
        phash = perceptual_hash(ctx.rgb)
        result.features["sha256"] = sha256
        result.features["perceptual_hash"] = phash
        result.details["nearest_duplicate"] = None

        known = list(ctx.known_hashes or [])
        if known:
            distances = [(candidate, hamming_distance(phash, candidate)) for candidate in known]
            nearest = min(distances, key=lambda item: item[1])
            result.details["nearest_duplicate"] = {"hash": nearest[0], "distance": nearest[1]}
            result.features["nearest_distance"] = nearest[1]
            if nearest[1] <= config.duplicate_hamming_distance:
                result.add(finding("near_duplicate", "Imagem igual ou muito semelhante a caso anterior", 35,
                                   f"Distância perceptual: {nearest[1]}.", "alto"))
        return result
