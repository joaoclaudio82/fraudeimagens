"""Integridade (SHA-256) e duplicidade exata ou aproximada contra um índice de hashes."""

from __future__ import annotations

import hashlib

from ..config import AnalysisConfig
from ..context import ImageContext
from ..hashing import as_index, dhash, hamming_distance, phash
from .base import SignalResult, finding

# Compatibilidade com o nome usado pela primeira versão da PoC.
perceptual_hash = dhash

__all__ = ["DuplicateSignal", "perceptual_hash", "hamming_distance"]


class DuplicateSignal:
    name = "duplicates"

    def run(self, ctx: ImageContext, config: AnalysisConfig) -> SignalResult:
        result = SignalResult(self.name)
        sha256 = hashlib.sha256(ctx.data).hexdigest()
        dhash_hex = dhash(ctx.rgb)
        phash_hex = phash(ctx.rgb)
        result.features["sha256"] = sha256
        result.features["perceptual_hash"] = dhash_hex
        result.features["phash"] = phash_hex
        result.features["exact_duplicate"] = False
        result.details["nearest_duplicate"] = None

        index = as_index(ctx.known_hashes)
        if index is None:
            return result

        exact = index.find_exact(sha256)
        if exact:
            first = exact[0]
            result.features["exact_duplicate"] = True
            result.details["exact_duplicates"] = exact[:5]
            reference = first.get("reference") or first.get("filename") or "registro anterior"
            result.add(finding("exact_duplicate", "Arquivo idêntico já analisado", 45,
                               f"SHA-256 igual ao de '{reference}' em {first.get('created_at', '?')}.", "alto"))

        nearest = index.find_near(dhash_hex, phash_hex, config.duplicate_hamming_distance, config.duplicate_dct_distance)
        if nearest is None:
            return result
        match = nearest.get("match") or {}
        result.details["nearest_duplicate"] = {
            "hash": match.get("dhash"),
            "distance": nearest["distance"],
            "dhash_distance": nearest.get("dhash_distance"),
            "phash_distance": nearest.get("phash_distance"),
            "reference": match.get("reference"),
            "filename": match.get("filename"),
            "created_at": match.get("created_at"),
        }
        result.features["nearest_distance"] = nearest["distance"]
        if nearest.get("within") and not exact:
            reference = match.get("reference") or match.get("filename") or "caso anterior"
            result.add(finding("near_duplicate", "Imagem igual ou muito semelhante a caso anterior", 35,
                               f"Distância perceptual {nearest['distance']} (dHash {nearest.get('dhash_distance')}, "
                               f"pHash {nearest.get('phash_distance')}) em relação a '{reference}'.", "alto"))
        return result
