from __future__ import annotations

import numpy as np

from hrm.multimodal.types import FusionResult, ModalityRepresentation


class FusionEngine:
    def __init__(self, strategy: str = "confidence_weighted") -> None:
        self.strategy = strategy

    def fuse(self, representations: list[ModalityRepresentation]) -> FusionResult:
        modalities = [rep.modality for rep in representations]
        confidences = {rep.modality: float(rep.confidence) for rep in representations}
        weights = self._compute_weights(representations)
        fused = self._weighted_sum(representations, weights)
        missing = tuple()
        contradictions = self._detect_contradictions(representations)
        provenance = tuple(f"{rep.modality}:{rep.source_id}" for rep in representations)
        diagnostics = {
            "strategy": self.strategy,
            "latent_norms": {rep.modality: float(np.linalg.norm(rep.latent)) for rep in representations},
            "projected_modalities": modalities,
            "weight_sum": float(sum(weights.values())),
            "contradictions": contradictions,
        }
        return FusionResult(
            fused_latent=fused,
            modality_weights=weights,
            modality_confidences=confidences,
            missing_modalities=missing,
            contradictions=tuple(contradictions),
            provenance=provenance,
            diagnostics=diagnostics,
        )

    def _compute_weights(self, representations: list[ModalityRepresentation]) -> dict[str, float]:
        if not representations:
            return {}
        if self.strategy == "confidence_weighted":
            total_confidence = sum(max(rep.confidence, 0.0) for rep in representations)
            if total_confidence > 0:
                return {rep.modality: float(rep.confidence / total_confidence) for rep in representations}
            return {rep.modality: 1.0 / len(representations) for rep in representations}
        if self.strategy == "uniform":
            return {rep.modality: 1.0 / len(representations) for rep in representations}
        return {rep.modality: 1.0 / len(representations) for rep in representations}

    def _weighted_sum(self, representations: list[ModalityRepresentation], weights: dict[str, float]) -> np.ndarray:
        fused: np.ndarray | None = None
        for rep in representations:
            scaled = np.asarray(rep.latent, dtype=np.float32) * weights.get(rep.modality, 0.0)
            fused = scaled if fused is None else fused + scaled
        if fused is None:
            fused = np.zeros(1, dtype=np.float32)
        return fused

    def _detect_contradictions(self, representations: list[ModalityRepresentation]) -> list[dict[str, object]]:
        contradictions: list[dict[str, object]] = []
        if len(representations) < 2:
            return contradictions
        modalities = [rep.modality for rep in representations]
        if len(set(modalities)) != len(modalities):
            return contradictions
        scores = {rep.modality: float(np.mean(rep.latent)) for rep in representations}
        if max(scores.values()) - min(scores.values()) > 0.5:
            contradictions.append(
                {
                    "contradiction_id": "latent_mean_disagreement",
                    "modalities": tuple(modalities),
                    "contradiction_type": "latent_mean_disagreement",
                    "severity": float(max(scores.values()) - min(scores.values())),
                    "confidence": float(sum(rep.confidence for rep in representations) / len(representations)),
                    "evidence": {m: scores[m] for m in scores},
                    "resolution": None,
                }
            )
        return contradictions
