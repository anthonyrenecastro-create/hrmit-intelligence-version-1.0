from __future__ import annotations

import numpy as np

from .types import FusionResult, ModalityRepresentation


class ConfidenceFusion:
    def __init__(self, expected_modalities: tuple[str, ...] = ("vision", "audio", "structured"), contradiction_threshold: float = -0.25) -> None:
        self.expected_modalities = expected_modalities
        self.contradiction_threshold = contradiction_threshold

    def fuse(self, representations: list[ModalityRepresentation]) -> FusionResult:
        if not representations:
            raise ValueError("At least one representation is required")
        dimensions = {item.latent.shape for item in representations}
        if len(dimensions) != 1:
            raise ValueError("All modality latents must have the same shape")
        raw = np.asarray([max(0.0, item.confidence) for item in representations], np.float32)
        weights = raw / raw.sum() if raw.sum() else np.full_like(raw, 1.0 / len(raw))
        fused = sum(float(weight) * item.latent for weight, item in zip(weights, representations))
        contradictions = []
        for i, left in enumerate(representations):
            for right in representations[i + 1:]:
                denom = np.linalg.norm(left.latent) * np.linalg.norm(right.latent) + 1e-9
                similarity = float(np.dot(left.latent, right.latent) / denom)
                if similarity < self.contradiction_threshold:
                    contradictions.append({"left": left.modality, "right": right.modality,
                                           "cosine_similarity": similarity})
        modalities = {item.modality for item in representations}
        return FusionResult(
            fused.astype(np.float32),
            {item.modality: float(weight) for item, weight in zip(representations, weights)},
            {item.modality: float(item.confidence) for item in representations},
            tuple(name for name in self.expected_modalities if name not in modalities),
            tuple(contradictions), tuple(item.source_id for item in representations),
            {"method": "confidence_weighted_sum", "latent_dim": int(fused.size),
             "weight_sum": float(weights.sum())},
        )
