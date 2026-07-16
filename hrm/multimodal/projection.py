from __future__ import annotations

import numpy as np

from hrm.multimodal.types import ModalityRepresentation


class HRMProjector:
    def __init__(self, target_dim: int = 64) -> None:
        self.target_dim = target_dim

    def project(self, representation: ModalityRepresentation) -> dict[str, object]:
        latent = representation.latent
        flattened = np.asarray(latent, dtype=np.float32).ravel()
        if flattened.size == 0:
            flattened = np.zeros(self.target_dim, dtype=np.float32)
        projected = flattened[: self.target_dim]
        if projected.size < self.target_dim:
            projected = np.pad(projected, (0, self.target_dim - projected.size), mode="constant")
        normalized = projected / (np.linalg.norm(projected) + 1e-9)
        return {
            "modality": representation.modality,
            "source_id": representation.source_id,
            "projected": normalized,
            "projected_shape": normalized.shape,
            "original_shape": representation.latent.shape,
            "confidence": representation.confidence,
            "mask": representation.mask,
            "encoder_name": representation.encoder_name,
            "metadata": representation.metadata,
        }
