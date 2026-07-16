from __future__ import annotations

import numpy as np

from hrm.multimodal.types import ModalityRepresentation


class HRMProjector:
    def __init__(self, target_dim: int = 64, target_component: str = "observation") -> None:
        self.target_dim = target_dim
        self.target_component = target_component

    def project(self, representation: ModalityRepresentation, target_component: str | None = None) -> dict[str, object]:
        latent = np.asarray(representation.latent, dtype=np.float32).ravel()
        projected = latent[: self.target_dim]
        if projected.size < self.target_dim:
            projected = np.pad(projected, (0, self.target_dim - projected.size), mode="constant")
        norm = np.linalg.norm(projected)
        normalized = projected / (norm + 1e-9)
        return {
            "modality": representation.modality,
            "source_id": representation.source_id,
            "projected": normalized,
            "projected_shape": normalized.shape,
            "original_shape": tuple(representation.latent.shape),
            "confidence": representation.confidence,
            "mask": representation.mask,
            "encoder_name": representation.encoder_name,
            "target_component": target_component or self.target_component,
            "projection_parameters": {
                "target_dim": self.target_dim,
                "normalization": "l2",
                "component": target_component or self.target_component,
            },
            "metadata": representation.metadata,
        }
