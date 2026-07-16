from __future__ import annotations

import numpy as np

from hrm.multimodal.types import DecodedModality, ModalityRepresentation


class StructuredEncoder:
    def __init__(self, latent_dim: int = 64) -> None:
        self.latent_dim = latent_dim

    def encode(self, decoded: DecodedModality) -> ModalityRepresentation:
        tensor = np.asarray(decoded.tensor, dtype=np.float32)
        row_summary = np.mean(tensor, axis=0) if tensor.ndim == 2 else np.array([np.mean(tensor)], dtype=np.float32)
        latent = row_summary[: self.latent_dim]
        if latent.size < self.latent_dim:
            latent = np.pad(latent, (0, self.latent_dim - latent.size), mode="constant")
        confidence = float(min(1.0, np.mean(decoded.mask) if decoded.mask is not None else 0.5 + 0.1))
        return ModalityRepresentation(
            modality="structured",
            source_id=decoded.source_id,
            latent=latent.astype(np.float32),
            confidence=confidence,
            mask=decoded.mask,
            timestamp=decoded.timestamp,
            encoder_name="schema_projection_baseline",
            metadata={**decoded.metadata, "latent_dim": self.latent_dim},
        )
