from __future__ import annotations

import numpy as np

from hrm.multimodal.types import DecodedModality, ModalityRepresentation


class VisionEncoder:
    def __init__(self, latent_dim: int = 64) -> None:
        self.latent_dim = latent_dim
        self.weights = np.linspace(0.1, 1.0, num=self.latent_dim, dtype=np.float32)

    def encode(self, decoded: DecodedModality) -> ModalityRepresentation:
        tensor = np.asarray(decoded.tensor, dtype=np.float32)
        flattened = tensor.ravel()
        if flattened.size == 0:
            latent = np.zeros(self.latent_dim, dtype=np.float32)
        else:
            patch_sum = flattened.reshape(-1, flattened.size // self.latent_dim or 1).mean(axis=1)
            latent = patch_sum[: self.latent_dim]
            if latent.size < self.latent_dim:
                latent = np.pad(latent, (0, self.latent_dim - latent.size), mode="constant")
        latent = latent * self.weights
        confidence = float(min(1.0, np.mean(tensor) * 1.2 + 0.1))
        return ModalityRepresentation(
            modality="vision",
            source_id=decoded.source_id,
            latent=latent.astype(np.float32),
            confidence=confidence,
            mask=None,
            timestamp=decoded.timestamp,
            encoder_name="light_conv_baseline",
            metadata={**decoded.metadata, "latent_dim": self.latent_dim},
        )
