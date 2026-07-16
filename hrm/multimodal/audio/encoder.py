from __future__ import annotations

import numpy as np

from hrm.multimodal.types import DecodedModality, ModalityRepresentation


class AudioEncoder:
    def __init__(self, latent_dim: int = 64) -> None:
        self.latent_dim = latent_dim

    def encode(self, decoded: DecodedModality) -> ModalityRepresentation:
        tensor = np.asarray(decoded.tensor, dtype=np.float32)
        if tensor.ndim == 1:
            tensor = tensor[None, :]
        latent = np.mean(tensor, axis=1)
        if latent.size == 0:
            latent = np.zeros(self.latent_dim, dtype=np.float32)
        elif latent.size < self.latent_dim:
            latent = np.pad(latent, (0, self.latent_dim - latent.size), mode="constant")
        else:
            latent = latent[: self.latent_dim]
        confidence = float(min(1.0, np.median(np.abs(tensor)) + 0.2))
        return ModalityRepresentation(
            modality="audio",
            source_id=decoded.source_id,
            latent=latent.astype(np.float32),
            confidence=confidence,
            mask=decoded.mask,
            timestamp=decoded.timestamp,
            encoder_name="spectral_pool_baseline",
            metadata={**decoded.metadata, "latent_dim": self.latent_dim},
        )
