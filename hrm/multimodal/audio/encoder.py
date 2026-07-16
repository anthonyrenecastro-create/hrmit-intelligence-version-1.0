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
        mean_spectrum = np.mean(tensor, axis=1)
        std_spectrum = np.std(tensor, axis=1)
        latent = np.concatenate([mean_spectrum, std_spectrum], axis=0)
        if latent.size < self.latent_dim:
            latent = np.pad(latent, (0, self.latent_dim - latent.size), mode="constant")
        else:
            latent = latent[: self.latent_dim]

        energy = float(np.mean(np.abs(tensor)))
        mask_fraction = float(np.mean(decoded.mask)) if decoded.mask is not None else 1.0
        confidence = float(min(1.0, 0.1 + energy * 0.9 * mask_fraction))
        return ModalityRepresentation(
            modality="audio",
            source_id=decoded.source_id,
            latent=latent.astype(np.float32),
            confidence=confidence,
            mask=decoded.mask,
            timestamp=decoded.timestamp,
            encoder_name="mean_std_mel_encoder",
            metadata={**decoded.metadata, "latent_dim": self.latent_dim},
        )
