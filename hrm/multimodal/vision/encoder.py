from __future__ import annotations

import numpy as np

from hrm.multimodal.types import DecodedModality, ModalityRepresentation


class VisionEncoder:
    def __init__(self, latent_dim: int = 64, patch_size: tuple[int, int] = (8, 8)) -> None:
        self.latent_dim = latent_dim
        self.patch_size = patch_size

    def encode(self, decoded: DecodedModality) -> ModalityRepresentation:
        tensor = np.asarray(decoded.tensor, dtype=np.float32)
        if tensor.ndim == 3 and tensor.shape[0] in {1, 3}:
            tensor = np.moveaxis(tensor, 0, -1)
        if tensor.ndim == 2:
            tensor = tensor[..., None]

        image = tensor
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        if image.max() > 1.0:
            image = image / 255.0

        height, width, channels = image.shape[:3]
        patch_h, patch_w = self.patch_size
        patch_values: list[float] = []
        for y in range(0, height, patch_h):
            for x in range(0, width, patch_w):
                patch = image[y : y + patch_h, x : x + patch_w, :]
                patch_values.append(float(np.mean(patch)))

        gradient_y, gradient_x = np.gradient(image[..., 0])
        gradient_energy = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
        stats = np.array(
            [
                float(np.mean(image)),
                float(np.std(image)),
                float(np.mean(gradient_energy)),
                float(np.std(gradient_energy)),
            ],
            dtype=np.float32,
        )
        latent = np.asarray(patch_values + stats.tolist(), dtype=np.float32)
        if latent.size < self.latent_dim:
            latent = np.pad(latent, (0, self.latent_dim - latent.size), mode="constant")
        else:
            latent = latent[: self.latent_dim]

        confidence = float(min(1.0, 0.2 + float(np.clip(np.max(image) - np.min(image), 0.0, 0.8))))
        return ModalityRepresentation(
            modality="image",
            source_id=decoded.source_id,
            latent=latent.astype(np.float32),
            confidence=confidence,
            mask=decoded.mask,
            timestamp=decoded.timestamp,
            encoder_name="patch_mean_gradient_encoder",
            metadata={**decoded.metadata, "latent_dim": self.latent_dim, "patch_size": self.patch_size},
        )
