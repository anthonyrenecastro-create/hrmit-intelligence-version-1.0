from __future__ import annotations

import numpy as np

from hrm.multimodal.types import DecodedModality, ModalityRepresentation


class StructuredEncoder:
    def __init__(self, latent_dim: int = 64) -> None:
        self.latent_dim = latent_dim

    def encode(self, decoded: DecodedModality) -> ModalityRepresentation:
        tensor = np.asarray(decoded.tensor, dtype=np.float32)
        mask = np.asarray(decoded.mask, dtype=np.float32) if decoded.mask is not None else np.ones_like(tensor, dtype=np.float32)
        if tensor.ndim != 2:
            tensor = tensor.reshape(1, -1)
            mask = mask.reshape(1, -1)
        output_mask = mask
        if mask.ndim == 2 and mask.shape[0] == 1:
            output_mask = mask[0].astype(bool)

        row_feature_mean = np.mean(tensor, axis=0)
        row_feature_std = np.std(tensor, axis=0)
        position_encoding = np.linspace(0.0, 1.0, num=tensor.shape[0], dtype=np.float32)
        position_summary = np.array([float(np.mean(position_encoding)), float(np.std(position_encoding))], dtype=np.float32)
        latent = np.concatenate([row_feature_mean, row_feature_std, position_summary], axis=0)
        if latent.size < self.latent_dim:
            latent = np.pad(latent, (0, self.latent_dim - latent.size), mode="constant")
        else:
            latent = latent[: self.latent_dim]

        mask_count = float(np.sum(mask > 0.0))
        mask_total = float(mask.size if mask.size else 1)
        confidence = mask_count / mask_total
        return ModalityRepresentation(
            modality="structured",
            source_id=decoded.source_id,
            latent=latent.astype(np.float32),
            confidence=confidence,
            mask=output_mask,
            timestamp=decoded.timestamp,
            encoder_name="schema_aware_summary_encoder",
            metadata={**decoded.metadata, "latent_dim": self.latent_dim, "record_count": decoded.metadata.get("record_count")},
        )
