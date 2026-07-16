from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hrm.multimodal.types import DecodedModality, ModalityRepresentation


@dataclass(frozen=True)
class TextDecoder:
    def decode(self, source: Any, source_id: str, timestamp: float | None = None) -> DecodedModality:
        if isinstance(source, (bytes, bytearray)):
            text = source.decode("utf-8")
        elif isinstance(source, str):
            text = source
        else:
            raise ValueError("Text source must be bytes or string")

        tensor = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.float32)
        return DecodedModality(
            modality="text",
            source_id=source_id,
            tensor=tensor,
            mask=np.ones_like(tensor, dtype=np.float32),
            shape=tensor.shape,
            dtype=str(tensor.dtype),
            timestamp=timestamp,
            metadata={"length": len(text), "chars": text[:32]},
        )


@dataclass(frozen=True)
class TextEncoder:
    latent_dim: int = 64

    def encode(self, decoded: DecodedModality) -> ModalityRepresentation:
        tensor = np.asarray(decoded.tensor, dtype=np.float32)
        counts = np.bincount(tensor.astype(np.int64), minlength=256).astype(np.float32)
        latent = counts[: self.latent_dim]
        if latent.size < self.latent_dim:
            latent = np.pad(latent, (0, self.latent_dim - latent.size), mode="constant")
        latent = latent / (np.linalg.norm(latent) + 1e-9)
        confidence = float(min(1.0, 0.2 + np.mean(latent)))
        return ModalityRepresentation(
            modality="text",
            source_id=decoded.source_id,
            latent=latent,
            confidence=confidence,
            mask=decoded.mask,
            timestamp=decoded.timestamp,
            encoder_name="text_token_histogram",
            metadata={**decoded.metadata, "latent_dim": self.latent_dim},
        )
