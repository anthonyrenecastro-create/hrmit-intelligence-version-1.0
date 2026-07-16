from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import io

import numpy as np
from PIL import Image

from hrm.multimodal.types import DecodedModality, ModalityRepresentation


@dataclass(frozen=True)
class ExperimentalVideoAdapter:
    status: str = "experimental"
    max_frames: int = 16

    def decode(self, source: Any, source_id: str, timestamp: float | None = None) -> DecodedModality:
        if not isinstance(source, list):
            raise ValueError("Video source must be a list of image frames")
        frames: list[np.ndarray] = []
        for idx, frame_source in enumerate(source[: self.max_frames]):
            if isinstance(frame_source, (bytes, bytearray)):
                with Image.open(io.BytesIO(frame_source)) as image:
                    image = image.convert("RGB")
                    frames.append(np.asarray(image, dtype=np.float32))
            elif isinstance(frame_source, np.ndarray):
                frames.append(frame_source.astype(np.float32))
            else:
                raise ValueError("Video frames must be bytes or NumPy arrays")
        if not frames:
            raise ValueError("Video source contains no frames")
        tensor = np.stack(frames, axis=0)
        metadata = {"status": self.status, "frame_count": len(frames), "max_frames": self.max_frames}
        return DecodedModality(
            modality="video",
            source_id=source_id,
            tensor=tensor,
            mask=np.ones((tensor.shape[0],), dtype=np.float32),
            shape=tensor.shape,
            dtype=str(tensor.dtype),
            timestamp=timestamp,
            metadata=metadata,
        )

    def encode(self, decoded: DecodedModality) -> ModalityRepresentation:
        tensor = np.asarray(decoded.tensor, dtype=np.float32)
        frame_means = np.mean(tensor, axis=(1, 2, 3)) if tensor.ndim == 4 else np.mean(tensor, axis=(1, 2))
        latent = np.concatenate([frame_means, frame_means[::-1]])
        latent = latent.astype(np.float32)
        if latent.size < 64:
            latent = np.pad(latent, (0, 64 - latent.size), mode="constant")
        else:
            latent = latent[:64]
        confidence = float(min(1.0, float(np.mean(frame_means))))
        return ModalityRepresentation(
            modality="video",
            source_id=decoded.source_id,
            latent=latent,
            confidence=confidence,
            mask=decoded.mask,
            timestamp=decoded.timestamp,
            encoder_name="experimental_video_encoder",
            metadata={**decoded.metadata, "latent_dim": latent.size},
        )
