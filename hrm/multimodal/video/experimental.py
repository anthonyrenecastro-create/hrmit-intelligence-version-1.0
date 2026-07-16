from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hrm.multimodal.types import ModalityRepresentation


@dataclass(frozen=True)
class ExperimentalVideoAdapter:
    status: str = "experimental"

    def decode(self, source: Any, source_id: str) -> ModalityRepresentation:
        return ModalityRepresentation(
            modality="video",
            source_id=source_id,
            latent=__import__("numpy").zeros(1, dtype=__import__("numpy").float32),
            confidence=0.0,
            mask=None,
            timestamp=None,
            encoder_name="experimental_video_adapter",
            metadata={"status": self.status},
        )
