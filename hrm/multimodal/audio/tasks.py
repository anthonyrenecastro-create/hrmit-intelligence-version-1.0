from __future__ import annotations

import numpy as np

from hrm.multimodal.types import ModalityRepresentation


class AudioTasks:
    @staticmethod
    def classify_tone(representation: ModalityRepresentation) -> str:
        if representation.modality != "audio":
            raise ValueError("Audio task requires an audio representation")
        mean_val = float(np.mean(representation.latent))
        return "high" if mean_val > 0.2 else "low"

    @staticmethod
    def recall_sequence(representation: ModalityRepresentation, reference: np.ndarray) -> bool:
        if representation.modality != "audio":
            raise ValueError("Audio task requires an audio representation")
        latent = representation.latent
        ref = reference.astype(np.float32)
        return float(np.corrcoef(latent[: ref.size], ref)[0, 1]) > 0.7
