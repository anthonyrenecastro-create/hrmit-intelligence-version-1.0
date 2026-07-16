from __future__ import annotations

import numpy as np

from hrm.multimodal.types import ModalityRepresentation


class VisionTasks:
    @staticmethod
    def classify_color_blob(representation: ModalityRepresentation) -> str:
        if representation.modality != "vision":
            raise ValueError("Vision task requires a vision representation")
        avg = float(np.mean(representation.latent))
        if avg > 0.6:
            return "bright"
        if avg > 0.3:
            return "medium"
        return "dark"

    @staticmethod
    def recall_pattern(representation: ModalityRepresentation, pattern_cache: dict[str, float]) -> bool:
        if representation.modality != "vision":
            raise ValueError("Vision task requires a vision representation")
        score = float(np.corrcoef(representation.latent, np.asarray(list(pattern_cache.values()), dtype=np.float32).ravel()[: representation.latent.size])[0, 1])
        return score > 0.8
