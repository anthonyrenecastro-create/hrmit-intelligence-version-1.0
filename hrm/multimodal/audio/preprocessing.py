from __future__ import annotations

import numpy as np
from scipy.signal import resample

from hrm.multimodal.types import DecodedModality


class AudioPreprocessor:
    def __init__(self, target_rate: int = 16000, target_duration: float = 2.0) -> None:
        self.target_rate = target_rate
        self.target_duration = target_duration

    def preprocess(self, decoded: DecodedModality) -> DecodedModality:
        tensor = np.asarray(decoded.tensor, dtype=np.float32)
        if decoded.metadata["sample_rate"] != self.target_rate:
            total_samples = int(self.target_duration * self.target_rate)
            tensor = resample(tensor, int(tensor.size * self.target_rate / decoded.metadata["sample_rate"]))
        target_samples = int(self.target_rate * self.target_duration)
        if tensor.size < target_samples:
            pad = np.zeros(target_samples - tensor.size, dtype=np.float32)
            mask = np.concatenate([np.ones(tensor.size, dtype=np.float32), np.zeros(pad.size, dtype=np.float32)])
            tensor = np.concatenate([tensor, pad])
        else:
            tensor = tensor[:target_samples]
            mask = np.ones(target_samples, dtype=np.float32)
        peak = np.max(np.abs(tensor)) if tensor.size else 1.0
        if peak > 0:
            tensor = tensor / peak
        return DecodedModality(
            modality=decoded.modality,
            source_id=decoded.source_id,
            tensor=tensor,
            mask=mask,
            shape=tensor.shape,
            dtype=str(tensor.dtype),
            timestamp=decoded.timestamp,
            metadata={**decoded.metadata, "target_rate": self.target_rate, "target_duration": self.target_duration},
        )
