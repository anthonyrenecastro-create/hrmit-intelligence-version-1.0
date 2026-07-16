from __future__ import annotations

import numpy as np
from PIL import Image

from hrm.multimodal.types import DecodedModality
from hrm.multimodal.preprocessing import normalize_tensor, channel_first, add_batch_dim


class VisionPreprocessor:
    def __init__(self, target_size: tuple[int, int] = (64, 64), normalize: bool = True) -> None:
        self.target_size = target_size
        self.normalize = normalize

    def preprocess(self, decoded: DecodedModality) -> DecodedModality:
        tensor = decoded.tensor
        if tensor.ndim == 3 and tensor.shape[-1] == 1:
            tensor = np.repeat(tensor, 3, axis=-1)
        image = Image.fromarray(tensor.astype(np.uint8).squeeze())
        image = image.resize(self.target_size, Image.BILINEAR)
        tensor = np.asarray(image, dtype=np.float32)
        if tensor.ndim == 2:
            tensor = tensor[..., None]
        if self.normalize:
            tensor = normalize_tensor(tensor, dtype=np.float32, range_min=0.0, range_max=1.0)
        tensor = channel_first(tensor)
        return DecodedModality(
            modality=decoded.modality,
            source_id=decoded.source_id,
            tensor=tensor,
            mask=None,
            shape=tensor.shape,
            dtype=str(tensor.dtype),
            timestamp=decoded.timestamp,
            metadata={**decoded.metadata, "preprocessed": True, "target_size": self.target_size},
        )
