from __future__ import annotations

import numpy as np


def normalize_tensor(tensor: np.ndarray, dtype: type = np.float32, range_min: float = 0.0, range_max: float = 1.0) -> np.ndarray:
    tensor = tensor.astype(dtype)
    tmin, tmax = tensor.min(), tensor.max()
    if tmax > tmin:
        tensor = (tensor - tmin) / (tmax - tmin)
    else:
        tensor = tensor - tmin
    return tensor * (range_max - range_min) + range_min


def add_batch_dim(tensor: np.ndarray) -> np.ndarray:
    if tensor.ndim == 3:
        return tensor[None, ...]
    return tensor


def channel_first(tensor: np.ndarray) -> np.ndarray:
    if tensor.ndim == 3 and tensor.shape[-1] in {1, 3}:
        return np.moveaxis(tensor, -1, 0)
    return tensor
