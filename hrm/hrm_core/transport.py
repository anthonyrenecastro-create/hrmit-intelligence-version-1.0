from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransportRecord:
    block: str
    mode: str
    before_shape: tuple[int, ...]
    after_shape: tuple[int, ...]
    discarded_norm: float
    introduced_norm: float


def identity_transport(value: np.ndarray) -> np.ndarray:
    return np.array(value, copy=True)


def record_identity_transport(block: str, value: np.ndarray) -> tuple[np.ndarray, TransportRecord]:
    transported = identity_transport(value)
    record = TransportRecord(
        block=block,
        mode="identity",
        before_shape=tuple(value.shape),
        after_shape=tuple(transported.shape),
        discarded_norm=0.0,
        introduced_norm=0.0,
    )
    return transported, record
