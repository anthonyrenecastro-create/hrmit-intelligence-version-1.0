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
    old_to_new_index_map: tuple[int, ...] | None = None
    new_to_old_index_map: tuple[int | None, ...] | None = None


def _identity_maps(size: int) -> tuple[tuple[int, ...], tuple[int | None, ...]]:
    mapping = tuple(range(size))
    return mapping, tuple(mapping)


def _new_to_old_map(old_to_new: tuple[int, ...], new_size: int) -> tuple[int | None, ...]:
    new_to_old: list[int | None] = [None] * new_size
    for old_index, new_index in enumerate(old_to_new):
        if new_index >= 0:
            new_to_old[new_index] = old_index
    return tuple(new_to_old)


def transport_with_index_map(
    block: str,
    value: np.ndarray,
    *,
    old_to_new_index_map: tuple[int, ...] | None = None,
    new_size: int | None = None,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, TransportRecord]:
    old_size = int(value.shape[0]) if value.ndim > 0 else 0
    if value.ndim == 0:
        transported = np.array(value, copy=True)
        record = TransportRecord(
            block=block,
            mode="scalar_copy",
            before_shape=tuple(value.shape),
            after_shape=tuple(transported.shape),
            discarded_norm=0.0,
            introduced_norm=0.0,
            old_to_new_index_map=None,
            new_to_old_index_map=None,
        )
        return transported, record

    if old_to_new_index_map is None:
        old_to_new_index_map, _ = _identity_maps(old_size)
    if new_size is None:
        new_size = max((idx for idx in old_to_new_index_map if idx >= 0), default=-1) + 1

    transported_shape = (int(new_size),) + tuple(value.shape[1:])
    transported = np.full(transported_shape, fill_value, dtype=value.dtype)
    discarded_norm = 0.0

    for old_index, new_index in enumerate(old_to_new_index_map):
        if new_index >= 0 and new_index < new_size:
            transported[new_index] = value[old_index]
        else:
            discarded_norm += float(np.linalg.norm(value[old_index]))

    new_to_old = _new_to_old_map(old_to_new_index_map, int(new_size))
    introduced_norm = 0.0
    for new_index, old_index in enumerate(new_to_old):
        if old_index is None:
            introduced_norm += float(np.linalg.norm(transported[new_index]))

    record = TransportRecord(
        block=block,
        mode="index_map",
        before_shape=tuple(value.shape),
        after_shape=tuple(transported.shape),
        discarded_norm=discarded_norm,
        introduced_norm=introduced_norm,
        old_to_new_index_map=tuple(int(idx) for idx in old_to_new_index_map),
        new_to_old_index_map=new_to_old,
    )
    return transported, record
