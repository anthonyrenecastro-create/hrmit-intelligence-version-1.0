from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BlockDelta:
    phi: np.ndarray | None = None
    memory: np.ndarray | None = None
    cognition_latent: np.ndarray | None = None
    hierarchy_coarse: np.ndarray | None = None


@dataclass(frozen=True)
class MechanismProposal:
    mechanism_id: str
    source_state_version: int
    read_blocks: frozenset[str]
    write_blocks: frozenset[str]
    activation: float
    delta: BlockDelta
    dtype: str
    device: str
    estimated_cost: float
    uncertainty: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    validation_status: str = "valid"


@dataclass(frozen=True)
class TransitionCorrection:
    correction_id: str
    reason: str
    block: str
    magnitude: float
    metadata: dict[str, Any] = field(default_factory=dict)
