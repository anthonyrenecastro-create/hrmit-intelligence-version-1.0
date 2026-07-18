from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .events import StructuralEvent
from .proposals import TransitionCorrection
from .transport import TransportRecord


@dataclass(frozen=True)
class TransitionLedger:
    source_version: int
    target_version: int
    proposal_activations: dict[str, float]
    proposed_phi_contributions: dict[str, np.ndarray]
    applied_phi_contributions: dict[str, np.ndarray]
    rejected_phi_contributions: dict[str, np.ndarray]
    proposed_memory_contributions: dict[str, np.ndarray]
    applied_memory_contributions: dict[str, np.ndarray]
    rejected_memory_contributions: dict[str, np.ndarray]
    proposed_cognition_contributions: dict[str, np.ndarray]
    applied_cognition_contributions: dict[str, np.ndarray]
    rejected_cognition_contributions: dict[str, np.ndarray]
    proposed_hierarchy_contributions: dict[str, np.ndarray]
    applied_hierarchy_contributions: dict[str, np.ndarray]
    rejected_hierarchy_contributions: dict[str, np.ndarray]
    proposed_events: tuple[StructuralEvent, ...]
    accepted_events: tuple[StructuralEvent, ...]
    rejected_events: tuple[StructuralEvent, ...]
    transport_records: tuple[TransportRecord, ...]
    corrections: tuple[TransitionCorrection, ...]
    residual: np.ndarray
    block_residuals: dict[str, np.ndarray]
    max_abs_reconstruction_error: float
    max_rel_reconstruction_error: float
    runtime_seconds: float
    rng_digest_before: str
    rng_digest_after: str
    metrics: dict[str, Any] = field(default_factory=dict)


def reconstruct_phi_delta(ledger: TransitionLedger) -> np.ndarray:
    if not ledger.applied_phi_contributions:
        return np.zeros_like(ledger.residual)
    first = next(iter(ledger.applied_phi_contributions.values()))
    total = np.zeros_like(first)
    for value in ledger.applied_phi_contributions.values():
        total = total + value
    total = total + ledger.residual
    return total
