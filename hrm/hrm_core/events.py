from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .snapshot import HRMSnapshot
from .observables import HRMObservables
from .mechanisms.base import HRMInput, TransitionContext


@dataclass(frozen=True)
class StructuralEvent:
    event_id: str
    event_type: str
    block: str
    accepted: bool
    reason: str
    priority: int = 0
    magnitude: float = 0.0
    metadata: dict[str, Any] = None


def propose_events(
    snapshot: HRMSnapshot,
    external_input: HRMInput,
    observables: HRMObservables,
    context: TransitionContext,
) -> tuple[StructuralEvent, ...]:
    events: list[StructuralEvent] = []
    metadata = external_input.metadata or {}

    if metadata.get("memory_query") or metadata.get("memory_write_candidate"):
        events.append(
            StructuralEvent(
                event_id=f"memory_write_head_step_{context.step}",
                event_type="memory_write_head",
                block="M",
                accepted=False,
                reason="proposed_write_head_advance",
                priority=10,
                magnitude=float(np.linalg.norm(snapshot.state.memory.working)),
                metadata={"write_index": snapshot.state.memory.write_index, "capacity": snapshot.state.memory.capacity},
            )
        )

    if observables.collapse_risk > 0.0 or observables.field_variance < 1e-3:
        events.append(
            StructuralEvent(
                event_id=f"field_stability_step_{context.step}",
                event_type="field_stability_guard",
                block="Phi",
                accepted=False,
                reason="low_variance_or_collapse_risk",
                priority=1,
                magnitude=observables.collapse_risk,
                metadata={"field_variance": observables.field_variance, "field_norm": observables.field_norm},
            )
        )

    return tuple(sorted(events, key=lambda event: (-event.priority, event.event_id)))
