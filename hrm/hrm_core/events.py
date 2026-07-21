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

    add_nodes = int(metadata.get("topology_add_nodes", 0) or 0)
    remove_nodes = int(metadata.get("topology_remove_nodes", 0) or 0)
    rewire = bool(metadata.get("topology_rewire", False))
    topology_events = metadata.get("topology_events")

    if add_nodes > 0:
        events.append(
            StructuralEvent(
                event_id=f"topology_add_step_{context.step}",
                event_type="topology_add_nodes",
                block="T",
                accepted=False,
                reason="proposed_topology_add_nodes",
                priority=20,
                magnitude=float(add_nodes),
                metadata={"add_nodes": add_nodes},
            )
        )

    if remove_nodes > 0:
        events.append(
            StructuralEvent(
                event_id=f"topology_remove_step_{context.step}",
                event_type="topology_remove_nodes",
                block="T",
                accepted=False,
                reason="proposed_topology_remove_nodes",
                priority=19,
                magnitude=float(remove_nodes),
                metadata={"remove_nodes": remove_nodes},
            )
        )

    if rewire:
        events.append(
            StructuralEvent(
                event_id=f"topology_rewire_step_{context.step}",
                event_type="topology_rewire_ring",
                block="T",
                accepted=False,
                reason="proposed_topology_rewire",
                priority=18,
                magnitude=1.0,
                metadata={"rewire": True},
            )
        )

    if isinstance(topology_events, (list, tuple)):
        for index, item in enumerate(topology_events):
            if not isinstance(item, dict):
                continue
            event_type = str(item.get("event_type", "")).strip()
            if event_type not in {"topology_add_nodes", "topology_remove_nodes", "topology_rewire_ring"}:
                continue
            payload = dict(item)
            priority = int(payload.get("priority", 15))
            magnitude = float(payload.get("magnitude", 1.0))
            events.append(
                StructuralEvent(
                    event_id=f"topology_typed_{event_type}_step_{context.step}_{index}",
                    event_type=event_type,
                    block="T",
                    accepted=False,
                    reason="proposed_topology_typed_event",
                    priority=priority,
                    magnitude=magnitude,
                    metadata=payload,
                )
            )

    return tuple(sorted(events, key=lambda event: (-event.priority, event.event_id)))
