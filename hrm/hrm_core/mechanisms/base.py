from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..observables import HRMObservables
from ..proposals import MechanismProposal
from ..snapshot import HRMSnapshot


@dataclass(frozen=True)
class TransitionContext:
    step: int
    dt: float
    deterministic: bool


@dataclass(frozen=True)
class HRMInput:
    field_drive: object
    target: object | None = None
    metadata: dict[str, object] | None = None


class HRMMechanism(Protocol):
    mechanism_id: str
    read_blocks: frozenset[str]
    write_blocks: frozenset[str]

    def propose(
        self,
        snapshot: HRMSnapshot,
        external_input: HRMInput,
        observables: HRMObservables,
        context: TransitionContext,
    ) -> MechanismProposal:
        ...
