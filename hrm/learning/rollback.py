from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RollbackEvent:
    rollback_id: str
    candidate_id: str
    restored_checkpoint_id: str
    reason: str
    timestamp: float
    metadata: dict[str, Any]


class RollbackManager:
    def __init__(self) -> None:
        self.events: list[RollbackEvent] = []

    def rollback(self, active_checkpoint: dict[str, Any], prior_checkpoint: dict[str, Any], candidate_id: str, reason: str) -> dict[str, Any]:
        restored = dict(prior_checkpoint)
        event = RollbackEvent(
            rollback_id=f"rb_{candidate_id}_{int(__import__("time").time())}",
            candidate_id=candidate_id,
            restored_checkpoint_id=restored.get("checkpoint_id", "unknown"),
            reason=reason,
            timestamp=__import__("time").time(),
            metadata={},
        )
        self.events.append(event)
        return restored
