from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hrm.distributed.types import CognitiveTask, DelegationRecord, ModuleCapability


@dataclass(frozen=True)
class Assignment:
    task: CognitiveTask
    module: "SpecialistModule"
    score: float
    reason: str


class Scheduler:
    def __init__(self, capability_weight: float = 1.5, load_weight: float = 1.0, confidence_weight: float = 1.0) -> None:
        self.capability_weight = capability_weight
        self.load_weight = load_weight
        self.confidence_weight = confidence_weight

    def assign(self, task: CognitiveTask, modules: list["SpecialistModule"], state: Any) -> Assignment | None:
        available = [module for module in modules if module.can_handle(task).can_handle]
        if not available:
            return None

        scored: list[tuple[float, str, Assignment]] = []
        for module in available:
            assessment = module.can_handle(task)
            capability_score = assessment.score
            load_penalty = float(module.workload or 0) * 0.1
            confidence_bonus = float(module.historical_reliability or 0.5)
            score = capability_score * self.capability_weight + confidence_bonus * self.confidence_weight - load_penalty * self.load_weight
            reason = (
                f"Capability match {capability_score:.2f}, "
                f"availability {1.0 - load_penalty:.2f}, "
                f"reliability {confidence_bonus:.2f}"
            )
            scored.append((score, module.module_id, Assignment(task, module, score, reason)))

        scored.sort(key=lambda item: item[0], reverse=True)
        assignment = scored[0][2]
        return assignment
