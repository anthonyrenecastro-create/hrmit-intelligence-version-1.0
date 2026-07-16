from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any

from hrm.distributed.types import CapabilityAssessment, CognitiveTask, ModuleCapability, ModuleProposal


@dataclass
class SpecialistModule:
    module_id: str
    role: str
    capabilities: tuple[ModuleCapability, ...]
    workload: float = 0.0
    historical_reliability: float = 0.7

    def can_handle(self, task: CognitiveTask) -> CapabilityAssessment:
        if not task.required_capabilities:
            return CapabilityAssessment(True, 0.5, "No specific capability requirements")

        best_score = 0.0
        reasons: list[str] = []
        for capability in self.capabilities:
            if capability.name in task.required_capabilities:
                best_score = max(best_score, capability.proficiency)
                reasons.append(f"Matched capability {capability.name} with proficiency {capability.proficiency:.2f}")
        can_handle = best_score >= 0.3
        reasoning = "; ".join(reasons) if reasons else "No matching capabilities found"
        return CapabilityAssessment(can_handle, best_score, reasoning)

    def execute(self, task: CognitiveTask, coordinator: Any) -> ModuleProposal:
        answer = f"{self.role} analysis for {task.objective}"
        confidence = min(1.0, 0.5 + 0.1 * len(task.dependencies) + 0.1 * self.historical_reliability)
        evidence = (f"module {self.module_id}", f"role {self.role}")
        rationale = f"Executed by {self.module_id} using capabilities {', '.join(cap.name for cap in self.capabilities)}"
        proposal = ModuleProposal.create(
            task_id=task.task_id,
            module_id=self.module_id,
            answer=answer,
            confidence=confidence,
            evidence=evidence,
            assumptions=(f"dependencies {task.dependencies}",),
            unresolved_questions=(),
            memory_reads=tuple(task.inputs.get("memory_reads", ())),
            memory_writes=tuple(task.inputs.get("memory_writes", ())),
            execution_metadata={"rationale": rationale, "role": self.role},
        )
        try:
            write_entries = task.inputs.get("memory_writes", ())
            for write_key in write_entries:
                coordinator.shared_memory.write(
                    record_id=write_key,
                    author_module_id=self.module_id,
                    task_id=task.task_id,
                    value={"proposal_id": proposal.proposal_id, "answer": proposal.answer},
                    expected_version=None,
                    evidence=("generated_by_execute",),
                    confidence=proposal.confidence,
                )
        except Exception:
            pass
        self.workload += 1
        return proposal


class RoleSpecialistModule(SpecialistModule):
    def __init__(self, role: str, memory: Any, historical_reliability: float = 0.7) -> None:
        super().__init__(module_id=f"agent_{role}", role=role, capabilities=tuple(), historical_reliability=historical_reliability)
        self.memory = memory

    def can_handle(self, task: CognitiveTask) -> CapabilityAssessment:
        base = super().can_handle(task)
        role_bonus = 0.15 if self.role in task.required_capabilities else 0.0
        return CapabilityAssessment(base.can_handle, min(1.0, base.score + role_bonus), base.reasoning)

    def execute(self, task: CognitiveTask, coordinator: Any) -> ModuleProposal:
        answer = f"{self.role} module recommendation for {task.objective}"
        confidence = min(1.0, 0.6 + 0.1 * self.historical_reliability)
        evidence = (f"role {self.role}",)
        rationale = f"Role-specific analysis for {self.role}"
        return ModuleProposal.create(
            task_id=task.task_id,
            module_id=self.module_id,
            answer=answer,
            confidence=confidence,
            evidence=evidence,
            assumptions=("role-specialized execution",),
            unresolved_questions=(),
            memory_reads=tuple(task.inputs.get("memory_reads", ())),
            memory_writes=tuple(task.inputs.get("memory_writes", ())),
            execution_metadata={"rationale": rationale, "role": self.role},
        )
