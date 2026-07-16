from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def _now() -> float:
    return time.time()


class TaskStatus(str, Enum):
    pending = "pending"
    ready = "ready"
    running = "running"
    completed = "completed"
    blocked = "blocked"
    failed = "failed"


@dataclass(frozen=True)
class CapabilityAssessment:
    can_handle: bool
    score: float
    reasoning: str


@dataclass(frozen=True)
class CognitiveTask:
    task_id: str
    objective: str
    inputs: dict[str, Any]
    required_capabilities: frozenset[str]
    dependencies: tuple[str, ...]
    priority: int
    deadline: float | None
    retry_limit: int
    metadata: dict[str, Any]

    @classmethod
    def create(
        cls,
        objective: str,
        inputs: dict[str, Any] | None = None,
        required_capabilities: frozenset[str] | None = None,
        dependencies: tuple[str, ...] | None = None,
        priority: int = 50,
        deadline: float | None = None,
        retry_limit: int = 3,
        metadata: dict[str, Any] | None = None,
    ) -> "CognitiveTask":
        return cls(
            task_id=uuid.uuid4().hex,
            objective=objective,
            inputs=inputs or {},
            required_capabilities=required_capabilities or frozenset(),
            dependencies=dependencies or (),
            priority=priority,
            deadline=deadline,
            retry_limit=retry_limit,
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class ModuleCapability:
    name: str
    proficiency: float
    supported_input_types: tuple[str, ...]
    resource_requirements: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModuleProposal:
    proposal_id: str
    task_id: str
    module_id: str
    answer: Any
    confidence: float
    evidence: tuple[Any, ...]
    assumptions: tuple[str, ...]
    unresolved_questions: tuple[str, ...]
    memory_reads: tuple[str, ...]
    memory_writes: tuple[str, ...]
    execution_metadata: dict[str, Any]

    @classmethod
    def create(
        cls,
        task_id: str,
        module_id: str,
        answer: Any,
        confidence: float,
        evidence: tuple[Any, ...] | None = None,
        assumptions: tuple[str, ...] | None = None,
        unresolved_questions: tuple[str, ...] | None = None,
        memory_reads: tuple[str, ...] | None = None,
        memory_writes: tuple[str, ...] | None = None,
        execution_metadata: dict[str, Any] | None = None,
    ) -> "ModuleProposal":
        return cls(
            proposal_id=uuid.uuid4().hex,
            task_id=task_id,
            module_id=module_id,
            answer=answer,
            confidence=float(confidence),
            evidence=tuple(evidence or ()),
            assumptions=tuple(assumptions or ()),
            unresolved_questions=tuple(unresolved_questions or ()),
            memory_reads=tuple(memory_reads or ()),
            memory_writes=tuple(memory_writes or ()),
            execution_metadata=execution_metadata or {},
        )


@dataclass(frozen=True)
class DelegationRecord:
    task_id: str
    assigned_module_id: str
    assignment_reason: str
    capability_match: float
    dependency_status: str
    attempt: int
    timestamp: float

    @classmethod
    def create(
        cls,
        task_id: str,
        assigned_module_id: str,
        assignment_reason: str,
        capability_match: float,
        dependency_status: str,
        attempt: int,
    ) -> "DelegationRecord":
        return cls(
            task_id=task_id,
            assigned_module_id=assigned_module_id,
            assignment_reason=assignment_reason,
            capability_match=float(capability_match),
            dependency_status=dependency_status,
            attempt=attempt,
            timestamp=_now(),
        )


@dataclass(frozen=True)
class DisagreementRecord:
    disagreement_id: str
    task_id: str
    proposal_ids: tuple[str, ...]
    disagreement_type: str
    severity: float
    unresolved_points: tuple[str, ...]
    resolution_status: str
    selected_resolution: str | None

    @classmethod
    def create(
        cls,
        task_id: str,
        proposal_ids: tuple[str, ...],
        disagreement_type: str,
        severity: float,
        unresolved_points: tuple[str, ...] | None = None,
        resolution_status: str = "unresolved",
        selected_resolution: str | None = None,
    ) -> "DisagreementRecord":
        return cls(
            disagreement_id=uuid.uuid4().hex,
            task_id=task_id,
            proposal_ids=proposal_ids,
            disagreement_type=disagreement_type,
            severity=float(severity),
            unresolved_points=tuple(unresolved_points or ()),
            resolution_status=resolution_status,
            selected_resolution=selected_resolution,
        )


@dataclass(frozen=True)
class ConsensusResult:
    task_id: str
    selected_proposal_id: str | None
    selected_answer: Any | None
    calibrated_confidence: float
    supporting_proposals: tuple[str, ...]
    dissenting_proposals: tuple[str, ...]
    unresolved_disagreements: tuple[str, ...]
    verifier_status: str
    strategy: str
    diagnostics: dict[str, Any]
