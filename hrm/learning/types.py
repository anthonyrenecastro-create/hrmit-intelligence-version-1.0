from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    task_type: str
    inputs: dict[str, Any]
    output: Any
    expected_output: Any | None
    success: bool
    score: float | None
    completion_time: float
    module_ids: tuple[str, ...] = ()
    tool_audit_ids: tuple[str, ...] = ()
    memory_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeedbackRecord:
    feedback_id: str
    task_id: str
    source: str
    feedback_type: str
    value: Any
    confidence: float
    timestamp: float
    scope: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperienceRecord:
    experience_id: str
    task_outcome: TaskOutcome
    feedback: tuple[FeedbackRecord, ...]
    reward: float
    priority: float
    replay_count: int
    created_at: float
    provenance: dict[str, Any]


@dataclass(frozen=True)
class AdaptationCandidate:
    candidate_id: str
    parent_checkpoint_id: str
    parameter_scope: tuple[str, ...]
    training_config: dict[str, Any]
    update_norm: float
    parameters: tuple[float, ...]
    created_at: float
    provenance_id: str


@dataclass(frozen=True)
class EvaluationReport:
    candidate_id: str
    primary_metrics: dict[str, float]
    regression_metrics: dict[str, float]
    safety_metrics: dict[str, float]
    calibration_metrics: dict[str, float]
    accepted: bool
    rejection_reasons: tuple[str, ...]
