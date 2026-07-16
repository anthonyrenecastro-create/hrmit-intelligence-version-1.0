from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


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
    module_ids: tuple[str, ...]
    tool_audit_ids: tuple[str, ...]
    memory_ids: tuple[str, ...]
    metadata: dict[str, Any]

    @property
    def outcome_id(self) -> str:
        return _stable_id(f"{self.task_id}:{self.task_type}:{self.completion_time}")


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
    metadata: dict[str, Any]
    objective: bool
    suitable_for_training: bool

    @staticmethod
    def create(
        task_id: str,
        source: str,
        feedback_type: str,
        value: Any,
        confidence: float,
        scope: str,
        objective: bool,
        suitable_for_training: bool,
        metadata: dict[str, Any] | None = None,
    ) -> "FeedbackRecord":
        metadata = metadata or {}
        timestamp = time.time()
        feedback_id = _stable_id(f"{task_id}:{source}:{feedback_type}:{timestamp}")
        return FeedbackRecord(
            feedback_id=feedback_id,
            task_id=task_id,
            source=source,
            feedback_type=feedback_type,
            value=value,
            confidence=float(min(max(confidence, 0.0), 1.0)),
            timestamp=timestamp,
            scope=scope,
            metadata=metadata,
            objective=objective,
            suitable_for_training=suitable_for_training,
        )


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

    @staticmethod
    def create(
        task_outcome: TaskOutcome,
        feedback: tuple[FeedbackRecord, ...],
        reward: float,
        priority: float,
        provenance: dict[str, Any] | None = None,
    ) -> "ExperienceRecord":
        provenance = provenance or {}
        created_at = time.time()
        experience_id = _stable_id(f"{task_outcome.outcome_id}:{created_at}")
        return ExperienceRecord(
            experience_id=experience_id,
            task_outcome=task_outcome,
            feedback=feedback,
            reward=float(reward),
            priority=float(priority),
            replay_count=0,
            created_at=created_at,
            provenance=provenance,
        )


@dataclass(frozen=True)
class AdaptationCandidate:
    candidate_id: str
    parent_checkpoint_id: str
    parameter_scope: tuple[str, ...]
    training_config: dict[str, Any]
    update_norm: float
    checkpoint_path: str
    created_at: float
    provenance_id: str
    state: str
    rejected_reasons: tuple[str, ...] = field(default_factory=tuple)

    @staticmethod
    def create(
        parent_checkpoint_id: str,
        parameter_scope: tuple[str, ...],
        training_config: dict[str, Any],
        checkpoint_path: str,
        provenance_id: str,
    ) -> "AdaptationCandidate":
        created_at = time.time()
        candidate_id = _stable_id(f"{parent_checkpoint_id}:{checkpoint_path}:{created_at}")
        return AdaptationCandidate(
            candidate_id=candidate_id,
            parent_checkpoint_id=parent_checkpoint_id,
            parameter_scope=parameter_scope,
            training_config=training_config,
            update_norm=0.0,
            checkpoint_path=checkpoint_path,
            created_at=created_at,
            provenance_id=provenance_id,
            state="created",
        )


@dataclass(frozen=True)
class EvaluationReport:
    candidate_id: str
    primary_metrics: dict[str, float]
    regression_metrics: dict[str, float]
    safety_metrics: dict[str, float]
    calibration_metrics: dict[str, float]
    accepted: bool
    rejection_reasons: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AdaptationProvenance:
    provenance_id: str
    candidate_id: str
    parent_checkpoint_id: str
    active_checkpoint_before: str
    active_checkpoint_after: str | None
    experience_ids: tuple[str, ...]
    feedback_ids: tuple[str, ...]
    replay_strategy: str
    training_seed: int
    training_config: dict[str, Any]
    trainable_parameter_names: tuple[str, ...]
    update_metrics: dict[str, float]
    held_out_metrics: dict[str, float]
    regression_metrics: dict[str, float]
    safety_metrics: dict[str, float]
    promotion_decision: str
    rejection_reasons: tuple[str, ...]
    rollback_id: str | None
    created_at: float

    @staticmethod
    def create(
        candidate_id: str,
        parent_checkpoint_id: str,
        active_checkpoint_before: str,
        experience_ids: tuple[str, ...],
        feedback_ids: tuple[str, ...],
        replay_strategy: str,
        training_seed: int,
        training_config: dict[str, Any],
        trainable_parameter_names: tuple[str, ...],
        update_metrics: dict[str, float],
        held_out_metrics: dict[str, float],
        regression_metrics: dict[str, float],
        safety_metrics: dict[str, float],
        promotion_decision: str,
        rejection_reasons: tuple[str, ...],
        rollback_id: str | None = None,
    ) -> "AdaptationProvenance":
        created_at = time.time()
        provenance_id = _stable_id(f"{candidate_id}:{created_at}")
        return AdaptationProvenance(
            provenance_id=provenance_id,
            candidate_id=candidate_id,
            parent_checkpoint_id=parent_checkpoint_id,
            active_checkpoint_before=active_checkpoint_before,
            active_checkpoint_after=None,
            experience_ids=experience_ids,
            feedback_ids=feedback_ids,
            replay_strategy=replay_strategy,
            training_seed=training_seed,
            training_config=training_config,
            trainable_parameter_names=trainable_parameter_names,
            update_metrics=update_metrics,
            held_out_metrics=held_out_metrics,
            regression_metrics=regression_metrics,
            safety_metrics=safety_metrics,
            promotion_decision=promotion_decision,
            rejection_reasons=rejection_reasons,
            rollback_id=rollback_id,
            created_at=created_at,
        )
