from __future__ import annotations

from typing import Any

from .types import FeedbackRecord, TaskOutcome


class FeedbackCapture:
    def capture_from_task(
        self,
        outcome: TaskOutcome,
        source: str,
        feedback_type: str,
        value: Any,
        confidence: float,
        scope: str,
        objective: bool,
        suitable_for_training: bool,
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackRecord:
        return FeedbackRecord.create(
            task_id=outcome.task_id,
            source=source,
            feedback_type=feedback_type,
            value=value,
            confidence=confidence,
            scope=scope,
            objective=objective,
            suitable_for_training=suitable_for_training,
            metadata=metadata,
        )

    def verify_duplicate(self, existing: list[FeedbackRecord], candidate: FeedbackRecord) -> bool:
        return any(
            feedback.feedback_id == candidate.feedback_id
            or (feedback.task_id == candidate.task_id and feedback.source == candidate.source and feedback.feedback_type == candidate.feedback_type)
            for feedback in existing
        )
