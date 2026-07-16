from __future__ import annotations

from hrm.learning.feedback import FeedbackCapture
from hrm.learning.types import FeedbackRecord, TaskOutcome


def make_outcome() -> TaskOutcome:
    return TaskOutcome(
        task_id="task_001",
        task_type="classification",
        inputs={"text": "a"},
        output="b",
        expected_output="b",
        success=True,
        score=1.0,
        completion_time=0.0,
        module_ids=("mod1",),
        tool_audit_ids=("tool1",),
        memory_ids=("mem1",),
        metadata={"phase": "test"},
    )


def test_feedback_is_captured_after_task_completion() -> None:
    capture = FeedbackCapture()
    outcome = make_outcome()
    feedback = capture.capture_from_task(
        outcome,
        source="verifier",
        feedback_type="binary_success",
        value=True,
        confidence=0.9,
        scope="task",
        objective=True,
        suitable_for_training=True,
    )

    assert isinstance(feedback, FeedbackRecord)
    assert feedback.task_id == outcome.task_id
    assert feedback.source == "verifier"
    assert feedback.feedback_type == "binary_success"
    assert feedback.confidence == 0.9
    assert feedback.objective is True
    assert feedback.suitable_for_training is True


def test_duplicate_feedback_is_detected() -> None:
    capture = FeedbackCapture()
    outcome = make_outcome()
    feedback = capture.capture_from_task(
        outcome,
        source="verifier",
        feedback_type="binary_success",
        value=True,
        confidence=0.9,
        scope="task",
        objective=True,
        suitable_for_training=True,
    )
    duplicate = capture.capture_from_task(
        outcome,
        source="verifier",
        feedback_type="binary_success",
        value=True,
        confidence=0.9,
        scope="task",
        objective=True,
        suitable_for_training=True,
    )

    assert capture.verify_duplicate([feedback], duplicate) is True
