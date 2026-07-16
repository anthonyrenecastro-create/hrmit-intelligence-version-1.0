from __future__ import annotations

import json
from pathlib import Path

from hrm.learning.experience import ExperienceStore
from hrm.learning.types import ExperienceRecord, FeedbackRecord, TaskOutcome


def test_experience_store_persists_and_loads(tmp_path: Path) -> None:
    outcome = TaskOutcome(
        task_id="task_001",
        task_type="classification",
        inputs={"text": "hello"},
        output="yes",
        expected_output="yes",
        success=True,
        score=1.0,
        completion_time=0.1,
        module_ids=("mod1",),
        tool_audit_ids=("tool1",),
        memory_ids=("mem1",),
        metadata={"phase": "test"},
    )
    feedback = FeedbackRecord.create(
        task_id=outcome.task_id,
        source="verifier",
        feedback_type="binary_success",
        value=True,
        confidence=1.0,
        scope="task",
        objective=True,
        suitable_for_training=True,
    )
    experience = ExperienceRecord.create(outcome, (feedback,), reward=1.0, priority=1.0)
    store = ExperienceStore()
    store.add(experience)

    path = tmp_path / "experience_store.json"
    store.save(path)

    loaded = ExperienceStore.load(path)
    assert loaded.get(experience.experience_id) is not None
    assert loaded.get(experience.experience_id).reward == 1.0
    assert loaded.get(experience.experience_id).feedback[0].feedback_id == feedback.feedback_id
    assert json.loads(path.read_text())
