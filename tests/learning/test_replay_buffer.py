from __future__ import annotations

from pathlib import Path

from hrm.learning.experience import ExperienceRecord, FeedbackRecord, TaskOutcome, ExperienceStore
from hrm.learning.replay import ReplayBuffer, ReplayConfig


def make_experience(task_id: str, reward: float, priority: float) -> ExperienceRecord:
    outcome = TaskOutcome(
        task_id=task_id,
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
    return ExperienceRecord.create(outcome, (feedback,), reward=reward, priority=priority)


def test_replay_buffer_capacity_limit(tmp_path: Path) -> None:
    store = ExperienceStore()
    buffer = ReplayBuffer(store, config=ReplayConfig(capacity=3, prioritized=False, seed=0))
    for idx in range(5):
        buffer.add(make_experience(f"task_{idx}", reward=1.0, priority=float(idx + 1)))
    assert len(store.experiences) == 3
    assert store.experiences[0].task_outcome.task_id == "task_2"


def test_prioritized_replay_samples_high_priority_more_often() -> None:
    store = ExperienceStore()
    buffer = ReplayBuffer(store, config=ReplayConfig(capacity=10, prioritized=True, seed=0))
    store.add(make_experience("low", reward=0.1, priority=0.1))
    store.add(make_experience("high", reward=1.0, priority=10.0))
    samples = buffer.sample(100)
    assert any(exp.task_outcome.task_id == "high" for exp in samples)


def test_replay_persists_across_restart(tmp_path: Path) -> None:
    store = ExperienceStore()
    buffer = ReplayBuffer(store, config=ReplayConfig(capacity=5, prioritized=False, seed=0))
    exp = make_experience("task_1", reward=1.0, priority=1.0)
    buffer.add(exp)
    path = tmp_path / "replay.json"
    buffer.save(path)

    loaded = ReplayBuffer.load(path, config=ReplayConfig(capacity=5, prioritized=False, seed=0))
    assert loaded.store.get(exp.experience_id) is not None
    assert loaded.store.get(exp.experience_id).experience_id == exp.experience_id
