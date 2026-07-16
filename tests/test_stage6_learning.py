from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from hrm.learning import ControlledLearningSystem, FeedbackRecord, TaskOutcome
from hrm.learning.experience import ExperienceStore
from hrm.learning.replay import ReplayBuffer


def dataset(seed: int, count: int = 200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(count, 2)).astype(np.float32)
    y = (x[:, 0] + .7 * x[:, 1] > 0).astype(np.float32)
    return x, y


def test_feedback_capture_persistence_and_prioritized_replay(tmp_path: Path) -> None:
    system = ControlledLearningSystem(2)
    records = []
    for index, success in enumerate((True, False, True, False)):
        outcome = TaskOutcome(f"task-{index}", "classification", {"x": [index, 1]}, int(success),
                              int(success), success, float(success), time.time())
        feedback = FeedbackRecord(f"feedback-{index}", outcome.task_id, "verifier", "binary_success",
                                  float(success), 1.0, time.time(), "linear_adapter")
        records.append(system.capture(outcome, (feedback,)))
    store = ExperienceStore()
    for record in records: store.add(record)
    path = tmp_path / "experiences.json"; store.save(path)
    loaded = ExperienceStore.load(path)
    assert len(loaded.all()) == 4
    assert {record.task_outcome.success for record in ReplayBuffer(loaded.all()).sample(4, "balanced")} == {True, False}


def test_candidate_improves_heldout_and_is_promoted_without_regression() -> None:
    train, heldout, regression = dataset(1), dataset(2), dataset(3)
    system = ControlledLearningSystem(2, max_update_norm=3.0, min_improvement=.3)
    parent = system.checkpoint_id
    candidate, report = system.adapt(train, heldout, regression, learning_rate=.2, epochs=80)
    assert report.accepted
    assert report.primary_metrics["accuracy_improvement"] >= .3
    assert report.regression_metrics["accuracy_drop"] <= .02
    assert candidate.update_norm <= 3.0
    assert system.checkpoint_id != parent
    assert system.history[-1]["decision"] == "accepted"


def test_failed_candidate_is_rejected_and_active_parameters_rollback_implicitly() -> None:
    train, heldout, regression = dataset(4), dataset(5), dataset(6)
    system = ControlledLearningSystem(2, max_update_norm=.05, min_improvement=.99)
    before = system.active.parameters.copy(); checkpoint = system.checkpoint_id
    candidate, report = system.adapt(train, heldout, regression, learning_rate=1.0, epochs=100)
    assert not report.accepted
    assert "insufficient_heldout_improvement" in report.rejection_reasons
    assert np.array_equal(system.active.parameters, before)
    assert system.checkpoint_id == checkpoint
    assert system.history[-1]["rollback_checkpoint_id"] == checkpoint
    assert candidate.update_norm <= .05 + 1e-6


def test_explicit_rollback_restores_prior_checkpoint() -> None:
    train, heldout, regression = dataset(7), dataset(8), dataset(9)
    system = ControlledLearningSystem(2, max_update_norm=3.0, min_improvement=.3)
    original, checkpoint = system.active.parameters.copy(), system.checkpoint_id
    _, report = system.adapt(train, heldout, regression, learning_rate=.2, epochs=80)
    assert report.accepted
    system.rollback(original, checkpoint, "post-promotion regression")
    assert np.array_equal(system.active.parameters, original)
    assert system.history[-1]["reason"] == "post-promotion regression"
