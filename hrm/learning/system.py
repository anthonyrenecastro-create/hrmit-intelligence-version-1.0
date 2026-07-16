from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import asdict
from typing import Any

import numpy as np

from .adapters import LinearAdapter
from .evaluator import CandidateEvaluator
from .types import AdaptationCandidate, EvaluationReport, ExperienceRecord, FeedbackRecord, TaskOutcome


class ControlledLearningSystem:
    def __init__(self, feature_dim: int, *, max_update_norm: float = 1.0,
                 min_improvement: float = .05, max_regression: float = .02) -> None:
        self.active = LinearAdapter(feature_dim)
        self.max_update_norm = max_update_norm
        self.evaluator = CandidateEvaluator(min_improvement, max_regression)
        self.checkpoint_id = self._checkpoint_id(self.active.parameters)
        self.history: list[dict[str, Any]] = []

    @staticmethod
    def _checkpoint_id(parameters: np.ndarray) -> str:
        return "checkpoint-" + hashlib.sha256(parameters.tobytes()).hexdigest()[:12]

    @staticmethod
    def capture(outcome: TaskOutcome, feedback: tuple[FeedbackRecord, ...]) -> ExperienceRecord:
        if any(item.task_id != outcome.task_id for item in feedback):
            raise ValueError("Feedback task IDs must match the outcome")
        reward = float(np.mean([float(item.value) for item in feedback
                                if item.feedback_type in {"binary_success", "numeric_score", "verifier_result"}])
                       if feedback else outcome.success)
        return ExperienceRecord("experience-" + uuid.uuid4().hex, outcome, feedback, reward,
                                abs(reward - .5) + (0 if outcome.success else .5), 0, time.time(),
                                {"task_id": outcome.task_id, "module_ids": outcome.module_ids,
                                 "feedback_ids": tuple(item.feedback_id for item in feedback)})

    def adapt(self, train: tuple[np.ndarray, np.ndarray], heldout: tuple[np.ndarray, np.ndarray],
              regression: tuple[np.ndarray, np.ndarray], *, learning_rate: float = .1,
              epochs: int = 50) -> tuple[AdaptationCandidate, EvaluationReport]:
        x, y = train
        parameters, norm = self.active.trained_candidate(x, y, learning_rate=learning_rate,
                                                          epochs=epochs, max_update_norm=self.max_update_norm)
        candidate_id, provenance_id = "candidate-" + uuid.uuid4().hex, "provenance-" + uuid.uuid4().hex
        candidate = AdaptationCandidate(candidate_id, self.checkpoint_id, ("linear_adapter",),
                                        {"learning_rate": learning_rate, "epochs": epochs,
                                         "train_examples": len(x)}, norm, tuple(float(v) for v in parameters),
                                        time.time(), provenance_id)
        report = self.evaluator.evaluate(candidate_id, self.active,
                                         LinearAdapter(self.active.feature_dim, parameters), heldout,
                                         regression, norm, self.max_update_norm)
        event = {"candidate": asdict(candidate), "evaluation": asdict(report),
                 "parent_checkpoint_id": self.checkpoint_id,
                 "decision": "accepted" if report.accepted else "rejected", "timestamp": time.time()}
        if report.accepted:
            self.active = LinearAdapter(self.active.feature_dim, parameters)
            self.checkpoint_id = self._checkpoint_id(self.active.parameters)
            event["promoted_checkpoint_id"] = self.checkpoint_id
        else:
            event["rollback_checkpoint_id"] = self.checkpoint_id
        self.history.append(event)
        return candidate, report

    def rollback(self, parameters: np.ndarray, checkpoint_id: str, reason: str) -> None:
        previous = self.checkpoint_id
        self.active = LinearAdapter(self.active.feature_dim, parameters)
        self.checkpoint_id = checkpoint_id
        self.history.append({"decision": "rollback", "from_checkpoint_id": previous,
                             "rollback_checkpoint_id": checkpoint_id, "reason": reason,
                             "timestamp": time.time()})
