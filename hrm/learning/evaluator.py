from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .types import EvaluationReport


@dataclass(frozen=True)
class EvaluationConfig:
    primary_metric: str = "held_out_accuracy"
    minimum_improvement: float = 0.01
    regression_threshold: float = 0.01
    safety_threshold: float = 0.0
    validation_size: int = 5


class Evaluator:
    def evaluate_parent_and_candidate(
        self,
        parent_checkpoint: dict[str, Any],
        candidate_checkpoint: dict[str, Any],
        held_out_data: list[dict[str, Any]],
        config: EvaluationConfig,
    ) -> EvaluationReport:
        parent_score = self._compute_score(parent_checkpoint, held_out_data, config.primary_metric)
        candidate_score = self._compute_score(candidate_checkpoint, held_out_data, config.primary_metric)
        improvement = candidate_score - parent_score
        regression = self._compute_regression(parent_checkpoint, candidate_checkpoint)
        safety = self._compute_safety(candidate_checkpoint)
        accepted = improvement >= config.minimum_improvement and regression >= -config.regression_threshold and safety >= config.safety_threshold
        reasons: list[str] = []
        if improvement < config.minimum_improvement:
            reasons.append("insufficient_primary_improvement")
        if regression < -config.regression_threshold:
            reasons.append("regression_threshold_exceeded")
        if safety < config.safety_threshold:
            reasons.append("safety_threshold_failed")
        return EvaluationReport(
            candidate_id=candidate_checkpoint.get("candidate_id", "unknown"),
            primary_metrics={config.primary_metric: candidate_score},
            regression_metrics={"delta": improvement},
            safety_metrics={"safety_score": safety},
            calibration_metrics={"calibration_error": 0.0},
            accepted=accepted,
            rejection_reasons=tuple(reasons),
            metadata={
                "parent_score": parent_score,
                "candidate_score": candidate_score,
                "minimum_improvement": config.minimum_improvement,
            },
        )

    def _compute_score(self, checkpoint: dict[str, Any], held_out_data: list[dict[str, Any]], metric: str) -> float:
        base = float(checkpoint.get(metric, 0.0))
        bias = sum(float(item.get("difficulty", 1.0)) for item in held_out_data) * 0.001
        return base + bias

    def _compute_regression(self, parent_checkpoint: dict[str, Any], candidate_checkpoint: dict[str, Any]) -> float:
        parent_recall = float(parent_checkpoint.get("prior_task_recall", 1.0))
        candidate_recall = float(candidate_checkpoint.get("prior_task_recall", parent_recall))
        return candidate_recall - parent_recall

    def _compute_safety(self, candidate_checkpoint: dict[str, Any]) -> float:
        return 1.0 - min(1.0, abs(float(candidate_checkpoint.get("safety_bias", 0.0))))


class CandidateEvaluator:
    def __init__(self, min_improvement: float = 0.05, max_regression: float = 0.02) -> None:
        self.min_improvement = float(min_improvement)
        self.max_regression = float(max_regression)

    @staticmethod
    def _accuracy(model, x: np.ndarray, y: np.ndarray) -> float:
        pred = np.asarray(model.predict(x)).astype(np.float32)
        target = np.asarray(y).astype(np.float32)
        return float(np.mean((pred > 0.5) == (target > 0.5)))

    def evaluate(
        self,
        candidate_id: str,
        parent_model,
        candidate_model,
        heldout: tuple[np.ndarray, np.ndarray],
        regression: tuple[np.ndarray, np.ndarray],
        update_norm: float,
        max_update_norm: float,
    ) -> EvaluationReport:
        heldout_x, heldout_y = heldout
        reg_x, reg_y = regression

        parent_heldout = self._accuracy(parent_model, heldout_x, heldout_y)
        candidate_heldout = self._accuracy(candidate_model, heldout_x, heldout_y)
        parent_reg = self._accuracy(parent_model, reg_x, reg_y)
        candidate_reg = self._accuracy(candidate_model, reg_x, reg_y)

        improvement = candidate_heldout - parent_heldout
        accuracy_drop = max(0.0, parent_reg - candidate_reg)
        accepted = improvement >= self.min_improvement and accuracy_drop <= self.max_regression and update_norm <= max_update_norm + 1e-6

        reasons = []
        if improvement < self.min_improvement:
            reasons.append("insufficient_heldout_improvement")
        if accuracy_drop > self.max_regression:
            reasons.append("regression_threshold_exceeded")
        if update_norm > max_update_norm + 1e-6:
            reasons.append("update_norm_exceeded")

        return EvaluationReport(
            candidate_id=candidate_id,
            primary_metrics={"accuracy_improvement": float(improvement)},
            regression_metrics={"accuracy_drop": float(accuracy_drop)},
            safety_metrics={"update_norm": float(update_norm)},
            calibration_metrics={"calibration_error": 0.0},
            accepted=accepted,
            rejection_reasons=tuple(reasons),
            metadata={
                "parent_heldout": float(parent_heldout),
                "candidate_heldout": float(candidate_heldout),
                "parent_regression": float(parent_reg),
                "candidate_regression": float(candidate_reg),
            },
        )
