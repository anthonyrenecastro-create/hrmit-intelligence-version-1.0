from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
