from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptationMetrics:
    held_out_accuracy: float
    prior_task_recall: float
    safety_score: float
    calibration_error: float
    update_norm: float
    rejection_rate: float


@dataclass(frozen=True)
class RegressionMetrics:
    prior_task_drop: float
    hard_failures: int
    soft_failures: int
