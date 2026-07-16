from __future__ import annotations

import numpy as np

from .adapters import LinearAdapter
from .types import EvaluationReport


def _metrics(adapter: LinearAdapter, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    p = adapter.probabilities(x)
    accuracy = float(np.mean((p >= .5) == y))
    log_loss = float(-np.mean(y * np.log(p + 1e-8) + (1-y) * np.log(1-p + 1e-8)))
    brier = float(np.mean((p-y) ** 2))
    return {"accuracy": accuracy, "log_loss": log_loss, "brier": brier}


class CandidateEvaluator:
    def __init__(self, min_improvement: float = .05, max_regression: float = .02) -> None:
        self.min_improvement, self.max_regression = min_improvement, max_regression

    def evaluate(self, candidate_id: str, active: LinearAdapter, candidate: LinearAdapter,
                 heldout: tuple[np.ndarray, np.ndarray], regression: tuple[np.ndarray, np.ndarray],
                 update_norm: float, max_update_norm: float) -> EvaluationReport:
        hx, hy = heldout; rx, ry = regression
        before, after = _metrics(active, hx, hy), _metrics(candidate, hx, hy)
        reg_before, reg_after = _metrics(active, rx, ry), _metrics(candidate, rx, ry)
        improvement = after["accuracy"] - before["accuracy"]
        regression_drop = reg_before["accuracy"] - reg_after["accuracy"]
        reasons = []
        if improvement < self.min_improvement: reasons.append("insufficient_heldout_improvement")
        if regression_drop > self.max_regression: reasons.append("regression_threshold_exceeded")
        if update_norm > max_update_norm + 1e-6: reasons.append("update_bound_exceeded")
        if not np.isfinite(candidate.parameters).all(): reasons.append("non_finite_parameters")
        return EvaluationReport(candidate_id,
            {"accuracy_before": before["accuracy"], "accuracy_after": after["accuracy"],
             "accuracy_improvement": improvement, "log_loss_after": after["log_loss"]},
            {"accuracy_before": reg_before["accuracy"], "accuracy_after": reg_after["accuracy"],
             "accuracy_drop": regression_drop},
            {"update_norm": update_norm, "max_update_norm": max_update_norm,
             "finite": float(np.isfinite(candidate.parameters).all())},
            {"brier_before": before["brier"], "brier_after": after["brier"]}, not reasons, tuple(reasons))
