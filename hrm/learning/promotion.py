from __future__ import annotations

from typing import Any

from .evaluator import EvaluationReport


class PromotionGate:
    def evaluate(self, report: EvaluationReport, candidate_checkpoint: dict[str, Any]) -> dict[str, Any]:
        if report.accepted and not report.rejection_reasons:
            decision = "accepted"
        elif "insufficient_primary_improvement" in report.rejection_reasons:
            decision = "rejected_no_improvement"
        elif "regression_threshold_exceeded" in report.rejection_reasons:
            decision = "rejected_regression"
        elif "safety_threshold_failed" in report.rejection_reasons:
            decision = "rejected_safety"
        else:
            decision = "rejected_incomplete_evidence"
        return {
            "decision": decision,
            "accepted": report.accepted,
            "rejection_reasons": report.rejection_reasons,
            "metrics": {
                **report.primary_metrics,
                **report.regression_metrics,
                **report.safety_metrics,
            },
            "metadata": report.metadata,
        }
