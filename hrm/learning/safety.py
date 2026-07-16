from __future__ import annotations

from typing import Any


class SafetyChecker:
    def check(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        safety_score = 1.0 - min(1.0, abs(float(checkpoint.get("safety_bias", 0.0))))
        return {"safety_score": safety_score, "violations": []}
