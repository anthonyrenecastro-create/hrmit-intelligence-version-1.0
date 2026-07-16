from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreferenceModel:
    weights: dict[str, float]
    bias: float


class LearningSystem:
    """Compatibility baseline for the former deterministic preference API.

    This remains available for callers that rely on ``update``. It is explicitly
    labelled heuristic and is not used as evidence of Stage 6 completion.
    """
    def __init__(self, memory: Any) -> None:
        self.memory = memory

    def update(self, baseline_record: dict[str, Any], starting_preferences: dict[str, float] | None = None,
               learning_rate: float = .05) -> dict[str, Any]:
        starting = starting_preferences or {"exploration": .2, "safety": .3, "efficiency": .5, "bias": 0.0}
        weights = {name: float(value) * (1 + learning_rate) for name, value in starting.items() if name != "bias"}
        bias = float(starting.get("bias", 0.0)) + learning_rate
        metrics = {name: value for name, value in baseline_record.items() if name.startswith("L_")}
        strength = float(sum(metrics.values()) / max(1, len(metrics))) if metrics else 0.0
        if strength > 0 and hasattr(self.memory, "add_entry"):
            self.memory.add_entry("adaptation_signal", f"Captured adaptation signal with strength {strength}",
                                  {"type": "heuristic_adaptation", "strength": strength})
        memory_growth = len(getattr(self.memory, "entries", ()))
        score = min(1.0, strength * .1 + (sum(weights.values()) + abs(bias)) * .05 + memory_growth * .01)
        return {"signals": {"baseline_phase": baseline_record.get("phase"), "metrics": metrics,
                            "signal_strength": strength}, "adaptation_mode": "controlled_heuristic",
                "preference_model": {"weights": weights, "bias": bias},
                "adaptation_metrics": {"signal_strength": float(baseline_record.get("L_total", 0.0)),
                                       "preference_shift": sum(weights.values()) + abs(bias),
                                       "memory_growth": memory_growth, "adaptation_score": score},
                "completion_claim": False}
