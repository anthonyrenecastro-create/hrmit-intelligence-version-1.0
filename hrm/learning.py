from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hrm.memory import LongTermMemory


@dataclass(frozen=True)
class LearningMetrics:
    signal_strength: float
    preference_shift: float
    memory_growth: int
    adaptation_score: float


@dataclass(frozen=True)
class PreferenceModel:
    weights: dict[str, float]
    bias: float


class LearningSystem:
    def __init__(self, memory: LongTermMemory) -> None:
        self.memory = memory
        self.preference_model = PreferenceModel(weights={}, bias=0.0)

    def capture_signals(self, baseline_record: dict[str, Any]) -> dict[str, Any]:
        metrics = {k: baseline_record[k] for k in baseline_record if k.startswith("L_")}
        signal_strength = float(sum(metrics.values()) / max(1.0, len(metrics))) if metrics else 0.0
        return {
            "baseline_phase": baseline_record.get("phase"),
            "metrics": metrics,
            "signal_strength": signal_strength,
        }

    def adapt_preferences(self, current_preferences: dict[str, float], adaptation_rate: float = 0.01) -> PreferenceModel:
        adjusted = {}
        for k, v in current_preferences.items():
            if k == "bias":
                continue
            adjusted[k] = float(v) * (1.0 + adaptation_rate)

        bias_value = float(current_preferences.get("bias", 0.0)) + adaptation_rate
        self.preference_model = PreferenceModel(weights=adjusted, bias=bias_value)
        return self.preference_model

    def record_adaptation_signal(self, learning_signals: dict[str, Any]) -> None:
        if learning_signals["signal_strength"] > 0.0:
            self.memory.add_entry(
                "adaptation_signal",
                f"Captured adaptation signal with strength {learning_signals['signal_strength']}",
                {"type": "heuristic_adaptation", "strength": learning_signals["signal_strength"]},
            )

    def compute_adaptation_metrics(self, baseline_record: dict[str, Any], preference_model: PreferenceModel) -> LearningMetrics:
        signal_strength = float(baseline_record.get("L_total", 0.0))
        preference_shift = float(sum(preference_model.weights.values()) + abs(preference_model.bias))
        memory_growth = len(self.memory.entries)
        adaptation_score = min(1.0, (signal_strength * 0.1) + (preference_shift * 0.05) + (memory_growth * 0.01))
        return LearningMetrics(
            signal_strength=signal_strength,
            preference_shift=preference_shift,
            memory_growth=memory_growth,
            adaptation_score=adaptation_score,
        )

    def update(self, baseline_record: dict[str, Any], starting_preferences: dict[str, float] | None = None, learning_rate: float = 0.05) -> dict[str, Any]:
        starting_preferences = starting_preferences or {"exploration": 0.2, "safety": 0.3, "efficiency": 0.5, "bias": 0.0}
        signals = self.capture_signals(baseline_record)
        self.record_adaptation_signal(signals)
        preference_model = self.adapt_preferences(starting_preferences, adaptation_rate=learning_rate)
        metrics = self.compute_adaptation_metrics(baseline_record, preference_model)
        return {
            "signals": signals,
            "adaptation_mode": "controlled_heuristic",
            "preference_model": {
                "weights": preference_model.weights,
                "bias": preference_model.bias,
            },
            "adaptation_metrics": metrics.__dict__,
        }
