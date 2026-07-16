from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreferenceModelBaseline:
    weights: dict[str, float]
    bias: float

    def update_preferences(self, current_preferences: dict[str, float], adaptation_rate: float = 0.01) -> "PreferenceModelBaseline":
        adjusted = {}
        for key, value in current_preferences.items():
            if key == "bias":
                continue
            adjusted[key] = float(value) * (1.0 + adaptation_rate)
        return PreferenceModelBaseline(weights=adjusted, bias=float(current_preferences.get("bias", 0.0)) + adaptation_rate)
