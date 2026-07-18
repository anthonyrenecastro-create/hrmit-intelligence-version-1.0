from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RandomStateManager:
    state: dict[str, Any]

    @classmethod
    def from_seed(cls, seed: int) -> "RandomStateManager":
        generator = np.random.default_rng(seed)
        return cls(state=deepcopy(generator.bit_generator.state))

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "RandomStateManager":
        return cls(state=deepcopy(state))

    def generator(self) -> np.random.Generator:
        generator = np.random.default_rng()
        generator.bit_generator.state = deepcopy(self.state)
        return generator

    def capture(self, generator: np.random.Generator) -> None:
        self.state = deepcopy(generator.bit_generator.state)

    def digest(self) -> str:
        payload = json.dumps(self.state, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
