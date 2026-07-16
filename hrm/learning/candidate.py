from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import AdaptationCandidate, AdaptationProvenance, ExperienceRecord


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 0
    steps: int = 10
    learning_rate: float = 0.01
    max_update_norm: float = 0.5
    max_relative_change: float = 0.05
    gradient_clip_norm: float = 1.0
    frozen_scopes: tuple[str, ...] = ()
    trainable_scopes: tuple[str, ...] = ()


class CandidateTrainer:
    def __init__(self, base_checkpoint: dict[str, Any], trainable_scopes: tuple[str, ...], config: TrainingConfig) -> None:
        self.base_checkpoint = base_checkpoint
        self.trainable_scopes = trainable_scopes
        self.config = config
        self.candidate_checkpoint: dict[str, Any] | None = None
        self.update_norm: float = 0.0

    def create_candidate(self, parent_checkpoint_id: str, provenance_id: str, scope_names: tuple[str, ...]) -> AdaptationCandidate:
        checkpoint_path = f"checkpoints/candidate_{parent_checkpoint_id}_{int(time.time())}.json"
        candidate = AdaptationCandidate.create(
            parent_checkpoint_id=parent_checkpoint_id,
            parameter_scope=scope_names,
            training_config=self.config.__dict__,
            checkpoint_path=checkpoint_path,
            provenance_id=provenance_id,
        )
        return candidate

    def train(self, experiences: list[ExperienceRecord]) -> dict[str, Any]:
        self.candidate_checkpoint = json.loads(json.dumps(self.base_checkpoint))
        delta_norm = 0.0
        for exp in experiences[: self.config.steps]:
            delta = float(exp.reward) * self.config.learning_rate * 0.01
            for scope in self.trainable_scopes:
                key = f"params_{scope}"
                base_value = float(self.candidate_checkpoint.get(key, 0.0))
                update = delta * (1.0 if base_value >= 0 else -1.0)
                self.candidate_checkpoint[key] = base_value + update
                delta_norm += abs(update)
        self.update_norm = delta_norm
        if self.update_norm > self.config.max_update_norm:
            raise ValueError("Candidate update exceeded max_update_norm")
        return self.candidate_checkpoint

    def checkpoint(self, path: Path | str) -> Path:
        if self.candidate_checkpoint is None:
            raise RuntimeError("Candidate checkpoint has not been trained yet")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.candidate_checkpoint, indent=2), encoding="utf-8")
        return path
