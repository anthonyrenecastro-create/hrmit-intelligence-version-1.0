from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .experience import ExperienceStore
from .types import ExperienceRecord


@dataclass(frozen=True)
class ReplayConfig:
    capacity: int = 512
    prioritized: bool = True
    priority_exponent: float = 0.7
    min_probability: float = 0.01
    recency_weight: float = 0.1
    balance_failure_ratio: float = 0.5
    seed: int = 0


class ReplayBuffer:
    def __init__(self, store: ExperienceStore | list[ExperienceRecord], config: ReplayConfig | None = None) -> None:
        self.store = store if isinstance(store, ExperienceStore) else ExperienceStore(list(store))
        self.config = config or ReplayConfig()
        self.random = random.Random(self.config.seed)

    def add(self, experience: ExperienceRecord) -> None:
        if len(self.store.experiences) >= self.config.capacity:
            self._prune_oldest()
        self.store.add(experience)

    def _prune_oldest(self) -> None:
        self.store.experiences.sort(key=lambda exp: exp.created_at)
        while len(self.store.experiences) >= self.config.capacity:
            self.store.experiences.pop(0)

    def sample(self, batch_size: int, strategy: str | None = None) -> list[ExperienceRecord]:
        if not self.store.experiences:
            return []
        if strategy == "balanced":
            successes = [exp for exp in self.store.experiences if bool(exp.task_outcome.success)]
            failures = [exp for exp in self.store.experiences if not bool(exp.task_outcome.success)]
            half = max(1, batch_size // 2)
            chosen = []
            if successes:
                chosen.extend(self.random.sample(successes, k=min(half, len(successes))))
            if failures:
                chosen.extend(self.random.sample(failures, k=min(batch_size - len(chosen), len(failures))))
            while len(chosen) < min(batch_size, len(self.store.experiences)):
                chosen.append(self.random.choice(self.store.experiences))
            for exp in chosen:
                self._increment_replay_count(exp)
            return chosen
        if self.config.prioritized:
            weights = [self._experience_weight(exp) for exp in self.store.experiences]
            total = sum(weights)
            if total <= 0.0:
                weights = [1.0] * len(weights)
                total = len(weights)
            probabilities = [max(self.config.min_probability, w / total) for w in weights]
            normalized = [p / sum(probabilities) for p in probabilities]
            batch = self.random.choices(self.store.experiences, weights=normalized, k=min(batch_size, len(self.store.experiences)))
        else:
            batch = self.random.sample(self.store.experiences, k=min(batch_size, len(self.store.experiences)))
        for exp in batch:
            self._increment_replay_count(exp)
        return batch

    def _experience_weight(self, experience: ExperienceRecord) -> float:
        failure_bonus = 1.0 if any(not f.task_id or f.value == "failure" for f in experience.feedback) else 0.0
        age = time.time() - experience.created_at
        age_score = 1.0 + self.config.recency_weight * math.log1p(age)
        return (experience.priority ** self.config.priority_exponent + failure_bonus) * age_score

    def _increment_replay_count(self, experience: ExperienceRecord) -> None:
        for index, exp in enumerate(self.store.experiences):
            if exp.experience_id == experience.experience_id:
                self.store.experiences[index] = ExperienceRecord(
                    experience_id=exp.experience_id,
                    task_outcome=exp.task_outcome,
                    feedback=exp.feedback,
                    reward=exp.reward,
                    priority=exp.priority,
                    replay_count=exp.replay_count + 1,
                    created_at=exp.created_at,
                    provenance=exp.provenance,
                )
                return

    def save(self, path: Path | str) -> Path:
        return self.store.save(path)

    @classmethod
    def load(cls, path: Path | str, config: ReplayConfig | None = None) -> "ReplayBuffer":
        store = ExperienceStore.load(path)
        return cls(store=store, config=config)
