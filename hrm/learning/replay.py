from __future__ import annotations

import numpy as np

from .types import ExperienceRecord


class ReplayBuffer:
    def __init__(self, records: tuple[ExperienceRecord, ...], seed: int = 0) -> None:
        self.records, self.rng = records, np.random.default_rng(seed)

    def sample(self, count: int, strategy: str = "prioritized") -> tuple[ExperienceRecord, ...]:
        if not self.records or count < 1:
            return ()
        replace = count > len(self.records)
        if strategy == "uniform":
            probabilities = None
        elif strategy == "prioritized":
            priorities = np.asarray([max(record.priority, 1e-6) for record in self.records])
            probabilities = priorities / priorities.sum()
        elif strategy == "balanced":
            successes = [r for r in self.records if r.task_outcome.success]
            failures = [r for r in self.records if not r.task_outcome.success]
            half = count // 2
            return tuple((successes * count)[:half] + (failures * count)[:count-half])
        else:
            raise ValueError(f"Unknown replay strategy: {strategy}")
        indices = self.rng.choice(len(self.records), count, replace=replace, p=probabilities)
        return tuple(self.records[int(i)] for i in indices)
