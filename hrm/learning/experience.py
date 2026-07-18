from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .types import ExperienceRecord, FeedbackRecord, TaskOutcome


class ExperienceStore:
    def __init__(self, experiences: list[ExperienceRecord] | None = None) -> None:
        self.experiences = experiences or []

    def add(self, experience: ExperienceRecord) -> None:
        self.experiences.append(experience)

    def get(self, experience_id: str) -> ExperienceRecord | None:
        return next((exp for exp in self.experiences if exp.experience_id == experience_id), None)

    def list_ids(self) -> list[str]:
        return [exp.experience_id for exp in self.experiences]

    def all(self) -> list[ExperienceRecord]:
        return list(self.experiences)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = [self._serialize_experience(exp) for exp in self.experiences]
        path.write_text(json.dumps(content, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "ExperienceStore":
        path = Path(path)
        content = json.loads(path.read_text(encoding="utf-8"))
        experiences: list[ExperienceRecord] = []
        for entry in content:
            task_outcome = TaskOutcome(**entry["task_outcome"])
            feedback = tuple(FeedbackRecord(**record) for record in entry["feedback"])
            experiences.append(
                ExperienceRecord(
                    experience_id=entry["experience_id"],
                    task_outcome=task_outcome,
                    feedback=feedback,
                    reward=entry["reward"],
                    priority=entry["priority"],
                    replay_count=entry["replay_count"],
                    created_at=entry["created_at"],
                    provenance=entry["provenance"],
                )
            )
        return cls(experiences=experiences)

    def _serialize_experience(self, experience: ExperienceRecord) -> dict[str, Any]:
        return {
            "experience_id": experience.experience_id,
            "task_outcome": asdict(experience.task_outcome),
            "feedback": [asdict(record) for record in experience.feedback],
            "reward": experience.reward,
            "priority": experience.priority,
            "replay_count": experience.replay_count,
            "created_at": experience.created_at,
            "provenance": experience.provenance,
        }
