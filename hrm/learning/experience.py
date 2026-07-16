from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .types import ExperienceRecord, FeedbackRecord, TaskOutcome


class ExperienceStore:
    def __init__(self) -> None:
        self._records: dict[str, ExperienceRecord] = {}

    def add(self, record: ExperienceRecord) -> None:
        if record.experience_id in self._records:
            raise ValueError(f"Duplicate experience: {record.experience_id}")
        self._records[record.experience_id] = record

    def all(self) -> tuple[ExperienceRecord, ...]:
        return tuple(self._records.values())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(record) for record in self.all()], indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ExperienceStore":
        store = cls()
        for raw in json.loads(path.read_text(encoding="utf-8")):
            outcome = TaskOutcome(**raw.pop("task_outcome"))
            feedback = tuple(FeedbackRecord(**item) for item in raw.pop("feedback"))
            store.add(ExperienceRecord(task_outcome=outcome, feedback=feedback, **raw))
        return store
