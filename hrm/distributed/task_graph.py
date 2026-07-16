from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from hrm.distributed.types import CognitiveTask


def _now() -> float:
    return time.time()


@dataclass(frozen=True)
class TaskRecord:
    task: CognitiveTask
    status: str
    attempt: int
    created_at: float
    updated_at: float
    completed_at: float | None
    result: dict[str, Any]


class TaskGraph:
    VALID_STATUSES = {
        "pending",
        "blocked",
        "ready",
        "running",
        "completed",
        "failed",
        "cancelled",
        "reassigned",
        "unresolved",
    }

    def __init__(self, tasks: list[CognitiveTask] | None = None) -> None:
        self.records: dict[str, TaskRecord] = {}
        self.dependencies: dict[str, tuple[str, ...]] = {}
        self.reverse_dependencies: dict[str, tuple[str, ...]] = {}
        for task in tasks or []:
            self.add_task(task)

    def add_task(self, task: CognitiveTask) -> None:
        if task.task_id in self.records:
            raise ValueError(f"Task {task.task_id} already exists")
        self.records[task.task_id] = TaskRecord(
            task=task,
            status="pending",
            attempt=0,
            created_at=_now(),
            updated_at=_now(),
            completed_at=None,
            result={},
        )
        self.dependencies[task.task_id] = task.dependencies
        for dep in task.dependencies:
            if dep not in self.reverse_dependencies:
                self.reverse_dependencies[dep] = ()
            self.reverse_dependencies[dep] = tuple(self.reverse_dependencies[dep] + (task.task_id,))

    def get_status(self, task_id: str) -> str:
        return self.records[task_id].status

    def set_status(self, task_id: str, status: str) -> None:
        if status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid status {status}")
        record = self.records[task_id]
        self.records[task_id] = TaskRecord(
            task=record.task,
            status=status,
            attempt=record.attempt,
            created_at=record.created_at,
            updated_at=_now(),
            completed_at=record.completed_at,
            result=record.result,
        )

    def set_result(self, task_id: str, result: dict[str, Any]) -> None:
        record = self.records[task_id]
        self.records[task_id] = TaskRecord(
            task=record.task,
            status=record.status,
            attempt=record.attempt,
            created_at=record.created_at,
            updated_at=_now(),
            completed_at=_now(),
            result=result,
        )

    def mark_attempt(self, task_id: str) -> None:
        record = self.records[task_id]
        self.records[task_id] = TaskRecord(
            task=record.task,
            status=record.status,
            attempt=record.attempt + 1,
            created_at=record.created_at,
            updated_at=_now(),
            completed_at=record.completed_at,
            result=record.result,
        )

    def validate(self) -> dict[str, Any]:
        missing = [task_id for task_id, deps in self.dependencies.items() if any(dep not in self.records for dep in deps)]
        if missing:
            return {"valid": False, "reason": "missing_dependencies", "missing": missing}
        visited: dict[str, str] = {}

        def visit(node: str) -> bool:
            if visited.get(node) == "visiting":
                return False
            if visited.get(node) == "visited":
                return True
            visited[node] = "visiting"
            for dep in self.dependencies.get(node, ()):
                if not visit(dep):
                    return False
            visited[node] = "visited"
            return True

        if not all(visit(task_id) for task_id in self.records):
            return {"valid": False, "reason": "cycle_detected"}

        unserviceable = [task_id for task_id, record in self.records.items() if record.task.required_capabilities and not record.task.required_capabilities]
        if unserviceable:
            return {"valid": False, "reason": "impossible_capability_requirements", "unserviceable": unserviceable}

        return {"valid": True, "reason": "ok"}

    def get_ready_tasks(self) -> list[CognitiveTask]:
        ready = []
        for record in self.records.values():
            if record.status not in {"pending", "blocked"}:
                continue
            if any(self.records[dep].status != "completed" for dep in record.task.dependencies):
                self.set_status(record.task.task_id, "blocked")
                continue
            ready.append(record.task)
        return sorted(ready, key=lambda t: (-t.priority, t.deadline or float("inf")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tasks": {
                task_id: {
                    "objective": rec.task.objective,
                    "status": rec.status,
                    "dependencies": rec.task.dependencies,
                    "attempt": rec.attempt,
                    "result": rec.result,
                }
                for task_id, rec in self.records.items()
            }
        }

    def dump(self, path: str | Any) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
