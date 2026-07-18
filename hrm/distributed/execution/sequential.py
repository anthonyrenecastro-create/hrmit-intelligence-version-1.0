from __future__ import annotations

from typing import Any


class SequentialExecutor:
    def __init__(self) -> None:
        self.backend = "sequential"

    def execute(self, tasks: list[Any], worker: Any) -> list[Any]:
        results = []
        for task in tasks:
            results.append(worker(task))
        return results

    def shutdown(self) -> None:
        return None
