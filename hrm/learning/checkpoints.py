from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _checksum(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CheckpointRecord:
    checkpoint_id: str
    path: str
    created_at: float
    checksum: str
    metadata: dict[str, Any]


class CheckpointManager:
    def __init__(self, active_checkpoint: dict[str, Any] | None = None):
        self.active_checkpoint = active_checkpoint or {}
        self.history: list[CheckpointRecord] = []

    def create_checkpoint(self, data: dict[str, Any], path: Path | str) -> CheckpointRecord:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(data, sort_keys=True)
        path.write_text(serialized, encoding="utf-8")
        record = CheckpointRecord(
            checkpoint_id=_checksum(serialized)[:16],
            path=str(path),
            created_at=__import__("time").time(),
            checksum=_checksum(serialized),
            metadata={"shape": len(serialized)},
        )
        self.history.append(record)
        return record

    def load_checkpoint(self, path: Path | str) -> dict[str, Any]:
        serialized = Path(path).read_text(encoding="utf-8")
        return json.loads(serialized)
