from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _now() -> float:
    return time.time()


@dataclass(frozen=True)
class SharedMemoryVersion:
    record_id: str
    version: int
    parent_version: int | None
    author_module_id: str
    task_id: str
    value: Any
    timestamp: float
    evidence: tuple[Any, ...]
    confidence: float
    status: str

    @classmethod
    def create(
        cls,
        record_id: str,
        version: int,
        parent_version: int | None,
        author_module_id: str,
        task_id: str,
        value: Any,
        evidence: tuple[Any, ...] | None = None,
        confidence: float = 0.0,
        status: str = "committed",
    ) -> "SharedMemoryVersion":
        return cls(
            record_id=record_id,
            version=version,
            parent_version=parent_version,
            author_module_id=author_module_id,
            task_id=task_id,
            value=value,
            timestamp=_now(),
            evidence=tuple(evidence or ()),
            confidence=float(confidence),
            status=status,
        )


@dataclass
class SharedMemoryRecord:
    record_id: str
    versions: list[SharedMemoryVersion] = field(default_factory=list)

    @property
    def current(self) -> SharedMemoryVersion | None:
        if not self.versions:
            return None
        return self.versions[-1]

    def append_version(self, version: SharedMemoryVersion) -> None:
        self.versions.append(version)


class SharedMemory:
    def __init__(self) -> None:
        self.records: dict[str, SharedMemoryRecord] = {}
        self.conflicts: list[dict[str, Any]] = []

    def write(
        self,
        record_id: str,
        author_module_id: str,
        task_id: str,
        value: Any,
        expected_version: int | None = None,
        evidence: tuple[Any, ...] | None = None,
        confidence: float = 0.0,
        status: str = "committed",
    ) -> SharedMemoryVersion:
        record = self.records.get(record_id)
        current_version = record.current.version if record and record.current else 0
        if expected_version is not None and expected_version != current_version:
            conflict = {
                "record_id": record_id,
                "expected_version": expected_version,
                "current_version": current_version,
                "author_module_id": author_module_id,
                "task_id": task_id,
                "timestamp": _now(),
            }
            self.conflicts.append(conflict)
            raise ValueError("Stale write rejected")
        next_version = current_version + 1
        version = SharedMemoryVersion.create(
            record_id=record_id,
            version=next_version,
            parent_version=current_version if record and record.current else None,
            author_module_id=author_module_id,
            task_id=task_id,
            value=value,
            evidence=evidence,
            confidence=confidence,
            status=status,
        )
        if record is None:
            record = SharedMemoryRecord(record_id=record_id)
            self.records[record_id] = record
        record.append_version(version)
        return version

    def read(self, record_id: str) -> SharedMemoryVersion | None:
        record = self.records.get(record_id)
        return record.current if record else None

    def snapshot(self) -> dict[str, Any]:
        return {
            record_id: [
                {
                    "version": version.version,
                    "parent_version": version.parent_version,
                    "author_module_id": version.author_module_id,
                    "task_id": version.task_id,
                    "value": version.value,
                    "timestamp": version.timestamp,
                    "evidence": version.evidence,
                    "confidence": version.confidence,
                    "status": version.status,
                }
                for version in record.versions
            ]
            for record_id, record in self.records.items()
        }

    def rollback(self, record_id: str, version: int) -> bool:
        record = self.records.get(record_id)
        if record is None:
            return False
        if not any(v.version == version for v in record.versions):
            return False
        record.versions = [v for v in record.versions if v.version <= version]
        return True

    def conflicts_snapshot(self) -> list[dict[str, Any]]:
        return list(self.conflicts)
