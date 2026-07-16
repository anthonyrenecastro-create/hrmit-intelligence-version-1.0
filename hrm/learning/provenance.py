from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import AdaptationProvenance


class ProvenanceLog:
    def __init__(self, path: Path | str = "logs/adaptation_events.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, provenance: AdaptationProvenance) -> None:
        record = {**provenance.__dict__}
        existing = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def load_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
