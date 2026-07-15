from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MemoryEntry:
    key: str
    content: str
    metadata: dict[str, Any]
    embedding: np.ndarray


class LongTermMemory:
    def __init__(self, entries: list[MemoryEntry] | None = None, dim: int = 32) -> None:
        self.entries = entries or []
        self.dim = dim

    @staticmethod
    def _embed_text(text: str, dim: int = 32) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        values = [b for b in h]
        vector = np.array(values[:dim], dtype=np.float32)
        if vector.size < dim:
            vector = np.pad(vector, (0, dim - vector.size), mode="constant")
        norm = np.linalg.norm(vector) + 1e-9
        return np.tanh(vector / norm)

    def add_entry(self, key: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        metadata = metadata or {}
        embedding = self._embed_text(content, dim=self.dim)
        self.entries.append(MemoryEntry(key=key, content=content, metadata=metadata, embedding=embedding))

    def retrieve(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        if not self.entries:
            return []
        q_emb = self._embed_text(query, dim=self.dim)
        scores = []
        q_norm = np.linalg.norm(q_emb) + 1e-9
        for entry in self.entries:
            dot = float(np.dot(q_emb, entry.embedding))
            score = dot / (q_norm * (np.linalg.norm(entry.embedding) + 1e-9))
            scores.append((score, entry))
        scores.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "key": entry.key,
                "score": float(score),
                "content": entry.content,
                "metadata": entry.metadata,
            }
            for score, entry in scores[:k]
        ]

    @classmethod
    def from_baseline_record(cls, baseline_record: dict[str, Any], dim: int = 32) -> "LongTermMemory":
        memory = cls(dim=dim)
        meta_keys = [
            "phase",
            "candidate",
            "config_hash",
            "seed",
            "backend",
            "profile",
            "steps",
            "batch",
            "perturb_step",
            "perturb_strength",
            "did_recover",
            "ledger_pass",
            "bounded_pass",
        ]
        for key in meta_keys:
            if key in baseline_record:
                content = f"{key}: {baseline_record[key]}"
                memory.add_entry(key, content, {"field": key, "value": baseline_record[key]})

        metrics = {k: baseline_record[k] for k in baseline_record if k.startswith("L_") or k in ("pre_mse", "peak_mse", "recovery_time")}
        for metric, value in metrics.items():
            content = f"metric {metric}: {value}"
            memory.add_entry(metric, content, {"metric": metric, "value": value})

        memory.add_entry(
            "baseline_summary",
            f"Baseline run summary for phase={baseline_record.get('phase')} candidate={baseline_record.get('candidate')} with final L_total={baseline_record.get('L_total')}",
            {"summary": True},
        )
        return memory


@dataclass(frozen=True)
class Plan:
    query: str
    steps: list[str]
    sources: list[str]


class Planner:
    def __init__(self, memory: LongTermMemory) -> None:
        self.memory = memory

    def create_plan(self, query: str, n_steps: int = 5) -> Plan:
        retrieved = self.memory.retrieve(query, k=n_steps)
        sources = [item["key"] for item in retrieved]
        steps: list[str] = [
            "Review retrieved long-term memory entries.",
            "Assess current HRM baseline performance and safety metrics.",
            "Identify opportunities for improved guidance, cognition, and allocation dynamics.",
            "Propose a sequence of experiment adjustments based on retrieved memory.",
            "Record the planned follow-up actions and the retrieval rationale.",
        ]
        if retrieved:
            top = retrieved[0]
            steps.insert(0, f"Starting from memory entry '{top['key']}' with score {top['score']:.3f}.")
        return Plan(query=query, steps=steps, sources=sources)


def summarize_memory(memory: LongTermMemory) -> dict[str, Any]:
    return {
        "entry_count": len(memory.entries),
        "sample_keys": [entry.key for entry in memory.entries[:5]],
    }
