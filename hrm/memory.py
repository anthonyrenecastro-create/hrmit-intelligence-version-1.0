from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MEMORY_SCHEMA_VERSION = "1.0"
DEFAULT_WORKING_CAPACITY = 32
DEFAULT_WORKING_EXPIRATION_STEPS = 200
DEFAULT_EMBEDDING_DIM = 32


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _embed_text(text: str, dim: int = DEFAULT_EMBEDDING_DIM) -> np.ndarray:
    h = hashlib.sha256(_normalize_text(text).encode("utf-8")).digest()
    values = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
    if values.size < dim:
        values = np.pad(values, (0, dim - values.size), mode="constant")
    vector = values[:dim]
    norm = np.linalg.norm(vector) + 1e-9
    return np.tanh(vector / norm)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    norm_a = np.linalg.norm(a) + 1e-9
    norm_b = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass(frozen=True)
class MemoryItem:
    memory_id: str
    key: str
    value: Any
    timestamp: float
    sequence_step: int
    source: str
    importance: float
    confidence: float
    metadata: dict[str, Any]
    version: int
    embedding: np.ndarray = field(compare=False, repr=False)


@dataclass(frozen=True)
class MemoryQuery:
    query: str
    task: str | None = None
    source_filters: tuple[str, ...] = ()
    min_confidence: float = 0.0
    since_timestamp: float = 0.0
    until_timestamp: float | None = None
    limit: int = 5


@dataclass(frozen=True)
class RetrievedMemory:
    memory_id: str
    value: Any
    score: float
    similarity_score: float
    recency_score: float
    importance_score: float
    confidence_score: float
    source_score: float
    retrieval_reason: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MemoryConflict:
    conflict_id: str
    memory_ids: tuple[str, ...]
    conflict_type: str
    detected_at: float
    status: str
    resolution: str | None
    resolution_confidence: float | None
    provenance: dict[str, Any]


@dataclass(frozen=True)
class MemoryRevision:
    memory_id: str
    previous_value: Any
    revised_value: Any
    reason: str
    timestamp: float
    actor: str
    evidence: dict[str, Any]
    version: int
    confidence_change: float


@dataclass(frozen=True)
class ConsolidationRecord:
    consolidation_id: str
    source_episode_ids: tuple[str, ...]
    concept: str
    prototype: Any
    confidence: float
    method: str
    created_at: float
    revisions: tuple[MemoryRevision, ...]
    unresolved_disagreements: tuple[str, ...]


@dataclass(frozen=True)
class MemoryMetrics:
    working_count: int
    episodic_count: int
    semantic_count: int
    conflict_count: int
    revision_count: int
    retrieval_latency_seconds: float


@dataclass(frozen=True)
class EpisodicMemoryItem:
    memory_id: str
    key: str
    value: Any
    timestamp: float
    sequence_step: int
    source: str
    importance: float
    confidence: float
    metadata: dict[str, Any]
    version: int
    embedding: np.ndarray = field(compare=False, repr=False)


@dataclass(frozen=True)
class SemanticMemoryItem:
    memory_id: str
    concept: str
    prototype: Any
    confidence: float
    source_episode_ids: tuple[str, ...]
    metadata: dict[str, Any]
    version: int
    embedding: np.ndarray = field(compare=False, repr=False)


@dataclass
class MemoryConfig:
    enabled: bool = True
    working_capacity: int = DEFAULT_WORKING_CAPACITY
    working_expiration_steps: int = DEFAULT_WORKING_EXPIRATION_STEPS
    retrieval_weights: dict[str, float] = field(default_factory=lambda: {
        "similarity": 0.35,
        "recency": 0.20,
        "importance": 0.15,
        "confidence": 0.15,
        "source": 0.10,
        "task_relevance": 0.05,
        "conflict_penalty": 0.20,
    })
    consolidation_threshold: float = 0.75
    consolidation_support: int = 2
    max_memory_age_seconds: float = 60.0 * 60.0 * 24.0 * 30.0
    schema_version: str = MEMORY_SCHEMA_VERSION
    memory_context_limit: int = 5
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    seed: int = 0


@dataclass(frozen=True)
class MemoryEntry:
    key: str
    content: str
    metadata: dict[str, Any]
    embedding: np.ndarray


class LongTermMemory:
    def __init__(self, entries: list[MemoryEntry] | None = None, dim: int = DEFAULT_EMBEDDING_DIM) -> None:
        self.entries = entries or []
        self.dim = dim

    def add_entry(self, key: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        metadata = metadata or {}
        embedding = _embed_text(content, dim=self.dim)
        self.entries.append(MemoryEntry(key=key, content=content, metadata=metadata, embedding=embedding))

    def retrieve(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        if not self.entries:
            return []
        q_emb = _embed_text(query, dim=self.dim)
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
    def from_baseline_record(cls, baseline_record: dict[str, Any], dim: int = DEFAULT_EMBEDDING_DIM) -> "LongTermMemory":
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "entries": [
                {
                    "key": entry.key,
                    "content": entry.content,
                    "metadata": entry.metadata,
                    "embedding": entry.embedding.tolist(),
                }
                for entry in self.entries
            ],
        }

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "LongTermMemory":
        path = Path(path)
        content = json.loads(path.read_text(encoding="utf-8"))
        memory = cls(dim=content.get("dim", DEFAULT_EMBEDDING_DIM))
        for entry_data in content.get("entries", []):
            embedding = np.asarray(entry_data.get("embedding", []), dtype=np.float32)
            memory.entries.append(
                MemoryEntry(
                    key=entry_data["key"],
                    content=entry_data["content"],
                    metadata=entry_data.get("metadata", {}),
                    embedding=embedding,
                )
            )
        return memory


class WorkingMemory:
    def __init__(self, capacity: int = DEFAULT_WORKING_CAPACITY, expiration_steps: int = DEFAULT_WORKING_EXPIRATION_STEPS, embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> None:
        self.capacity = capacity
        self.expiration_steps = expiration_steps
        self.embedding_dim = embedding_dim
        self.items: list[MemoryItem] = []

    def insert(self, item: MemoryItem) -> None:
        self.items = [existing for existing in self.items if existing.memory_id != item.memory_id]
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items.sort(key=lambda entry: entry.timestamp)
            self.items = self.items[-self.capacity:]

    def retrieve(self, query: MemoryQuery, limit: int = 5, weights: dict[str, float] | None = None, seed: int | None = None) -> list[RetrievedMemory]:
        if not self.items:
            return []
        weights = weights or {}
        weights = {**MemoryConfig().retrieval_weights, **weights}
        q_emb = _embed_text(query.query, dim=self.embedding_dim)
        now = time.time()
        scored: list[RetrievedMemory] = []
        for item in self.items:
            similarity = _cosine_similarity(q_emb, item.embedding)
            recency = 1.0 / (1.0 + max(0.0, query.since_timestamp or now - item.timestamp))
            importance = min(1.0, max(0.0, item.importance))
            confidence = min(1.0, max(0.0, item.confidence))
            source_relevance = 1.0 if not query.source_filters or item.source in query.source_filters else 0.0
            task_relevance = 1.0 if query.task and query.task.lower() in item.metadata.get("context", item.key).lower() else 0.0
            conflict_penalty = 0.0
            score = (
                weights.get("similarity", 0.0) * similarity
                + weights.get("recency", 0.0) * recency
                + weights.get("importance", 0.0) * importance
                + weights.get("confidence", 0.0) * confidence
                + weights.get("source", 0.0) * source_relevance
                + weights.get("task_relevance", 0.0) * task_relevance
                - weights.get("conflict_penalty", 0.0) * conflict_penalty
            )
            scored.append(
                RetrievedMemory(
                    memory_id=item.memory_id,
                    value=item.value,
                    score=score,
                    similarity_score=similarity,
                    recency_score=recency,
                    importance_score=importance,
                    confidence_score=confidence,
                    source_score=source_relevance,
                    retrieval_reason=f"working memory: {item.key}",
                    metadata=item.metadata,
                )
            )
        scored.sort(key=lambda item: (item.score, item.similarity_score, item.recency_score), reverse=True)
        if seed is not None:
            random.Random(seed).shuffle(scored)
            scored.sort(key=lambda item: (item.score, item.similarity_score, item.recency_score), reverse=True)
        return scored[:limit]

    def expire(self, current_step: int) -> list[str]:
        cutoff = current_step - self.expiration_steps
        expired = [item.memory_id for item in self.items if item.sequence_step < cutoff]
        self.items = [item for item in self.items if item.sequence_step >= cutoff]
        return expired

    def clear(self) -> None:
        self.items.clear()


class EpisodicMemory:
    def __init__(self, embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> None:
        self.embedding_dim = embedding_dim
        self.items: dict[str, EpisodicMemoryItem] = {}
        self.revisions: dict[str, list[MemoryRevision]] = {}
        self.conflicts: dict[str, MemoryConflict] = {}

    def insert(self, item: EpisodicMemoryItem) -> None:
        existing = self.items.get(item.memory_id)
        if existing is not None and existing.value != item.value:
            self._record_conflict(existing, item)
        self.items[item.memory_id] = item

    def retrieve(self, query: MemoryQuery, limit: int = 5, weights: dict[str, float] | None = None, seed: int | None = None) -> list[RetrievedMemory]:
        if not self.items:
            return []
        weights = weights or {}
        weights = {**MemoryConfig().retrieval_weights, **weights}
        q_emb = _embed_text(query.query, dim=self.embedding_dim)
        scored: list[RetrievedMemory] = []
        for item in self.items.values():
            if item.confidence < query.min_confidence:
                continue
            if query.until_timestamp is not None and item.timestamp > query.until_timestamp:
                continue
            if item.timestamp < query.since_timestamp:
                continue
            similarity = _cosine_similarity(q_emb, item.embedding)
            recency = 1.0 / (1.0 + max(0.0, time.time() - item.timestamp))
            importance = min(1.0, max(0.0, item.importance))
            confidence = min(1.0, max(0.0, item.confidence))
            source_relevance = 1.0 if not query.source_filters or item.source in query.source_filters else 0.0
            task_relevance = 1.0 if query.task and query.task.lower() in item.key.lower() else 0.0
            unresolved_conflict = 1.0 if any(item.memory_id in conflict.memory_ids and conflict.status != "resolved" for conflict in self.conflicts.values()) else 0.0
            score = (
                weights.get("similarity", 0.0) * similarity
                + weights.get("recency", 0.0) * recency
                + weights.get("importance", 0.0) * importance
                + weights.get("confidence", 0.0) * confidence
                + weights.get("source", 0.0) * source_relevance
                + weights.get("task_relevance", 0.0) * task_relevance
                - weights.get("conflict_penalty", 0.0) * unresolved_conflict
            )
            scored.append(
                RetrievedMemory(
                    memory_id=item.memory_id,
                    value=item.value,
                    score=score,
                    similarity_score=similarity,
                    recency_score=recency,
                    importance_score=importance,
                    confidence_score=confidence,
                    source_score=source_relevance,
                    retrieval_reason=f"episodic memory: {item.key}",
                    metadata=item.metadata,
                )
            )
        if seed is not None:
            random.Random(seed).shuffle(scored)
        scored.sort(key=lambda item: (item.score, item.similarity_score, item.recency_score), reverse=True)
        return scored[:limit]

    def revise(self, memory_id: str, new_value: Any, reason: str, source: str, evidence: dict[str, Any], confidence: float | None = None) -> MemoryRevision:
        if memory_id not in self.items:
            raise KeyError(f"Memory id {memory_id} not found")
        existing = self.items[memory_id]
        revised_value = new_value
        revision = MemoryRevision(
            memory_id=memory_id,
            previous_value=existing.value,
            revised_value=revised_value,
            reason=reason,
            timestamp=time.time(),
            actor=source,
            evidence=evidence,
            version=existing.version + 1,
            confidence_change=(confidence or existing.confidence) - existing.confidence,
        )
        self.revisions.setdefault(memory_id, []).append(revision)
        updated = EpisodicMemoryItem(
            memory_id=existing.memory_id,
            key=existing.key,
            value=revised_value,
            timestamp=revision.timestamp,
            sequence_step=existing.sequence_step,
            source=source,
            importance=existing.importance,
            confidence=confidence if confidence is not None else existing.confidence,
            metadata={**existing.metadata, "revised": True},
            version=revision.version,
            embedding=_embed_text(str(revised_value), dim=self.embedding_dim),
        )
        self.items[memory_id] = updated
        return revision

    def _record_conflict(self, existing: EpisodicMemoryItem, incoming: EpisodicMemoryItem) -> None:
        conflict_id = uuid.uuid4().hex
        conflict = MemoryConflict(
            conflict_id=conflict_id,
            memory_ids=(existing.memory_id, incoming.memory_id),
            conflict_type="direct_contradiction",
            detected_at=time.time(),
            status="unresolved",
            resolution=None,
            resolution_confidence=None,
            provenance={
                "existing_source": existing.source,
                "incoming_source": incoming.source,
                "existing_value": existing.value,
                "incoming_value": incoming.value,
            },
        )
        self.conflicts[conflict_id] = conflict

    def get_revisions(self, memory_id: str) -> list[MemoryRevision]:
        return list(self.revisions.get(memory_id, []))

    def get_conflicts(self) -> list[MemoryConflict]:
        return list(self.conflicts.values())

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        archive = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "items": [
                {
                    "memory_id": item.memory_id,
                    "key": item.key,
                    "value": item.value,
                    "timestamp": item.timestamp,
                    "sequence_step": item.sequence_step,
                    "source": item.source,
                    "importance": item.importance,
                    "confidence": item.confidence,
                    "metadata": item.metadata,
                    "version": item.version,
                    "embedding": item.embedding.tolist(),
                }
                for item in self.items.values()
            ],
            "revisions": [
                {
                    "memory_id": rev.memory_id,
                    "previous_value": rev.previous_value,
                    "revised_value": rev.revised_value,
                    "reason": rev.reason,
                    "timestamp": rev.timestamp,
                    "actor": rev.actor,
                    "evidence": rev.evidence,
                    "version": rev.version,
                    "confidence_change": rev.confidence_change,
                }
                for revisions in self.revisions.values()
                for rev in revisions
            ],
            "conflicts": [conflict.__dict__ for conflict in self.conflicts.values()],
        }
        content = json.dumps(archive, indent=2)
        checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
        payload = {"checksum": checksum, "archive": archive}
        temp = Path(tempfile.mkstemp(prefix="episodic_", suffix=".json", dir=path.parent)[1])
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp.replace(path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "EpisodicMemory":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        checksum = payload.get("checksum", "")
        archive = payload.get("archive", {})
        if hashlib.sha256(json.dumps(archive, indent=2).encode("utf-8")).hexdigest() != checksum:
            raise ValueError("Episodic memory file checksum mismatch or corruption detected")
        memory = cls()
        for item_data in archive.get("items", []):
            embedding = np.asarray(item_data.get("embedding", []), dtype=np.float32)
            memory.items[item_data["memory_id"]] = EpisodicMemoryItem(
                memory_id=item_data["memory_id"],
                key=item_data["key"],
                value=item_data["value"],
                timestamp=item_data["timestamp"],
                sequence_step=item_data["sequence_step"],
                source=item_data["source"],
                importance=item_data["importance"],
                confidence=item_data["confidence"],
                metadata=item_data["metadata"],
                version=item_data["version"],
                embedding=embedding,
            )
        for rev_data in archive.get("revisions", []):
            rev = MemoryRevision(
                memory_id=rev_data["memory_id"],
                previous_value=rev_data["previous_value"],
                revised_value=rev_data["revised_value"],
                reason=rev_data["reason"],
                timestamp=rev_data["timestamp"],
                actor=rev_data["actor"],
                evidence=rev_data["evidence"],
                version=rev_data["version"],
                confidence_change=rev_data["confidence_change"],
            )
            memory.revisions.setdefault(rev.memory_id, []).append(rev)
        for conflict_data in archive.get("conflicts", []):
            memory.conflicts[conflict_data["conflict_id"]]=MemoryConflict(
                conflict_id=conflict_data["conflict_id"],
                memory_ids=tuple(conflict_data["memory_ids"]),
                conflict_type=conflict_data["conflict_type"],
                detected_at=conflict_data["detected_at"],
                status=conflict_data["status"],
                resolution=conflict_data.get("resolution"),
                resolution_confidence=conflict_data.get("resolution_confidence"),
                provenance=conflict_data.get("provenance", {}),
            )
        return memory


class SemanticMemory:
    def __init__(self, embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> None:
        self.embedding_dim = embedding_dim
        self.records: dict[str, SemanticMemoryItem] = {}
        self.consolidations: dict[str, ConsolidationRecord] = {}

    def retrieve(self, query: MemoryQuery, limit: int = 5, weights: dict[str, float] | None = None, seed: int | None = None) -> list[RetrievedMemory]:
        if not self.records:
            return []
        weights = weights or {}
        weights = {**MemoryConfig().retrieval_weights, **weights}
        q_emb = _embed_text(query.query, dim=self.embedding_dim)
        scored: list[RetrievedMemory] = []
        for record in self.records.values():
            similarity = _cosine_similarity(q_emb, record.embedding)
            recency = 1.0 / (1.0 + max(0.0, time.time() - record.metadata.get("created_at", time.time())))
            importance = 1.0
            confidence = min(1.0, max(0.0, record.confidence))
            source_relevance = 1.0
            task_relevance = 1.0 if query.task and query.task.lower() in record.concept.lower() else 0.0
            score = (
                weights.get("similarity", 0.0) * similarity
                + weights.get("recency", 0.0) * recency
                + weights.get("importance", 0.0) * importance
                + weights.get("confidence", 0.0) * confidence
                + weights.get("source", 0.0) * source_relevance
                + weights.get("task_relevance", 0.0) * task_relevance
            )
            scored.append(
                RetrievedMemory(
                    memory_id=record.memory_id,
                    value=record.prototype,
                    score=score,
                    similarity_score=similarity,
                    recency_score=recency,
                    importance_score=importance,
                    confidence_score=confidence,
                    source_score=source_relevance,
                    retrieval_reason=f"semantic memory: {record.concept}",
                    metadata=record.metadata,
                )
            )
        if seed is not None:
            random.Random(seed).shuffle(scored)
        scored.sort(key=lambda item: (item.score, item.similarity_score), reverse=True)
        return scored[:limit]

    def consolidate(self, episodes: list[EpisodicMemoryItem], method: str = "frequency", threshold: float = 0.75, support: int = 2) -> SemanticMemoryItem | None:
        if len(episodes) < support:
            return None
        grouped: dict[tuple[str, str], list[EpisodicMemoryItem]] = {}
        for episode in episodes:
            key = (_normalize_text(episode.key), _normalize_text(str(episode.value)))
            grouped.setdefault(key, []).append(episode)
        chosen = None
        for (concept, prototype_text), group in grouped.items():
            avg_confidence = sum(item.confidence for item in group) / len(group)
            if len(group) >= support and avg_confidence >= threshold:
                consolidation_id = uuid.uuid4().hex
                record = SemanticMemoryItem(
                    memory_id=consolidation_id,
                    concept=concept,
                    prototype=group[0].value,
                    confidence=avg_confidence,
                    source_episode_ids=tuple(item.memory_id for item in group),
                    metadata={
                        "created_at": time.time(),
                        "source_count": len(group),
                        "method": method,
                    },
                    version=1,
                    embedding=_embed_text(concept, dim=self.embedding_dim),
                )
                self.records[consolidation_id] = record
                consolidation_record = ConsolidationRecord(
                    consolidation_id=consolidation_id,
                    source_episode_ids=record.source_episode_ids,
                    concept=concept,
                    prototype=record.prototype,
                    confidence=avg_confidence,
                    method=method,
                    created_at=record.metadata["created_at"],
                    revisions=(),
                    unresolved_disagreements=(),
                )
                self.consolidations[consolidation_id] = consolidation_record
                chosen = record
        return chosen

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        archive = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "records": [
                {
                    "memory_id": record.memory_id,
                    "concept": record.concept,
                    "prototype": record.prototype,
                    "confidence": record.confidence,
                    "source_episode_ids": list(record.source_episode_ids),
                    "metadata": record.metadata,
                    "version": record.version,
                    "embedding": record.embedding.tolist(),
                }
                for record in self.records.values()
            ],
            "consolidations": [
                {
                    "consolidation_id": consolidation.consolidation_id,
                    "source_episode_ids": list(consolidation.source_episode_ids),
                    "concept": consolidation.concept,
                    "prototype": consolidation.prototype,
                    "confidence": consolidation.confidence,
                    "method": consolidation.method,
                    "created_at": consolidation.created_at,
                    "revisions": [rev.__dict__ for rev in consolidation.revisions],
                    "unresolved_disagreements": list(consolidation.unresolved_disagreements),
                }
                for consolidation in self.consolidations.values()
            ],
        }
        content = json.dumps(archive, indent=2)
        checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
        payload = {"checksum": checksum, "archive": archive}
        temp = Path(tempfile.mkstemp(prefix="semantic_", suffix=".json", dir=path.parent)[1])
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp.replace(path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "SemanticMemory":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        checksum = payload.get("checksum", "")
        archive = payload.get("archive", {})
        if hashlib.sha256(json.dumps(archive, indent=2).encode("utf-8")).hexdigest() != checksum:
            raise ValueError("Semantic memory file checksum mismatch or corruption detected")
        memory = cls()
        for record_data in archive.get("records", []):
            embedding = np.asarray(record_data.get("embedding", []), dtype=np.float32)
            memory.records[record_data["memory_id"]] = SemanticMemoryItem(
                memory_id=record_data["memory_id"],
                concept=record_data["concept"],
                prototype=record_data["prototype"],
                confidence=record_data["confidence"],
                source_episode_ids=tuple(record_data.get("source_episode_ids", [])),
                metadata=record_data.get("metadata", {}),
                version=record_data.get("version", 1),
                embedding=embedding,
            )
        for consolidation_data in archive.get("consolidations", []):
            memory.consolidations[consolidation_data["consolidation_id"]] = ConsolidationRecord(
                consolidation_id=consolidation_data["consolidation_id"],
                source_episode_ids=tuple(consolidation_data.get("source_episode_ids", [])),
                concept=consolidation_data["concept"],
                prototype=consolidation_data.get("prototype"),
                confidence=consolidation_data.get("confidence", 0.0),
                method=consolidation_data.get("method", ""),
                created_at=consolidation_data.get("created_at", time.time()),
                revisions=tuple(MemoryRevision(**rev_data) for rev_data in consolidation_data.get("revisions", [])),
                unresolved_disagreements=tuple(consolidation_data.get("unresolved_disagreements", [])),
            )
        return memory


class MemorySystem:
    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()
        self.enabled = self.config.enabled
        self.working = WorkingMemory(capacity=self.config.working_capacity, expiration_steps=self.config.working_expiration_steps, embedding_dim=self.config.embedding_dim)
        self.episodic = EpisodicMemory(embedding_dim=self.config.embedding_dim)
        self.semantic = SemanticMemory(embedding_dim=self.config.embedding_dim)
        self.prior_use: dict[str, int] = {}

    def create_episode(self, key: str, value: Any, sequence_step: int, source: str = "system", importance: float = 0.5, confidence: float = 0.8, metadata: dict[str, Any] | None = None) -> str:
        if not self.enabled:
            return ""
        metadata = metadata or {}
        memory_id = uuid.uuid4().hex
        timestamp = time.time()
        embedding = _embed_text(str(value), dim=self.config.embedding_dim)
        episode = EpisodicMemoryItem(
            memory_id=memory_id,
            key=key,
            value=value,
            timestamp=timestamp,
            sequence_step=sequence_step,
            source=source,
            importance=importance,
            confidence=confidence,
            metadata=metadata,
            version=1,
            embedding=embedding,
        )
        self.episodic.insert(episode)
        self.working.insert(MemoryItem(
            memory_id=memory_id,
            key=key,
            value=value,
            timestamp=timestamp,
            sequence_step=sequence_step,
            source=source,
            importance=importance,
            confidence=confidence,
            metadata=metadata,
            version=1,
            embedding=embedding,
        ))
        return memory_id

    def revise(self, memory_id: str, new_value: Any, reason: str, source: str, evidence: dict[str, Any], confidence: float | None = None) -> MemoryRevision:
        revision = self.episodic.revise(
            memory_id=memory_id,
            new_value=new_value,
            reason=reason,
            source=source,
            evidence=evidence,
            confidence=confidence,
        )
        item = self.episodic.items[memory_id]
        self.working.insert(MemoryItem(
            memory_id=item.memory_id,
            key=item.key,
            value=item.value,
            timestamp=item.timestamp,
            sequence_step=item.sequence_step,
            source=item.source,
            importance=item.importance,
            confidence=item.confidence,
            metadata=item.metadata,
            version=item.version,
            embedding=item.embedding,
        ))
        return revision

    def retrieve(self, query: MemoryQuery, limit: int = 5, seed: int | None = None) -> list[RetrievedMemory]:
        if not self.enabled:
            return []
        semantic_results = self.semantic.retrieve(query, limit=limit, weights=self.config.retrieval_weights, seed=seed)
        episodic_results = self.episodic.retrieve(query, limit=limit, weights=self.config.retrieval_weights, seed=seed)
        working_results = self.working.retrieve(query, limit=limit, weights=self.config.retrieval_weights, seed=seed)
        combined = semantic_results + episodic_results + working_results
        unique: dict[str, RetrievedMemory] = {}
        for result in combined:
            existing = unique.get(result.memory_id)
            if existing is None or result.score > existing.score:
                unique[result.memory_id] = result
        results = sorted(unique.values(), key=lambda item: (item.score, item.similarity_score, item.recency_score), reverse=True)
        for result in results[:limit]:
            self.prior_use[result.memory_id] = self.prior_use.get(result.memory_id, 0) + 1
        return results[:limit]

    def retrieve_for_state(self, state: Any, task: str, limit: int | None = None) -> list[RetrievedMemory]:
        query_text = task
        if isinstance(state, dict) and "context" in state:
            query_text = f"{task} {state['context']}"
        return self.retrieve(MemoryQuery(query=query_text, task=task, limit=limit or self.config.memory_context_limit), limit=limit or self.config.memory_context_limit, seed=self.config.seed)

    def consolidate(self) -> SemanticMemoryItem | None:
        if not self.enabled:
            return None
        episodes = list(self.episodic.items.values())
        return self.semantic.consolidate(episodes, method="automatic", threshold=self.config.consolidation_threshold, support=self.config.consolidation_support)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.config.schema_version,
            "config": {**self.config.__dict__},
            "working": [
                {
                    "memory_id": item.memory_id,
                    "key": item.key,
                    "value": item.value,
                    "timestamp": item.timestamp,
                    "sequence_step": item.sequence_step,
                    "source": item.source,
                    "importance": item.importance,
                    "confidence": item.confidence,
                    "metadata": item.metadata,
                    "version": item.version,
                    "embedding": item.embedding.tolist(),
                }
                for item in self.working.items
            ],
        }
        episodic_payload = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "items": [
                {
                    "memory_id": item.memory_id,
                    "key": item.key,
                    "value": item.value,
                    "timestamp": item.timestamp,
                    "sequence_step": item.sequence_step,
                    "source": item.source,
                    "importance": item.importance,
                    "confidence": item.confidence,
                    "metadata": item.metadata,
                    "version": item.version,
                    "embedding": item.embedding.tolist(),
                }
                for item in self.episodic.items.values()
            ],
            "revisions": [
                {
                    "memory_id": rev.memory_id,
                    "previous_value": rev.previous_value,
                    "revised_value": rev.revised_value,
                    "reason": rev.reason,
                    "timestamp": rev.timestamp,
                    "actor": rev.actor,
                    "evidence": rev.evidence,
                    "version": rev.version,
                    "confidence_change": rev.confidence_change,
                }
                for revisions in self.episodic.revisions.values()
                for rev in revisions
            ],
            "conflicts": [conflict.__dict__ for conflict in self.episodic.conflicts.values()],
        }
        semantic_payload = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "records": [
                {
                    "memory_id": record.memory_id,
                    "concept": record.concept,
                    "prototype": record.prototype,
                    "confidence": record.confidence,
                    "source_episode_ids": list(record.source_episode_ids),
                    "metadata": record.metadata,
                    "version": record.version,
                    "embedding": record.embedding.tolist(),
                }
                for record in self.semantic.records.values()
            ],
            "consolidations": [
                {
                    "consolidation_id": consolidation.consolidation_id,
                    "source_episode_ids": list(consolidation.source_episode_ids),
                    "concept": consolidation.concept,
                    "prototype": consolidation.prototype,
                    "confidence": consolidation.confidence,
                    "method": consolidation.method,
                    "created_at": consolidation.created_at,
                    "revisions": [rev.__dict__ for rev in consolidation.revisions],
                    "unresolved_disagreements": list(consolidation.unresolved_disagreements),
                }
                for consolidation in self.semantic.consolidations.values()
            ],
        }
        wire = {
            "schema_version": self.config.schema_version,
            "config": self.config.__dict__,
            "working": payload["working"],
            "episodic": episodic_payload,
            "semantic": semantic_payload,
            "prior_use": self.prior_use,
        }
        content = json.dumps(wire, indent=2)
        checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
        final = {"checksum": checksum, "payload": wire}
        temp_file = Path(tempfile.mkstemp(prefix="memory_system_", suffix=".json", dir=path.parent)[1])
        temp_file.write_text(json.dumps(final, indent=2), encoding="utf-8")
        temp_file.replace(path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "MemorySystem":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        checksum = payload.get("checksum", "")
        wire = payload.get("payload", {})
        if hashlib.sha256(json.dumps(wire, indent=2).encode("utf-8")).hexdigest() != checksum:
            raise ValueError("Memory system file checksum mismatch or corruption detected")
        config = MemoryConfig(**wire.get("config", {}))
        system = cls(config=config)
        system.working.items = [
            MemoryItem(
                memory_id=item["memory_id"],
                key=item["key"],
                value=item["value"],
                timestamp=item["timestamp"],
                sequence_step=item["sequence_step"],
                source=item["source"],
                importance=item["importance"],
                confidence=item["confidence"],
                metadata=item["metadata"],
                version=item["version"],
                embedding=np.asarray(item["embedding"], dtype=np.float32),
            )
            for item in wire.get("working", [])
        ]
        episodic_archive = wire.get("episodic", {})
        for item in episodic_archive.get("items", []):
            system.episodic.items[item["memory_id"]] = EpisodicMemoryItem(
                memory_id=item["memory_id"],
                key=item["key"],
                value=item["value"],
                timestamp=item["timestamp"],
                sequence_step=item["sequence_step"],
                source=item["source"],
                importance=item["importance"],
                confidence=item["confidence"],
                metadata=item["metadata"],
                version=item["version"],
                embedding=np.asarray(item["embedding"], dtype=np.float32),
            )
        for rev_data in episodic_archive.get("revisions", []):
            rev = MemoryRevision(
                memory_id=rev_data["memory_id"],
                previous_value=rev_data["previous_value"],
                revised_value=rev_data["revised_value"],
                reason=rev_data["reason"],
                timestamp=rev_data["timestamp"],
                actor=rev_data["actor"],
                evidence=rev_data["evidence"],
                version=rev_data["version"],
                confidence_change=rev_data["confidence_change"],
            )
            system.episodic.revisions.setdefault(rev.memory_id, []).append(rev)
        for conflict_data in episodic_archive.get("conflicts", []):
            system.episodic.conflicts[conflict_data["conflict_id"]] = MemoryConflict(
                conflict_id=conflict_data["conflict_id"],
                memory_ids=tuple(conflict_data["memory_ids"]),
                conflict_type=conflict_data["conflict_type"],
                detected_at=conflict_data["detected_at"],
                status=conflict_data["status"],
                resolution=conflict_data.get("resolution"),
                resolution_confidence=conflict_data.get("resolution_confidence"),
                provenance=conflict_data.get("provenance", {}),
            )
        semantic_archive = wire.get("semantic", {})
        for record_data in semantic_archive.get("records", []):
            system.semantic.records[record_data["memory_id"]] = SemanticMemoryItem(
                memory_id=record_data["memory_id"],
                concept=record_data["concept"],
                prototype=record_data["prototype"],
                confidence=record_data["confidence"],
                source_episode_ids=tuple(record_data.get("source_episode_ids", [])),
                metadata=record_data.get("metadata", {}),
                version=record_data.get("version", 1),
                embedding=np.asarray(record_data.get("embedding", []), dtype=np.float32),
            )
        for consolidation_data in semantic_archive.get("consolidations", []):
            system.semantic.consolidations[consolidation_data["consolidation_id"]] = ConsolidationRecord(
                consolidation_id=consolidation_data["consolidation_id"],
                source_episode_ids=tuple(consolidation_data.get("source_episode_ids", [])),
                concept=consolidation_data["concept"],
                prototype=consolidation_data.get("prototype"),
                confidence=consolidation_data.get("confidence", 0.0),
                method=consolidation_data.get("method", ""),
                created_at=consolidation_data.get("created_at", time.time()),
                revisions=tuple(MemoryRevision(**rev_data) for rev_data in consolidation_data.get("revisions", [])),
                unresolved_disagreements=tuple(consolidation_data.get("unresolved_disagreements", [])),
            )
        system.prior_use = {str(k): int(v) for k, v in wire.get("prior_use", {}).items()}
        return system


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
