from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OperatingMode(str, Enum):
    SOVEREIGN_LOCAL = "sovereign_local"
    LOCAL_WITH_OPTIONAL_REMOTE_FALLBACK = "local_with_optional_remote_fallback"
    HYBRID = "hybrid"
    DEVELOPMENT = "development"
    TEST = "test"


@dataclass(frozen=True)
class InferenceRequest:
    model_id: str
    messages: tuple[dict[str, str], ...]
    system_constraints: tuple[str, ...]
    expected_output_schema: dict[str, Any]
    temperature: float
    seed: int | None
    token_limit: int
    timeout_seconds: float
    tool_policy: dict[str, Any]
    privacy_policy: dict[str, Any]
    cost_budget: float
    required_provenance_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class InferenceContext:
    request_id: str
    session_id: str
    mode: OperatingMode
    allow_remote_fallback: bool


@dataclass(frozen=True)
class ProviderSelection:
    provider_id: str
    fallback_used: bool
    sovereignty_reduced: bool
    reason: str


@dataclass(frozen=True)
class InferenceResult:
    raw_output: str
    parsed_output: dict[str, Any] | None
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    model_id: str
    provider_id: str
    sampling: dict[str, Any]
    seed: int | None
    parse_status: str
    safety_status: str
    retry_history: tuple[str, ...]
    provenance: dict[str, Any]
    error_state: str | None = None


@dataclass(frozen=True)
class ExecutivePlanProposal:
    proposal_id: str
    task: str
    intent: str
    risk_level: str
    required_tools: tuple[str, ...]
    memory_query: str
    confidence: float
    rationale: str


@dataclass(frozen=True)
class LLMProposal:
    proposal_id: str
    intent: str
    hypotheses: tuple[str, ...]
    response_draft: str
    confidence: float
    memory_write_candidate: str | None
    tool_action_candidate: dict[str, Any] | None
    provenance: dict[str, Any]


@dataclass(frozen=True)
class ResponseEnvelope:
    request_id: str
    session_id: str
    answer_content: str
    confidence: float
    unresolved_uncertainties: tuple[str, ...]
    memory_references_used: tuple[str, ...]
    tool_results_used: tuple[str, ...]
    internal_task_status: str
    safety_status: str
    provenance_digest: str
    response_style: str
    clarification_needed: bool


@dataclass(frozen=True)
class ProcessResult:
    response_text: str
    response_envelope: ResponseEnvelope
    state_version_before: int
    state_version_after: int
    ledger_max_rel_error: float
    checkpoint_path: str
    provider_used: str
    transition_id: str
    observability: dict[str, Any] = field(default_factory=dict)
