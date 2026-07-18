from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .types import ExecutivePlanProposal, LLMProposal, ResponseEnvelope


@dataclass(frozen=True)
class GovernanceDecision:
    accepted: bool
    reason: str
    safety_status: str
    envelope: ResponseEnvelope | None


class QuadraSeerGovernor:
    def propose_execution_plan(self, user_text: str, *, risk_level: str = "low", required_tools: tuple[str, ...] = ()) -> ExecutivePlanProposal:
        return ExecutivePlanProposal(
            proposal_id=f"plan-{hashlib.sha256(user_text.encode('utf-8')).hexdigest()[:12]}",
            task="respond_to_user",
            intent="answer_user_query",
            risk_level=risk_level,
            required_tools=required_tools,
            memory_query=f"query::{user_text[:24]}",
            confidence=0.75,
            rationale="governed single-turn user request",
        )

    def validate_plan(self, plan: ExecutivePlanProposal) -> GovernanceDecision:
        if plan.risk_level.lower() in {"high", "critical"} and "execute_code" in plan.required_tools:
            return GovernanceDecision(
                accepted=False,
                reason="Rejected high-risk plan with code execution tool requirement",
                safety_status="rejected",
                envelope=None,
            )
        return GovernanceDecision(accepted=True, reason="plan_accepted", safety_status="ok", envelope=None)

    def resolve_conflicts(self, proposals: tuple[LLMProposal, ...]) -> LLMProposal:
        if not proposals:
            raise ValueError("At least one proposal is required for conflict resolution")
        return sorted(proposals, key=lambda proposal: (-proposal.confidence, proposal.proposal_id))[0]

    def validate_llm_proposal(
        self,
        *,
        request_id: str,
        session_id: str,
        proposal: LLMProposal,
        memory_references_used: tuple[str, ...] = (),
        tool_results_used: tuple[str, ...] = (),
        unresolved_uncertainties: tuple[str, ...] = (),
        response_style: str = "concise_professional",
    ) -> GovernanceDecision:
        if not proposal.response_draft.strip():
            return GovernanceDecision(
                accepted=False,
                reason="empty_response_draft",
                safety_status="rejected",
                envelope=None,
            )
        unsafe_terms = ("rm -rf", "DROP TABLE", "exfiltrate secret")
        if any(term.lower() in proposal.response_draft.lower() for term in unsafe_terms):
            return GovernanceDecision(
                accepted=False,
                reason="unsafe_content_blocked",
                safety_status="rejected",
                envelope=None,
            )

        provenance_digest = hashlib.sha256(
            json.dumps(proposal.provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        envelope = ResponseEnvelope(
            request_id=request_id,
            session_id=session_id,
            answer_content=proposal.response_draft,
            confidence=float(max(0.0, min(1.0, proposal.confidence))),
            unresolved_uncertainties=unresolved_uncertainties,
            memory_references_used=memory_references_used,
            tool_results_used=tool_results_used,
            internal_task_status="completed",
            safety_status="ok",
            provenance_digest=provenance_digest,
            response_style=response_style,
            clarification_needed=False,
        )
        return GovernanceDecision(accepted=True, reason="proposal_accepted", safety_status="ok", envelope=envelope)


class StarbornRenderer:
    def render(self, envelope: ResponseEnvelope) -> str:
        base = envelope.answer_content.strip()
        if not base:
            base = "No answer content generated."
        return f"{base}\n\n[confidence={envelope.confidence:.2f} safety={envelope.safety_status}]"
