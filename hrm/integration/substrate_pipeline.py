from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hrm.hrm_core.configuration import HRMTransitionConfig
from hrm.hrm_core.experiments import build_engine
from hrm.hrm_core.mechanisms.base import HRMInput
from hrm.hrm_core.state import HRMState, make_initial_state, state_to_dict
from hrm.hrm_core.snapshot import freeze_state
from hrm.hrm_core.state import state_digest
from hrm.memory import _embed_text

from .governor import QuadraSeerGovernor, StarbornRenderer
from .providers import InferenceProvider, MockDeterministicProvider, choose_provider, run_inference_sync
from .types import (
    ExecutivePlanProposal,
    InferenceContext,
    InferenceRequest,
    LLMProposal,
    OperatingMode,
    ProcessResult,
    ProviderSelection,
)


@dataclass
class IntegratedRuntime:
    mode: OperatingMode = OperatingMode.SOVEREIGN_LOCAL
    preferred_provider: str = "mock_deterministic"
    session_id: str = "session-default"
    checkpoint_dir: Path = Path("checkpoints")

    def __post_init__(self) -> None:
        self.config = HRMTransitionConfig()
        self.engine = build_engine(self.config)
        self.state: HRMState = make_initial_state(
            node_count=16,
            channels=6,
            latent_dim=12,
            memory_capacity=16,
            seed=13,
        )
        self.governor = QuadraSeerGovernor()
        self.renderer = StarbornRenderer()
        self.providers: dict[str, InferenceProvider] = {}
        self.providers["mock_deterministic"] = MockDeterministicProvider()

    def register_provider(self, provider: InferenceProvider) -> None:
        self.providers[provider.provider_id] = provider

    def _build_plan(self, user_text: str) -> ExecutivePlanProposal:
        return self.governor.propose_execution_plan(user_text)

    @staticmethod
    def _build_field_drive(user_text: str, *, node_count: int, channels: int) -> np.ndarray:
        text_projection = _embed_text(user_text, dim=channels).astype(np.float32)
        return np.repeat(text_projection[None, :], node_count, axis=0)

    def _build_inference_request(self, user_text: str) -> InferenceRequest:
        schema = {
            "type": "object",
            "required": ["intent", "hypotheses", "response_draft", "confidence"],
        }
        return InferenceRequest(
            model_id="llama3.1:8b",
            messages=(
                {"role": "system", "content": "You are a local governed HRM assistant."},
                {"role": "user", "content": user_text},
            ),
            system_constraints=("json_first", "no_tool_execution"),
            expected_output_schema=schema,
            temperature=0.0,
            seed=7,
            token_limit=256,
            timeout_seconds=8.0,
            tool_policy={"allow": []},
            privacy_policy={"allow_remote": False},
            cost_budget=0.01,
            required_provenance_fields=("provider", "model", "seed"),
        )

    def process_text_request(self, user_text: str) -> ProcessResult:
        transition_id = f"tr-{uuid.uuid4().hex[:12]}"
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        state_before = self.state
        snapshot = freeze_state(state_before)

        plan = self._build_plan(user_text)
        plan_decision = self.governor.validate_plan(plan)
        if not plan_decision.accepted:
            raise RuntimeError(f"Plan rejected: {plan_decision.reason}")

        drive = self._build_field_drive(user_text, node_count=state_before.phi.phi.shape[0], channels=state_before.phi.phi.shape[1])

        provider_selection: ProviderSelection = choose_provider(
            mode=self.mode,
            preferred_provider=self.preferred_provider,
            providers=self.providers,
            allow_remote_fallback=self.mode in {
                OperatingMode.LOCAL_WITH_OPTIONAL_REMOTE_FALLBACK,
                OperatingMode.HYBRID,
                OperatingMode.DEVELOPMENT,
            },
        )
        provider = self.providers[provider_selection.provider_id]
        inference_request = self._build_inference_request(user_text)
        inference_context = InferenceContext(
            request_id=request_id,
            session_id=self.session_id,
            mode=self.mode,
            allow_remote_fallback=self.mode != OperatingMode.SOVEREIGN_LOCAL,
        )
        inference_result = run_inference_sync(provider, inference_request, inference_context)
        parsed = inference_result.parsed_output or {
            "intent": "answer_user_query",
            "hypotheses": ("text_only",),
            "response_draft": inference_result.raw_output or "No response available.",
            "confidence": 0.4,
            "memory_write_candidate": None,
            "tool_action_candidate": None,
        }
        llm_proposal = LLMProposal(
            proposal_id=f"llm-{uuid.uuid4().hex[:12]}",
            intent=str(parsed.get("intent", "answer_user_query")),
            hypotheses=tuple(parsed.get("hypotheses", ("none",))),
            response_draft=str(parsed.get("response_draft", "No response available.")),
            confidence=float(parsed.get("confidence", 0.4)),
            memory_write_candidate=parsed.get("memory_write_candidate"),
            tool_action_candidate=parsed.get("tool_action_candidate"),
            provenance={
                "provider": inference_result.provider_id,
                "model": inference_result.model_id,
                "seed": inference_result.seed,
                "latency_seconds": inference_result.latency_seconds,
                "parse_status": inference_result.parse_status,
            },
        )

        governor_decision = self.governor.validate_llm_proposal(
            request_id=request_id,
            session_id=self.session_id,
            proposal=llm_proposal,
            memory_references_used=(plan.memory_query,),
            unresolved_uncertainties=(provider_selection.reason,) if provider_selection.fallback_used else (),
        )
        if not governor_decision.accepted or governor_decision.envelope is None:
            raise RuntimeError(f"LLM proposal rejected: {governor_decision.reason}")

        hrm_input = HRMInput(
            field_drive=drive,
            metadata={
                "request_id": request_id,
                "transition_id": transition_id,
                "plan_id": plan.proposal_id,
                "memory_query": plan.memory_query,
                "intent": llm_proposal.intent,
                "hypotheses": llm_proposal.hypotheses,
                "response_draft": llm_proposal.response_draft,
                "memory_write_candidate": llm_proposal.memory_write_candidate,
                "provider_id": provider.provider_id,
                "provider_selection_reason": provider_selection.reason,
                "fallback_used": provider_selection.fallback_used,
                "sovereignty_reduced": provider_selection.sovereignty_reduced,
                "response_digest": governor_decision.envelope.provenance_digest,
                "state_digest_before": state_digest(snapshot.state),
            },
        )

        tr = self.engine.step(state_before, hrm_input)
        self.state = tr.state

        response_text = self.renderer.render(governor_decision.envelope)

        checkpoint_payload = {
            "transition_id": transition_id,
            "request_id": request_id,
            "session_id": self.session_id,
            "state_version_before": state_before.version,
            "state_version_after": self.state.version,
            "state_digest_before": state_digest(snapshot.state),
            "state_digest_after": state_digest(self.state),
            "response_envelope": {
                "request_id": governor_decision.envelope.request_id,
                "session_id": governor_decision.envelope.session_id,
                "answer_content": governor_decision.envelope.answer_content,
                "confidence": governor_decision.envelope.confidence,
                "safety_status": governor_decision.envelope.safety_status,
                "provenance_digest": governor_decision.envelope.provenance_digest,
            },
            "ledger": {
                "max_rel": tr.ledger.max_rel_reconstruction_error,
                "max_abs": tr.ledger.max_abs_reconstruction_error,
            },
            "inference": {
                "provider": inference_result.provider_id,
                "model": inference_result.model_id,
                "parse_status": inference_result.parse_status,
                "selection": provider_selection.__dict__,
            },
            "state": state_to_dict(self.state),
            "created_at": time.time(),
        }
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / f"{request_id}.json"
        checkpoint_path.write_text(json.dumps(checkpoint_payload, sort_keys=True), encoding="utf-8")

        return ProcessResult(
            response_text=response_text,
            response_envelope=governor_decision.envelope,
            state_version_before=state_before.version,
            state_version_after=self.state.version,
            ledger_max_rel_error=tr.ledger.max_rel_reconstruction_error,
            checkpoint_path=str(checkpoint_path),
            provider_used=inference_result.provider_id,
            transition_id=transition_id,
            observability={
                "request_id": request_id,
                "transition_runtime": tr.ledger.runtime_seconds,
                "state_digest_before": state_digest(snapshot.state),
                "state_digest_after": state_digest(self.state),
                "provider_selection": provider_selection.__dict__,
                "mechanism_activations": tr.ledger.proposal_activations,
            },
        )

    def process_tool_request(
        self,
        *,
        objective: str,
        tool_operations: list[tuple[str, dict[str, Any]]],
        tool_executor: Callable[[str, dict[str, Any]], dict[str, Any]],
    ) -> ProcessResult:
        transition_id = f"tr-{uuid.uuid4().hex[:12]}"
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        state_before = self.state

        required_tools = tuple(op[0] for op in tool_operations)
        plan = ExecutivePlanProposal(
            proposal_id=f"plan-{uuid.uuid4().hex[:12]}",
            task="governed_tool_verification",
            intent="verify_and_report_tool_execution",
            risk_level="medium" if "execute_code" in required_tools else "low",
            required_tools=required_tools,
            memory_query=f"tools::{objective[:24]}",
            confidence=0.74,
            rationale="governed tool execution path with envelope response",
        )
        plan_decision = self.governor.validate_plan(plan)
        if not plan_decision.accepted:
            raise RuntimeError(f"Plan rejected: {plan_decision.reason}")

        drive_value = min(1.0, max(0.0, float(len(objective)) / 256.0))
        drive = np.full((16, 6), drive_value, dtype=np.float32)
        tr = self.engine.step(
            self.state,
            HRMInput(field_drive=drive, metadata={"memory_query": plan.memory_query, "transition_id": transition_id}),
        )

        executed_tools: list[dict[str, Any]] = []
        tool_refs: list[str] = []
        uncertainties: list[str] = []
        for index, (tool_name, arguments) in enumerate(tool_operations):
            execution = tool_executor(tool_name, arguments)
            ref = f"{tool_name}:{index}"
            tool_refs.append(ref)
            success = bool(execution.get("success", False))
            if not success:
                uncertainties.append(f"{ref} failed")
            executed_tools.append(
                {
                    "ref": ref,
                    "tool": tool_name,
                    "arguments": arguments,
                    "success": success,
                    "output": str(execution.get("output", ""))[:1000],
                    "error": execution.get("error"),
                }
            )

        provider_selection = choose_provider(
            mode=self.mode,
            preferred_provider=self.preferred_provider,
            providers=self.providers,
            allow_remote_fallback=self.mode in {
                OperatingMode.LOCAL_WITH_OPTIONAL_REMOTE_FALLBACK,
                OperatingMode.HYBRID,
                OperatingMode.DEVELOPMENT,
            },
        )
        provider = self.providers[provider_selection.provider_id]
        inference_context = InferenceContext(
            request_id=request_id,
            session_id=self.session_id,
            mode=self.mode,
            allow_remote_fallback=self.mode != OperatingMode.SOVEREIGN_LOCAL,
        )
        inference_request = self._build_inference_request(
            "\n".join(
                [
                    f"Objective: {objective}",
                    "Summarize governed tool outcomes.",
                    f"Tool outcomes JSON: {json.dumps(executed_tools, sort_keys=True)}",
                ]
            )
        )
        inference_result = run_inference_sync(provider, inference_request, inference_context)
        parsed = inference_result.parsed_output or {
            "intent": "verify_and_report_tool_execution",
            "hypotheses": ("tool_summary_text_only",),
            "response_draft": f"Governed tool execution completed for {len(executed_tools)} tools.",
            "confidence": 0.45,
            "memory_write_candidate": None,
            "tool_action_candidate": None,
        }

        llm_proposal = LLMProposal(
            proposal_id=f"llm-{uuid.uuid4().hex[:12]}",
            intent=str(parsed.get("intent", "verify_and_report_tool_execution")),
            hypotheses=tuple(parsed.get("hypotheses", ("none",))),
            response_draft=str(parsed.get("response_draft", "Governed tool execution completed.")),
            confidence=float(parsed.get("confidence", 0.45)),
            memory_write_candidate=parsed.get("memory_write_candidate"),
            tool_action_candidate=parsed.get("tool_action_candidate"),
            provenance={
                "provider": inference_result.provider_id,
                "model": inference_result.model_id,
                "seed": inference_result.seed,
                "parse_status": inference_result.parse_status,
                "latency_seconds": inference_result.latency_seconds,
                "executed_tools": len(executed_tools),
            },
        )

        gov2 = self.governor.validate_llm_proposal(
            request_id=request_id,
            session_id=self.session_id,
            proposal=llm_proposal,
            memory_references_used=(plan.memory_query,),
            tool_results_used=tuple(tool_refs),
            unresolved_uncertainties=tuple(uncertainties),
        )
        if not gov2.accepted or gov2.envelope is None:
            raise RuntimeError(f"LLM proposal rejected: {gov2.reason}")

        response_text = self.renderer.render(gov2.envelope)
        self.state = tr.state

        checkpoint_payload = {
            "transition_id": transition_id,
            "request_id": request_id,
            "session_id": self.session_id,
            "state_version_before": state_before.version,
            "state_version_after": self.state.version,
            "response_envelope": {
                "request_id": gov2.envelope.request_id,
                "session_id": gov2.envelope.session_id,
                "answer_content": gov2.envelope.answer_content,
                "confidence": gov2.envelope.confidence,
                "safety_status": gov2.envelope.safety_status,
                "provenance_digest": gov2.envelope.provenance_digest,
                "tool_results_used": list(gov2.envelope.tool_results_used),
            },
            "ledger": {
                "max_rel": tr.ledger.max_rel_reconstruction_error,
                "max_abs": tr.ledger.max_abs_reconstruction_error,
            },
            "inference": {
                "provider": inference_result.provider_id,
                "model": inference_result.model_id,
                "parse_status": inference_result.parse_status,
                "selection": provider_selection.__dict__,
            },
            "tool_execution": executed_tools,
            "state": state_to_dict(self.state),
            "created_at": time.time(),
        }
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / f"{request_id}.json"
        checkpoint_path.write_text(json.dumps(checkpoint_payload, sort_keys=True), encoding="utf-8")

        return ProcessResult(
            response_text=response_text,
            response_envelope=gov2.envelope,
            state_version_before=state_before.version,
            state_version_after=self.state.version,
            ledger_max_rel_error=tr.ledger.max_rel_reconstruction_error,
            checkpoint_path=str(checkpoint_path),
            provider_used=inference_result.provider_id,
            transition_id=transition_id,
            observability={
                "request_id": request_id,
                "transition_runtime": tr.ledger.runtime_seconds,
                "tool_execution": executed_tools,
                "provider_selection": provider_selection.__dict__,
            },
        )
