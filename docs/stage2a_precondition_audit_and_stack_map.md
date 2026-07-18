# Stage 2A Precondition Audit and Stack Map

## Precondition Audit

Status vocabulary: complete, partial, placeholder, absent, bypassed, test-only, incompatible.

1. Authoritative typed state substrate: complete
   - Evidence: `hrm/hrm_core/state.py` defines immutable typed block state and canonical serialization/digest.
2. Canonical transition application path: complete
   - Evidence: `hrm/hrm_core/transition.py` validates state, collects proposals, applies safety, and commits next state.
3. Safety gate and correction ledger: complete
   - Evidence: `hrm/hrm_core/safety.py` + correction integration in `hrm/hrm_core/transition.py`.
4. Reconstruction/ledger tolerance checks: complete
   - Evidence: residual + relative error checks in `hrm/hrm_core/transition.py`.
5. Determinism controls and reproducibility hooks: partial
   - Evidence: deterministic mode config and seeded pathways exist, but RNG state digest is recorded without full update lifecycle.
6. Stage 1 production path uses substrate: complete
   - Evidence: `_run_stage_1` in `hrm/theory.py` executes `run_spatial_reconstruction` and `run_sequence_memory`.
7. Stage 2 memory/planning integration with substrate state: partial
   - Evidence: legacy Stage 2 memory/planning remains separate from canonical request lifecycle by default.
8. Provider-neutral inference abstraction: complete (for Stage 2A vertical slice)
   - Evidence: `hrm/integration/providers.py` defines `InferenceProvider`, `MockDeterministicProvider`, `OllamaProvider`.
9. No direct LLM to response bypass in integrated request path: complete (for Stage 2A vertical slice)
   - Evidence: pipeline path in `hrm/integration/substrate_pipeline.py` always routes through governor and response envelope.
10. Governance and rendering layers (QuadraSeer/Starborn): complete (minimal implementation)
    - Evidence: `hrm/integration/governor.py` provides governance checks and rendering.
11. Single governed text request path with persisted checkpoint: complete (for Stage 2A vertical slice)
    - Evidence: `IntegratedRuntime.process_text_request` produces checkpoint and commits state once.
12. End-to-end tests for governed vertical slice: complete
    - Evidence: `tests/test_stage2a_governed_text.py`.

## Input-to-Response Stack Map

### Entry and control plane
- CLI entry: `hrm/runner.py` parses stage and stage-specific args.
- Stage dispatch: `hrm/theory.py` `run_stage` routes to stage handlers.

### Stage 2A governed text path
1. Request entry
   - `hrm/runner.py` stage 2 options: `--governed-text`, `--governed-input`, `--inference-provider`.
   - `hrm/theory.py` `_run_stage_2` toggles governed slice when enabled.
2. Plan generation and governance pre-check
   - `hrm/integration/substrate_pipeline.py` `_build_plan` creates executive plan.
   - `hrm/integration/governor.py` `validate_plan` approves/rejects plan.
3. State transition before inference
   - `hrm/integration/substrate_pipeline.py` calls canonical transition with `HRMInput` metadata.
4. Inference provider selection and call
   - `hrm/integration/providers.py` `choose_provider` selects by mode/provider id.
   - `run_inference_sync` executes provider call.
5. LLM proposal governance and envelope formation
   - Parsed output is normalized into `LLMProposal`.
   - `QuadraSeerGovernor.validate_llm_proposal` creates authoritative `ResponseEnvelope`.
6. Response rendering
   - `StarbornRenderer.render` formats response text from envelope.
7. State transition after inference
   - Second canonical transition applies response-linked metadata.
8. Authoritative commit and persistence
   - Runtime sets `self.state = tr2.state` after successful governance and transitions.
   - Checkpoint persisted to `checkpoints` path with transition/inference/ledger/state payload.
9. Return contract
   - `ProcessResult` returns response text, envelope, versions, provider, ledger metric, checkpoint path, transition id.

### Bypasses and convertibles
- Existing bypass retained outside Stage 2A: Stage 3/4/5/6 remain independent stage-specific orchestrations not yet unified under one canonical request lifecycle.
- Convertible points:
  - Stage 3 tool verification can be wrapped as a governed tool invocation inside the same request envelope protocol.
  - Stage 4 multimodal outputs can be fed as structured context into the same governed inference request contract.
  - Stage 5/6 outputs can be promoted to proposal channels governed by the same `ResponseEnvelope` schema.
