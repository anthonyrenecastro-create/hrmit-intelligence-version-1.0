# Stage 2 and Stage 3 Initial Audit

## Current Functionality

- `hrm/memory.py` implements a simple `LongTermMemory` with:
  - `MemoryEntry` dataclass containing `key`, `content`, `metadata`, and `embedding`
  - `LongTermMemory.add_entry`, `retrieve`, `save`, `load`
  - cosine similarity retrieval using deterministic SHA256-based text embeddings
  - persistence as JSON via `save` and `load`
- `hrm/tools.py` implements a naive tool registry with:
  - `ToolRegistry` and `ToolResult`
  - built-in tools: `echo`, `sum`, `python`, `api_call`
  - in-process Python execution via `eval` / `exec`
  - naive API connector stub and schema validation if `jsonschema` installed
- `hrm/theory.py` orchestrates stage 2 and stage 3 using the simple memory and tool modules.

## Missing Requirements

- No explicit `WorkingMemory`, `EpisodicMemory`, or `SemanticMemory` classes.
- No typed shared memory models beyond `MemoryEntry`.
- Retrieval scoring is only similarity; no configurable weighted scoring with recency, importance, confidence, source relevance, task relevance, or conflict penalty.
- No memory revision history or conflict records.
- No explicit consolidation pipeline or semantic memory generation.
- Persistence is one-process JSON save/load without versioning, atomic writes, checksum, corruption detection, or encryption hooks.
- No process restart tests or subprocess-based memory restoration.
- No permission model, tool effect/risk classification, path policy, or audit logging.
- Tool execution is not isolated; Python code executes in-process with only a minimal safe global environment.
- No filesystem allowlist, path-traversal protection, or verified mutation checks.
- No append-only audit log, chained hashes, or secret redaction.
- No deterministic local mock HTTP server or network allowlist enforcement.
- Documentation and benchmark scaffolding absent.

## Security Risks

- In-process Python evaluation can execute arbitrary code in the current interpreter.
- `api_call` allows external network access without explicit allowlist enforcement.
- No permission checks mean any code path may mutate files or external systems.
- Save/load persistence has no integrity checks or version control.
- Path handling may permit absolute or symlink traversal through file tools once implemented.

## Test Gaps

- No tests for explicit memory type behavior or long-horizon recall.
- No tests for delayed recall, distractor resistance, revision, conflict handling, consolidation, or ablation.
- No tests for subprocess sandboxing, resource limits, or tool permissions.
- No tests for path traversal, audit logging, redaction, or mock API server behavior.

## Files to Modify

- `hrm/memory.py`
- `hrm/tools.py`
- `hrm/theory.py`
- `tests/test_stage5_stage6.py` (possibly to preserve compatibility)
- plus new tests under `tests/memory/` and `tests/tools/`
- new benchmark/ config/ docs/ files

## Migration Risks

- Existing `LongTermMemory.save/load` JSON semantics must remain usable for current stage2 setup if possible.
- New interfaces should preserve `LongTermMemory.from_baseline_record` and `Planner` for backward compatibility.
- Stage 2 and 3 architecture changes may require adapting `hrm/theory.py` stage outputs carefully.

## Compatibility Considerations

- Keep `LongTermMemory.retrieve(query, k)` semantics while adding a richer retrieval API.
- Preserve `ToolRegistry.register_builtin_tools()` and the existing stage 3 smoke flow until new secure executor is added.
- Avoid breaking `HRMTheory.run_stage` signatures for existing tests.
