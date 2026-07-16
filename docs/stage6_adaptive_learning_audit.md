# Stage 6 Adaptive Learning Audit

## Existing Implementation

Stage 6 is currently implemented in `hrm/learning.py` and invoked by `hrm/theory.py::_run_stage_6`.

### Current behavior
- Captures baseline learning signals from `baseline_record` metrics beginning with `L_`.
- Records an adaptation signal entry in `LongTermMemory`.
- Adapts a `PreferenceModel` by scaling non-bias preference weights by `1 + adaptation_rate`.
- Computes `adaptation_score` as a bounded function of baseline signal strength, preference shift, and memory growth.

### Existing structures
- `LearningMetrics` records `signal_strength`, `preference_shift`, `memory_growth`, and `adaptation_score`.
- `PreferenceModel` stores interpreted preference weights and bias.
- `LearningSystem.update()` returns a deterministic preference update summary.

## Gaps relative to complete Stage 6

### Missing adaptation components
- No structured feedback capture from completed tasks.
- No task outcome or experience record abstractions.
- No replay buffer or prioritized replay mechanism.
- No separate candidate checkpoint training pipeline.
- No held-out evaluation on data excluded from candidate training.
- No regression protection or safety gate.
- No bounded update magnitude checks beyond a static adaptation rate.
- No rollback or checkpoint lineage.
- No append-only provenance logging.

### Unsupported claims
- Stage 6 currently claims "learning systems" but performs only a deterministic preference weight scaling.
- There is no evidence of parameter adaptation, held-out improvement, or catastrophic forgetting mitigation.
- The existing `adaptation_score` is heuristic and not tied to held-out evaluation.

## Current test coverage

### Verified behaviors
- Stage 6 returns the correct pipeline stage and structure.
- Learned preference weights are updated according to the deterministic rule.
- Memory growth is reported after adaptation.

### Not verified
- Feedback capture
- Experience persistence or replay
- Candidate training and checkpoint separation
- Evaluation on held-out data
- Regression protection
- Rollback capability
- Provenance completeness
- Adaptation improvement or safety guarantees

## Baseline conclusions

The current Stage 6 implementation is a valid deterministic preference update baseline, but it is not a completed adaptive learning stage under the requested criteria. A new package architecture is required to add experience replay, candidate adaptation, evaluation gates, rollback, and provenance while preserving the existing deterministic logic as a safe baseline.
