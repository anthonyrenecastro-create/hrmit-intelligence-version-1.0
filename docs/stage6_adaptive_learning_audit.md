# Stage 6 Adaptive-Learning Audit

## Baseline inspected

Repository `anthonyrenecastro-create/hrmit-intelligence-version-1.0`, branch `main`, commit `c3a36f4`.

`hrm/learning.py` extracted `L_*` values, multiplied each preference by `1 + learning_rate`, incremented bias, optionally appended one memory entry, and calculated a bounded descriptive score. The update was deterministic, interpretable, and bounded only indirectly.

The Stage 6 test asserted the hard-coded result `safety == 0.432`, memory growth, and a nonnegative adaptation score. It did not test completed-task feedback, replay, learned parameters, held-out improvement, prior-task regression, candidate isolation, promotion, rollback, calibration, or provenance.

The baseline is therefore classified as a controlled heuristic preference update, not empirical adaptive learning.

## New direction

The replacement separates immutable task/feedback/experience/candidate/evaluation records, experience persistence and replay, a narrowly scoped tunable adapter, norm-clipped candidate training, held-out and regression evaluation, promotion gating, rollback, and complete decision history. The deterministic preference update may remain as a compatibility baseline but cannot establish Stage 6 completion.
