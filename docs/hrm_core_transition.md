# HRM Core Computational Substrate

This document defines the production HRM computational substrate now implemented in the repository.

## Scope and Claims Boundary

- Mathematical specification is not empirical validation.
- Passing tests establishes implementation integrity, not intelligence.
- Synthetic benchmark benefit is limited to those benchmark tasks.
- General intelligence and consciousness are not established.

## Authoritative State

The production state is a typed seven-block state:

- Phi: distributed field state
- G: geometry and metric state
- T: topology and structural state
- M: memory state
- C: cognition state
- H: hierarchy state
- B: resource and budget state

The implementation lives in hrm/hrm_core/state.py.

## Canonical Transition

The production transition is implemented in hrm/hrm_core/transition.py with the sequence:

1. Validate admissible state invariants.
2. Freeze immutable snapshot.
3. Capture deterministic RNG provenance.
4. Compute shared observables.
5. Collect side-effect-free mechanism proposals.
6. Aggregate provisional continuous updates.
7. Apply safe-transition policy and corrections.
8. Reconstruct and validate ledger residual/tolerance.
9. Commit the new authoritative state exactly once.

## Mechanism Proposal Contract

Mechanisms implement the protocol in hrm/hrm_core/mechanisms/base.py and return MechanismProposal records with:

- mechanism id
- source state version
- read/write block declarations
- activation
- block-structured delta
- dtype/device
- estimated cost
- diagnostics and provenance

Mechanisms must not mutate shared state during proposal generation.

## Interaction Contract

Cross-mechanism interactions are explicit in the transition engine. In this first fixed-carrier phase, interactions are represented as explicit zero or declared terms and never rely on Python call order side effects.

## Event Lifecycle and Transport

This first deliverable is fixed-carrier and fixed-topology. Structural event handling and non-identity transport are intentionally deferred. Carrier-compatibility invariants are already enforced.

## Safe-Transition Policy

The policy in hrm/hrm_core/safety.py detects and handles:

- non-finite values
- bounded projection for field and cognition
- field norm guardrails
- low-variance collapse warnings

All interventions are recorded as corrections in the ledger.

## Ledger Reconstruction

Each transition records named contribution owners and computes residual error with absolute and relative maxima. Transitions fail if reconstruction exceeds configured tolerance.

## Determinism

Deterministic replay is default. State includes RNG provenance and deterministic updates from snapshot + input + config.

## Ablations

Configuration-level gates are available for:

- input projection
- diffusion
- reaction
- memory
- cognition
- hierarchy

## Benchmark Protocol (Initial)

Two controlled experiments are included:

- spatial reconstruction task: fixed-graph field dynamics
- sequence-memory task: memory/cognition activation on repetitive symbolic drives

Both run through the production canonical transition path.

## Extending with New Mechanisms

Add a class implementing HRMMechanism and return MechanismProposal with explicit read/write block sets and deterministic diagnostics. Register the mechanism in hrm/hrm_core/experiments.py build_engine or equivalent production factory.
