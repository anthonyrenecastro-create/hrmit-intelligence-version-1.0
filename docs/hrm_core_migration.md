# Migration Notes: Legacy Stages to HRM Core

## Summary

The repository now includes an authoritative HRM core state and canonical transition engine under hrm/hrm_core.

Stage 1 execution in hrm/theory.py now runs through the HRM core canonical transition path and emits controlled experiment metrics and ledger diagnostics.

## Mapping

- Legacy Stage 1 baseline orchestration -> canonical transition in hrm/hrm_core/transition.py
- Legacy field-style dynamics in hrm/baseline.py -> mechanismized contributions in hrm/hrm_core/mechanisms/*
- Memory and cognition side modules -> proposal providers writing to M and C blocks

## Compatibility

- Existing Stage 2-6 stage interfaces remain available.
- Legacy baseline module remains present and can be used for backward compatibility analysis.

## Checkpoint Compatibility

- Legacy baseline artifacts are not silently treated as authoritative HRM state checkpoints.
- Authoritative HRM state serialization format is defined by hrm/hrm_core/state.py (state_to_dict/state_from_dict).

## Deprecation Direction

- Standalone shadow transitions should be considered compatibility mode.
- New experiments should use hrm/hrm_core canonical transition only.
