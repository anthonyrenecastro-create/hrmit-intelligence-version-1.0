from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransitionRollback:
    reason: str
    source_version: int


def rollback_to_prior(prior_state):
    return prior_state
