from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .configuration import HRMTransitionConfig
from .state import HRMState


@dataclass(frozen=True)
class InvariantReport:
    valid: bool
    errors: tuple[str, ...]


def validate_state(state: HRMState, config: HRMTransitionConfig) -> InvariantReport:
    errors: list[str] = []
    phi = state.phi.phi
    if phi.ndim != 2:
        errors.append("Phi must be rank-2 [node, channel]")
    if not np.isfinite(phi).all():
        errors.append("Phi contains non-finite values")
    if not np.isfinite(state.memory.working).all():
        errors.append("Memory working block contains non-finite values")
    if not np.isfinite(state.cognition.latent).all():
        errors.append("Cognition latent contains non-finite values")
    if state.geometry.laplacian.shape[0] != state.geometry.laplacian.shape[1]:
        errors.append("Geometry Laplacian must be square")
    if state.geometry.laplacian.shape[0] != phi.shape[0]:
        errors.append("Geometry Laplacian node count must match Phi nodes")
    if state.memory.working.shape != phi.shape:
        errors.append("Memory working shape must match Phi shape")
    if state.memory.associative_keys.shape[0] != state.memory.capacity:
        errors.append("Memory associative key capacity must match declared capacity")
    if state.memory.associative_values.shape != state.memory.associative_keys.shape:
        errors.append("Memory associative values shape must match associative keys shape")
    if state.cognition.latent.ndim != 1:
        errors.append("Cognition latent must be rank-1")
    if state.cognition.prediction.shape != (phi.shape[1],):
        errors.append("Cognition prediction width must match Phi channels")
    if state.cognition.residual.shape != state.cognition.prediction.shape:
        errors.append("Cognition residual shape must match prediction shape")
    if state.cognition.uncertainty.shape != state.cognition.prediction.shape:
        errors.append("Cognition uncertainty shape must match prediction shape")
    if state.hierarchy.coarse.ndim != 2:
        errors.append("Hierarchy coarse state must be rank-2")
    if state.hierarchy.restriction.shape[1] != phi.shape[0]:
        errors.append("Hierarchy restriction input width must match Phi nodes")
    if state.hierarchy.prolongation.shape[0] != phi.shape[0]:
        errors.append("Hierarchy prolongation output width must match Phi nodes")
    if state.budget.remaining_budget < 0:
        errors.append("Remaining budget cannot be negative")
    if state.budget.active_width > phi.shape[1]:
        errors.append("Active width cannot exceed Phi channel width")
    if state.budget.active_width <= 0:
        errors.append("Active width must be positive")
    if state.topology.node_count != phi.shape[0]:
        errors.append("Topology node count must match Phi nodes")
    if state.topology.edge_count < 0:
        errors.append("Topology edge count cannot be negative")
    if state.dtype != str(phi.dtype):
        errors.append("State dtype metadata does not match Phi dtype")
    if np.linalg.norm(phi) > config.max_field_norm * 10.0:
        errors.append("Phi norm exceeds hard admissible guardrail")
    return InvariantReport(valid=not errors, errors=tuple(errors))
