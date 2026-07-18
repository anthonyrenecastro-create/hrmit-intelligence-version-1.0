from __future__ import annotations

import numpy as np

from .state import HRMState


def state_metrics(state: HRMState) -> dict[str, float]:
    phi = state.phi.phi
    field_norm = float(np.linalg.norm(phi))
    field_var = float(np.var(phi))
    field_energy = float(np.mean(phi * phi))
    coherence = float(1.0 / (1.0 + np.mean((phi[1:] - phi[:-1]) ** 2))) if phi.shape[0] > 1 else 1.0
    collapse_risk = float(max(0.0, 1e-3 - field_var))
    return {
        "field_norm": field_norm,
        "field_variance": field_var,
        "field_energy": field_energy,
        "coherence": coherence,
        "collapse_risk": collapse_risk,
        "memory_norm": float(np.linalg.norm(state.memory.working)),
        "cognition_norm": float(np.linalg.norm(state.cognition.latent)),
        "hierarchy_norm": float(np.linalg.norm(state.hierarchy.coarse)),
        "budget_remaining": float(state.budget.remaining_budget),
        "budget_cumulative_cost": float(state.budget.cumulative_cost),
        "active_width": float(state.budget.active_width),
    }
