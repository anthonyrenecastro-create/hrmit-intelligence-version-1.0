from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .snapshot import HRMSnapshot


@dataclass(frozen=True)
class HRMObservables:
    field_norm: float
    field_variance: float
    field_energy: float
    coherence: float
    collapse_risk: float


def compute_observables(snapshot: HRMSnapshot) -> HRMObservables:
    phi = snapshot.state.phi.phi
    field_norm = float(np.linalg.norm(phi))
    field_variance = float(np.var(phi))
    field_energy = float(np.mean(phi * phi))
    if phi.shape[0] > 1:
        disagreement = np.mean((phi[1:] - phi[:-1]) ** 2)
    else:
        disagreement = 0.0
    coherence = float(1.0 / (1.0 + disagreement))
    collapse_risk = float(max(0.0, 1e-3 - field_variance))
    return HRMObservables(
        field_norm=field_norm,
        field_variance=field_variance,
        field_energy=field_energy,
        coherence=coherence,
        collapse_risk=collapse_risk,
    )
