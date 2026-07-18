from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .state import HRMState


@dataclass(frozen=True)
class HRMSnapshot:
    state: HRMState
    source_version: int


def _freeze_array(array: np.ndarray) -> np.ndarray:
    frozen = np.array(array, copy=True)
    frozen.setflags(write=False)
    return frozen


def freeze_state(state: HRMState) -> HRMSnapshot:
    frozen = HRMState(
        version=state.version,
        step=state.step,
        dtype=state.dtype,
        device=state.device,
        rng_state=deepcopy(state.rng_state),
        phi=type(state.phi)(phi=_freeze_array(state.phi.phi)),
        geometry=type(state.geometry)(
            laplacian=_freeze_array(state.geometry.laplacian),
            metric_scale=state.geometry.metric_scale,
        ),
        topology=state.topology,
        memory=type(state.memory)(
            working=_freeze_array(state.memory.working),
            associative_keys=_freeze_array(state.memory.associative_keys),
            associative_values=_freeze_array(state.memory.associative_values),
            capacity=state.memory.capacity,
            write_index=state.memory.write_index,
        ),
        cognition=type(state.cognition)(
            latent=_freeze_array(state.cognition.latent),
            prediction=_freeze_array(state.cognition.prediction),
            residual=_freeze_array(state.cognition.residual),
            uncertainty=_freeze_array(state.cognition.uncertainty),
        ),
        hierarchy=type(state.hierarchy)(
            coarse=_freeze_array(state.hierarchy.coarse),
            restriction=_freeze_array(state.hierarchy.restriction),
            prolongation=_freeze_array(state.hierarchy.prolongation),
            gain=state.hierarchy.gain,
        ),
        budget=state.budget,
    )
    return HRMSnapshot(state=frozen, source_version=state.version)
