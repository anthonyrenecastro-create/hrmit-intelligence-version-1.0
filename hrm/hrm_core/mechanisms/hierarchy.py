from __future__ import annotations

import numpy as np

from ..proposals import BlockDelta, MechanismProposal
from .base import HRMInput, HRMMechanism, TransitionContext


class HierarchyMechanism(HRMMechanism):
    mechanism_id = "hierarchy"
    read_blocks = frozenset({"Phi", "H", "B"})
    write_blocks = frozenset({"Phi", "H"})

    def __init__(self, gain: float = 0.1, enabled: bool = True) -> None:
        self.gain = float(gain)
        self.enabled = enabled

    def propose(self, snapshot, external_input: HRMInput, observables, context: TransitionContext) -> MechanismProposal:
        phi = snapshot.state.phi.phi
        restriction = snapshot.state.hierarchy.restriction
        prolongation = snapshot.state.hierarchy.prolongation

        coarse = restriction @ phi
        lifted = prolongation @ coarse
        if not self.enabled:
            phi_delta = np.zeros_like(phi)
            coarse_delta = np.zeros_like(coarse)
            activation = 0.0
        else:
            phi_delta = context.dt * self.gain * (lifted - phi)
            coarse_delta = coarse - snapshot.state.hierarchy.coarse
            activation = float(np.linalg.norm(phi_delta))

        return MechanismProposal(
            mechanism_id=self.mechanism_id,
            source_state_version=snapshot.source_version,
            read_blocks=self.read_blocks,
            write_blocks=self.write_blocks,
            activation=activation,
            delta=BlockDelta(phi=phi_delta, hierarchy_coarse=coarse_delta),
            dtype=str(phi.dtype),
            device=snapshot.state.device,
            estimated_cost=float(phi.shape[0] * phi.shape[1]),
            diagnostics={"enabled": self.enabled},
            provenance={"owner": self.mechanism_id},
        )
