from __future__ import annotations

import numpy as np

from ..proposals import BlockDelta, MechanismProposal
from .base import HRMInput, HRMMechanism, TransitionContext


class GeometryMechanism(HRMMechanism):
    mechanism_id = "geometry"
    read_blocks = frozenset({"G", "Phi"})
    write_blocks = frozenset({"G"})

    def propose(self, snapshot, external_input: HRMInput, observables, context: TransitionContext) -> MechanismProposal:
        phi = snapshot.state.phi.phi
        return MechanismProposal(
            mechanism_id=self.mechanism_id,
            source_state_version=snapshot.source_version,
            read_blocks=self.read_blocks,
            write_blocks=self.write_blocks,
            activation=0.0,
            delta=BlockDelta(phi=np.zeros_like(phi)),
            dtype=str(phi.dtype),
            device=snapshot.state.device,
            estimated_cost=0.0,
            diagnostics={"note": "fixed geometry phase"},
            provenance={"owner": self.mechanism_id},
        )
