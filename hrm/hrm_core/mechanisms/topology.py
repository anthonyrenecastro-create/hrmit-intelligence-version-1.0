from __future__ import annotations

import numpy as np

from ..proposals import BlockDelta, MechanismProposal
from .base import HRMInput, HRMMechanism, TransitionContext


class TopologyMechanism(HRMMechanism):
    mechanism_id = "topology"
    read_blocks = frozenset({"T", "G"})
    write_blocks = frozenset({"T"})

    def propose(self, snapshot, external_input: HRMInput, observables, context: TransitionContext) -> MechanismProposal:
        phi = snapshot.state.phi.phi
        metadata = external_input.metadata or {}
        requested = bool(
            metadata.get("topology_events")
            or metadata.get("topology_add_nodes")
            or metadata.get("topology_remove_nodes")
            or metadata.get("topology_rewire")
        )
        activation = 1.0 if requested else 0.0
        return MechanismProposal(
            mechanism_id=self.mechanism_id,
            source_state_version=snapshot.source_version,
            read_blocks=self.read_blocks,
            write_blocks=self.write_blocks,
            activation=activation,
            delta=BlockDelta(phi=np.zeros_like(phi)),
            dtype=str(phi.dtype),
            device=snapshot.state.device,
            estimated_cost=0.05 * activation,
            diagnostics={"note": "topology events requested via metadata" if requested else "no topology request"},
            provenance={"owner": self.mechanism_id},
        )
