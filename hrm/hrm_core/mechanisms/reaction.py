from __future__ import annotations

import numpy as np

from ..proposals import BlockDelta, MechanismProposal
from .base import HRMInput, HRMMechanism, TransitionContext


class ReactionMechanism(HRMMechanism):
    mechanism_id = "reaction"
    read_blocks = frozenset({"Phi", "B"})
    write_blocks = frozenset({"Phi"})

    def __init__(self, gain: float = 0.5, saturation: float = 0.35, enabled: bool = True) -> None:
        self.gain = float(gain)
        self.saturation = float(saturation)
        self.enabled = enabled

    def propose(self, snapshot, external_input: HRMInput, observables, context: TransitionContext) -> MechanismProposal:
        phi = snapshot.state.phi.phi
        if not self.enabled:
            delta = np.zeros_like(phi)
            activation = 0.0
        else:
            reaction = self.gain * phi - self.saturation * np.power(phi, 3)
            delta = context.dt * reaction
            activation = float(np.linalg.norm(delta))
        return MechanismProposal(
            mechanism_id=self.mechanism_id,
            source_state_version=snapshot.source_version,
            read_blocks=self.read_blocks,
            write_blocks=self.write_blocks,
            activation=activation,
            delta=BlockDelta(phi=delta),
            dtype=str(phi.dtype),
            device=snapshot.state.device,
            estimated_cost=float(phi.size),
            diagnostics={"enabled": self.enabled},
            provenance={"owner": self.mechanism_id},
        )
