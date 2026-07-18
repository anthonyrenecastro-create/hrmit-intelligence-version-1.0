from __future__ import annotations

import numpy as np

from ..proposals import BlockDelta, MechanismProposal
from .base import HRMInput, HRMMechanism, TransitionContext


class DiffusionMechanism(HRMMechanism):
    mechanism_id = "diffusion"
    read_blocks = frozenset({"Phi", "G", "T", "B"})
    write_blocks = frozenset({"Phi"})

    def __init__(self, gain: float = 0.2, enabled: bool = True) -> None:
        self.gain = float(gain)
        self.enabled = enabled

    def propose(self, snapshot, external_input: HRMInput, observables, context: TransitionContext) -> MechanismProposal:
        phi = snapshot.state.phi.phi
        laplacian = snapshot.state.geometry.laplacian
        if not self.enabled:
            delta = np.zeros_like(phi)
            activation = 0.0
        else:
            # Positive-semidefinite Laplacian with sign-correct smoothing term.
            delta = -context.dt * self.gain * (laplacian @ phi)
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
            estimated_cost=float(phi.shape[0] * phi.shape[0] * phi.shape[1]),
            diagnostics={"enabled": self.enabled},
            provenance={"owner": self.mechanism_id},
        )
