from __future__ import annotations

import numpy as np

from hrm.memory import _embed_text

from ..proposals import BlockDelta, MechanismProposal
from .base import HRMInput, HRMMechanism, TransitionContext


class MemoryMechanism(HRMMechanism):
    mechanism_id = "memory"
    read_blocks = frozenset({"Phi", "M", "B"})
    write_blocks = frozenset({"Phi", "M"})

    def __init__(self, gain: float = 0.15, decay: float = 0.08, enabled: bool = True) -> None:
        self.gain = float(gain)
        self.decay = float(decay)
        self.enabled = enabled

    def propose(self, snapshot, external_input: HRMInput, observables, context: TransitionContext) -> MechanismProposal:
        phi = snapshot.state.phi.phi
        memory_working = snapshot.state.memory.working
        focus_row = snapshot.state.memory.write_index % max(1, memory_working.shape[0])
        focused_memory = memory_working[focus_row]
        metadata = external_input.metadata or {}
        query_text = str(metadata.get("memory_query", ""))
        write_candidate = str(metadata.get("memory_write_candidate", ""))
        response_draft = str(metadata.get("response_draft", ""))
        memory_signal = _embed_text(" ".join(item for item in (query_text, write_candidate, response_draft) if item), dim=phi.shape[1]).astype(phi.dtype)

        if not self.enabled:
            phi_delta = np.zeros_like(phi)
            memory_next = memory_working.copy()
            activation = 0.0
        else:
            # Coupling with explicit bounded memory write dynamics.
            phi_delta = context.dt * self.gain * ((memory_working - phi) + 0.05 * memory_signal[None, :] + 0.02 * focused_memory[None, :])
            memory_next = memory_working.copy()
            memory_next[focus_row] = (1.0 - self.decay) * memory_working[focus_row] + self.decay * phi[focus_row]
            if query_text or write_candidate or response_draft:
                memory_next[focus_row] = memory_next[focus_row] + 0.01 * memory_signal
            activation = float(np.linalg.norm(phi_delta))

        memory_delta = memory_next - memory_working
        return MechanismProposal(
            mechanism_id=self.mechanism_id,
            source_state_version=snapshot.source_version,
            read_blocks=self.read_blocks,
            write_blocks=self.write_blocks,
            activation=activation,
            delta=BlockDelta(phi=phi_delta, memory=memory_delta),
            dtype=str(phi.dtype),
            device=snapshot.state.device,
            estimated_cost=float(phi.size),
            diagnostics={"enabled": self.enabled, "focus_row": focus_row},
            provenance={"owner": self.mechanism_id},
        )
