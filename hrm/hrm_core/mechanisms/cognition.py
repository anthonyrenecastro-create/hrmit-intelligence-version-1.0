from __future__ import annotations

import numpy as np

from hrm.memory import _embed_text

from ..proposals import BlockDelta, MechanismProposal
from .base import HRMInput, HRMMechanism, TransitionContext


class CognitionMechanism(HRMMechanism):
    mechanism_id = "cognition"
    read_blocks = frozenset({"Phi", "C", "M", "H", "B"})
    write_blocks = frozenset({"Phi", "C"})

    def __init__(self, gain: float = 0.12, guidance_gain: float = 0.08, enabled: bool = True) -> None:
        self.gain = float(gain)
        self.guidance_gain = float(guidance_gain)
        self.enabled = enabled

    def propose(self, snapshot, external_input: HRMInput, observables, context: TransitionContext) -> MechanismProposal:
        phi = snapshot.state.phi.phi
        latent = snapshot.state.cognition.latent
        memory = snapshot.state.memory.working
        focus_row = snapshot.state.memory.write_index % max(1, memory.shape[0])
        focused_memory = memory[focus_row]
        metadata = external_input.metadata or {}
        channel_mean = phi.mean(axis=0)
        memory_mean = memory.mean(axis=0)

        executive_text = " ".join(
            item
            for item in (
                str(metadata.get("intent", "")),
                " ".join(str(h) for h in metadata.get("hypotheses", ())),
                str(metadata.get("response_draft", "")),
                str(metadata.get("memory_write_candidate", "")),
            )
            if item
        )
        executive_signal = np.zeros_like(channel_mean)
        if executive_text:
            executive_signal = _embed_text(executive_text, dim=channel_mean.shape[0]).astype(phi.dtype)

        projection = np.zeros_like(channel_mean)
        width = min(channel_mean.shape[0], latent.shape[0])
        projection[:width] = latent[:width]

        if not self.enabled:
            phi_delta = np.zeros_like(phi)
            latent_next = latent.copy()
            uncertainty = snapshot.state.cognition.uncertainty.copy()
            activation = 0.0
        else:
            guidance = channel_mean - projection + 0.05 * executive_signal + 0.02 * focused_memory
            coupled = 0.45 * channel_mean + 0.35 * memory_mean + 0.15 * focused_memory + 0.05 * executive_signal
            latent_next = latent.copy()
            latent_next[:width] = latent[:width] + context.dt * self.gain * coupled[:width]
            phi_delta = context.dt * self.guidance_gain * guidance[None, :]
            uncertainty = np.clip(snapshot.state.cognition.uncertainty * 0.99 + np.abs(guidance) * 0.01, 0.0, 10.0)
            activation = float(np.linalg.norm(phi_delta))

        residual = channel_mean - projection
        diagnostics = {
            "enabled": self.enabled,
            "residual_norm": float(np.linalg.norm(residual)),
            "uncertainty_mean": float(np.mean(uncertainty)),
            "executive_signal_norm": float(np.linalg.norm(executive_signal)),
            "focus_row": focus_row,
        }

        return MechanismProposal(
            mechanism_id=self.mechanism_id,
            source_state_version=snapshot.source_version,
            read_blocks=self.read_blocks,
            write_blocks=self.write_blocks,
            activation=activation,
            delta=BlockDelta(phi=phi_delta, cognition_latent=latent_next - latent),
            dtype=str(phi.dtype),
            device=snapshot.state.device,
            estimated_cost=float(phi.shape[0] * phi.shape[1]),
            uncertainty=float(np.mean(uncertainty)),
            diagnostics=diagnostics,
            provenance={"owner": self.mechanism_id},
        )
