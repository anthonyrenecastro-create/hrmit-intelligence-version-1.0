from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import FusionResult


@dataclass(frozen=True)
class ProjectionResult:
    state: np.ndarray
    delta_norm: float
    input_norm: float
    gate: float
    provenance: tuple[str, ...]


class HRMStateProjector:
    """Bounded residual projection into a one-dimensional HRM cognitive state."""
    def __init__(self, state_dim: int, max_delta_norm: float = 1.0, seed: int = 71) -> None:
        self.state_dim, self.max_delta_norm, self.seed = state_dim, max_delta_norm, seed

    def project(self, fusion: FusionResult, state: np.ndarray) -> ProjectionResult:
        state = np.asarray(state, np.float32)
        if state.shape != (self.state_dim,):
            raise ValueError(f"Expected HRM state shape {(self.state_dim,)}, got {state.shape}")
        rng = np.random.default_rng(self.seed + fusion.fused_latent.size)
        matrix = rng.normal(0, 1 / np.sqrt(fusion.fused_latent.size),
                            (fusion.fused_latent.size, self.state_dim)).astype(np.float32)
        raw = np.tanh(fusion.fused_latent @ matrix)
        norm = float(np.linalg.norm(raw))
        delta = raw * min(1.0, self.max_delta_norm / (norm + 1e-9))
        gate = float(np.mean(list(fusion.modality_confidences.values())))
        delta *= gate
        return ProjectionResult(state + delta, float(np.linalg.norm(delta)),
                                float(np.linalg.norm(fusion.fused_latent)), gate, fusion.provenance)
