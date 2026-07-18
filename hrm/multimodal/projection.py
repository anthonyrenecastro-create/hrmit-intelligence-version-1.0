from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hrm.multimodal.types import FusionResult
from hrm.multimodal.types import ModalityRepresentation


class HRMProjector:
    def __init__(self, target_dim: int = 64, target_component: str = "observation") -> None:
        self.target_dim = target_dim
        self.target_component = target_component

    def project(self, representation: ModalityRepresentation, target_component: str | None = None) -> dict[str, object]:
        latent = np.asarray(representation.latent, dtype=np.float32).ravel()
        projected = latent[: self.target_dim]
        if projected.size < self.target_dim:
            projected = np.pad(projected, (0, self.target_dim - projected.size), mode="constant")
        norm = np.linalg.norm(projected)
        normalized = projected / (norm + 1e-9)
        return {
            "modality": representation.modality,
            "source_id": representation.source_id,
            "projected": normalized,
            "projected_shape": normalized.shape,
            "original_shape": tuple(representation.latent.shape),
            "confidence": representation.confidence,
            "mask": representation.mask,
            "encoder_name": representation.encoder_name,
            "target_component": target_component or self.target_component,
            "projection_parameters": {
                "target_dim": self.target_dim,
                "normalization": "l2",
                "component": target_component or self.target_component,
            },
            "metadata": representation.metadata,
        }


@dataclass(frozen=True)
class StateProjectionResult:
    state: np.ndarray
    delta_norm: float
    input_norm: float
    gate: float
    provenance: tuple[str, ...]


class HRMStateProjector:
    def __init__(self, state_dim: int, max_delta_norm: float = 1.0) -> None:
        self.state_dim = state_dim
        self.max_delta_norm = float(max_delta_norm)

    def project(self, fusion: FusionResult, state: np.ndarray) -> StateProjectionResult:
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        latent = np.asarray(fusion.fused_latent, dtype=np.float32).reshape(-1)
        if latent.size < self.state_dim:
            latent = np.pad(latent, (0, self.state_dim - latent.size), mode="constant")
        projected = latent[: self.state_dim]
        delta = projected - state_array[: self.state_dim]
        delta_norm = float(np.linalg.norm(delta))
        gate = 1.0
        if delta_norm > self.max_delta_norm > 0.0:
            gate = self.max_delta_norm / (delta_norm + 1e-9)
            delta = delta * gate
            delta_norm = float(np.linalg.norm(delta))
        updated = state_array.copy()
        updated[: self.state_dim] = state_array[: self.state_dim] + delta
        return StateProjectionResult(
            state=updated,
            delta_norm=delta_norm,
            input_norm=float(np.linalg.norm(projected)),
            gate=float(gate),
            provenance=fusion.provenance,
        )
