from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ModalityInput:
    modality: str
    source_id: str
    payload: Any
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecodedModality:
    modality: str
    source_id: str
    tensor: np.ndarray
    mask: np.ndarray | None
    shape: tuple[int, ...]
    dtype: str
    timestamp: float | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ModalityRepresentation:
    modality: str
    source_id: str
    latent: np.ndarray
    confidence: float
    mask: np.ndarray | None
    timestamp: float | None
    encoder_name: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FusionResult:
    fused_latent: np.ndarray
    modality_weights: dict[str, float]
    modality_confidences: dict[str, float]
    missing_modalities: tuple[str, ...]
    contradictions: tuple[dict[str, Any], ...]
    provenance: tuple[str, ...]
    diagnostics: dict[str, Any]
