from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ArrayLike = np.ndarray


@dataclass(frozen=True)
class ModalityInput:
    modality: str
    source_id: str
    payload: Any
    timestamp: float | None = None
    metadata: dict[str, Any] = None


@dataclass(frozen=True)
class DecodedModality:
    modality: str
    source_id: str
    tensor: ArrayLike
    mask: ArrayLike | None
    shape: tuple[int, ...]
    dtype: str
    timestamp: float | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ModalityRepresentation:
    modality: str
    source_id: str
    latent: ArrayLike
    confidence: float
    mask: ArrayLike | None
    timestamp: float | None
    encoder_name: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ModalityContradiction:
    contradiction_id: str
    modalities: tuple[str, ...]
    contradiction_type: str
    severity: float
    confidence: float
    evidence: dict[str, Any]
    resolution: str | None


@dataclass(frozen=True)
class FusionResult:
    fused_latent: ArrayLike
    modality_weights: dict[str, float]
    modality_confidences: dict[str, float]
    missing_modalities: tuple[str, ...]
    contradictions: tuple[ModalityContradiction, ...]
    provenance: tuple[str, ...]
    diagnostics: dict[str, Any]
