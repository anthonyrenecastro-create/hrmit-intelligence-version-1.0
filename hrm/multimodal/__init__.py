"""Functional multimodal processing for HRMIT Stage 4."""

from .fusion import ConfidenceFusion
from .pipeline import MultimodalPipeline
from .projection import HRMStateProjector
from .types import DecodedModality, FusionResult, ModalityInput, ModalityRepresentation

__all__ = [
    "ConfidenceFusion",
    "DecodedModality",
    "FusionResult",
    "HRMStateProjector",
    "ModalityInput",
    "ModalityRepresentation",
    "MultimodalPipeline",
]
