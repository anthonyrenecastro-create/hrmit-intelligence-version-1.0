from __future__ import annotations

from hrm.multimodal.types import ModalityRepresentation


class StructuredTasks:
    @staticmethod
    def predict_category(representation: ModalityRepresentation) -> str:
        if representation.modality != "structured":
            raise ValueError("Structured task requires a structured representation")
        score = float(sum(representation.latent[:3]))
        return "positive" if score > 1.0 else "negative"

    @staticmethod
    def validate_record_order(representation: ModalityRepresentation) -> bool:
        if representation.modality != "structured":
            raise ValueError("Structured task requires a structured representation")
        return representation.metadata.get("record_count", 1) > 0
