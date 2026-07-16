from __future__ import annotations

from .decoders import AudioDecoder, ImageDecoder, StructuredDecoder
from .encoders import AudioEncoder, StructuredEncoder, VisionEncoder
from .fusion import ConfidenceFusion
from .types import FusionResult, ModalityInput, ModalityRepresentation


class MultimodalPipeline:
    def __init__(self, latent_dim: int = 32) -> None:
        self.decoders = {"vision": ImageDecoder(), "audio": AudioDecoder(), "structured": StructuredDecoder()}
        self.encoders = {"vision": VisionEncoder(latent_dim), "audio": AudioEncoder(latent_dim),
                         "structured": StructuredEncoder(latent_dim)}
        self.fusion = ConfidenceFusion()

    def encode(self, value: ModalityInput) -> ModalityRepresentation:
        if value.modality not in self.decoders:
            raise ValueError(f"Unsupported modality: {value.modality}")
        return self.encoders[value.modality].encode(self.decoders[value.modality].decode(value))

    def process(self, values: list[ModalityInput]) -> tuple[list[ModalityRepresentation], FusionResult]:
        seen = set()
        for value in values:
            if value.modality in seen:
                raise ValueError(f"Duplicate modality: {value.modality}")
            seen.add(value.modality)
        representations = [self.encode(value) for value in values]
        return representations, self.fusion.fuse(representations)
