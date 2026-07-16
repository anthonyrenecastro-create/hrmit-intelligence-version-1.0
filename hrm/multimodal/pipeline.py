from __future__ import annotations

from typing import Any

from hrm.multimodal import FusionEngine, HRMProjector, ModalityRegistry
from hrm.multimodal.audio import AudioDecoder, AudioEncoder, AudioFeatures, AudioPreprocessor
from hrm.multimodal.structured import StructuredDecoder, StructuredEncoder, StructuredSchema, FieldSchema
from hrm.multimodal.types import ModalityInput
from hrm.multimodal.vision import ImageDecoder, VisionEncoder, VisionPreprocessor


class MultimodalPipeline:
    def __init__(self) -> None:
        self.registry = ModalityRegistry()
        self.registry.register("vision", self)
        self.registry.register("audio", self)
        self.registry.register("structured", self)
        self.vision_decoder = ImageDecoder()
        self.vision_preprocessor = VisionPreprocessor()
        self.vision_encoder = VisionEncoder()
        self.audio_decoder = AudioDecoder()
        self.audio_preprocessor = AudioPreprocessor()
        self.audio_features = AudioFeatures()
        self.audio_encoder = AudioEncoder()
        schema = StructuredSchema(
            schema_id="default_tabular_schema",
            version="1.0",
            fields=(
                FieldSchema(name="value", dtype="numeric", required=True, nullable=False, numeric_range=(0.0, 10.0)),
                FieldSchema(name="category", dtype="categorical", required=True, nullable=False, categorical_values=("A", "B", "C")),
            ),
        )
        self.structured_decoder = StructuredDecoder(schema=schema)
        self.structured_encoder = StructuredEncoder()
        self.fusion = FusionEngine()
        self.projector = HRMProjector()

    def decode(self, input_data: ModalityInput) -> Any:
        if input_data.modality == "vision":
            return self.vision_decoder.decode(input_data.payload, input_data.source_id, input_data.timestamp)
        if input_data.modality == "audio":
            return self.audio_decoder.decode(input_data.payload, input_data.source_id, input_data.timestamp)
        if input_data.modality == "structured":
            return self.structured_decoder.decode(input_data.payload, input_data.source_id, input_data.timestamp)
        raise ValueError(f"Unsupported modality: {input_data.modality}")

    def preprocess(self, decoded: Any) -> Any:
        if decoded.modality == "vision":
            return self.vision_preprocessor.preprocess(decoded)
        if decoded.modality == "audio":
            return self.audio_preprocessor.preprocess(decoded)
        if decoded.modality == "structured":
            return decoded
        raise ValueError(f"Unsupported modality: {decoded.modality}")

    def represent(self, decoded: Any) -> Any:
        if decoded.modality == "vision":
            return self.vision_encoder.encode(decoded)
        if decoded.modality == "audio":
            processed = self.audio_features.log_mel_spectrogram(decoded)
            return self.audio_encoder.encode(processed)
        if decoded.modality == "structured":
            return self.structured_encoder.encode(decoded)
        raise ValueError(f"Unsupported modality: {decoded.modality}")

    def project(self, representation: Any) -> dict[str, object]:
        return self.projector.project(representation)

    def fuse(self, representations: list[Any]) -> Any:
        return self.fusion.fuse(representations)

    def process(self, input_data: ModalityInput) -> Any:
        decoded = self.decode(input_data)
        preprocessed = self.preprocess(decoded)
        return self.represent(preprocessed)

    def explain(self, representations: list[Any]) -> dict[str, Any]:
        fused = self.fuse(representations)
        return {
            "fusion": fused,
            "projected": [self.project(rep) for rep in representations],
        }
