from __future__ import annotations

import io
import json
import math
import wave
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from hrm.multimodal import FusionEngine, HRMProjector, ModalityRegistry
from hrm.multimodal.audio import AudioDecoder, AudioEncoder, AudioFeatures, AudioPreprocessor
from hrm.multimodal.structured import FieldSchema, StructuredDecoder, StructuredEncoder, StructuredSchema
from hrm.multimodal.text import TextDecoder, TextEncoder
from hrm.multimodal.types import DecodedModality, ModalityInput, ModalityRepresentation
from hrm.multimodal.video.experimental import ExperimentalVideoAdapter
from hrm.multimodal.vision import ImageDecoder, VisionEncoder, VisionPreprocessor


@dataclass(frozen=True)
class MultimodalSample:
    text: str
    image: bytes
    audio: bytes
    structured: str
    video: list[bytes]


class MultimodalPipeline:
    def __init__(self) -> None:
        self.registry = ModalityRegistry()
        self.vision_decoder = ImageDecoder()
        self.vision_preprocessor = VisionPreprocessor()
        self.vision_encoder = VisionEncoder()
        self.audio_decoder = AudioDecoder()
        self.audio_preprocessor = AudioPreprocessor()
        self.audio_features = AudioFeatures()
        self.audio_encoder = AudioEncoder()
        self.text_decoder = TextDecoder()
        self.text_encoder = TextEncoder()
        self.video_adapter = ExperimentalVideoAdapter()
        schema = StructuredSchema(
            schema_id="default_tabular_schema",
            version="1.0",
            fields=(
                FieldSchema(name="value", dtype="numeric", required=True, nullable=False, numeric_range=(0.0, 10.0)),
                FieldSchema(name="category", dtype="categorical", required=True, nullable=False, categorical_values=("A", "B", "C")),
                FieldSchema(name="timestamp", dtype="timestamp", required=False, nullable=True),
            ),
        )
        self.structured_decoder = StructuredDecoder(schema=schema)
        self.structured_encoder = StructuredEncoder()
        self.fusion = FusionEngine(strategy="learned_gating", modality_order=["image", "audio", "structured", "text", "video"])
        self.projector = HRMProjector(target_dim=64)

    def decode(self, input_data: ModalityInput) -> DecodedModality:
        modality = input_data.modality
        if modality in {"image", "vision"}:
            decoded = self.vision_decoder.decode(input_data.payload, input_data.source_id, input_data.timestamp)
            return DecodedModality(
                modality="image",
                source_id=decoded.source_id,
                tensor=decoded.tensor,
                mask=decoded.mask,
                shape=decoded.shape,
                dtype=decoded.dtype,
                timestamp=decoded.timestamp,
                metadata=decoded.metadata,
            )
        if modality == "audio":
            return self.audio_decoder.decode(input_data.payload, input_data.source_id, input_data.timestamp)
        if modality == "structured":
            return self.structured_decoder.decode(input_data.payload, input_data.source_id, input_data.timestamp)
        if modality == "text":
            return self.text_decoder.decode(input_data.payload, input_data.source_id, input_data.timestamp)
        if modality == "video":
            return self.video_adapter.decode(input_data.payload, input_data.source_id, input_data.timestamp)
        raise ValueError(f"Unsupported modality: {input_data.modality}")

    def preprocess(self, decoded: DecodedModality) -> DecodedModality:
        if decoded.modality == "image":
            return self.vision_preprocessor.preprocess(decoded)
        if decoded.modality == "audio":
            return self.audio_preprocessor.preprocess(decoded)
        if decoded.modality in {"structured", "text", "video"}:
            return decoded
        raise ValueError(f"Unsupported modality: {decoded.modality}")

    def represent(self, decoded: DecodedModality) -> ModalityRepresentation:
        if decoded.modality == "image":
            return self.vision_encoder.encode(decoded)
        if decoded.modality == "audio":
            processed = self.audio_features.log_mel_spectrogram(decoded)
            return self.audio_encoder.encode(processed)
        if decoded.modality == "structured":
            return self.structured_encoder.encode(decoded)
        if decoded.modality == "text":
            return self.text_encoder.encode(decoded)
        if decoded.modality == "video":
            return self.video_adapter.encode(decoded)
        raise ValueError(f"Unsupported modality: {decoded.modality}")

    def project(self, representation: ModalityRepresentation) -> dict[str, object]:
        return self.projector.project(representation)

    def fuse(self, representations: list[ModalityRepresentation], expected_modalities: list[str] | None = None) -> Any:
        return self.fusion.fuse(representations, expected_modalities=expected_modalities)

    def process(self, input_data: ModalityInput) -> ModalityRepresentation:
        decoded = self.decode(input_data)
        preprocessed = self.preprocess(decoded)
        return self.represent(preprocessed)

    def explain(self, representations: list[ModalityRepresentation]) -> dict[str, Any]:
        fused = self.fuse(representations)
        return {
            "fusion": fused,
            "projected": [self.project(rep) for rep in representations],
        }

    @staticmethod
    def _make_image_bytes(width: int = 64, height: int = 64, color_value: int = 128, format: str = "PNG") -> bytes:
        image = Image.new("RGB", (width, height), color=(color_value, 255 - color_value, color_value))
        with io.BytesIO() as output:
            image.save(output, format=format)
            return output.getvalue()

    @staticmethod
    def _make_wav_bytes(sample_rate: int = 16000, duration: float = 1.0, frequency: float = 440.0) -> bytes:
        count = int(sample_rate * duration)
        t = np.linspace(0.0, duration, count, endpoint=False)
        waveform = 0.5 * np.sin(2 * np.pi * frequency * t)
        waveform_int16 = np.int16(np.clip(waveform * 32767, -32768, 32767))
        with io.BytesIO() as output:
            with wave.open(output, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(waveform_int16.tobytes())
            return output.getvalue()

    @classmethod
    def sample_inputs(cls) -> MultimodalSample:
        text = "HRM receives structured conditions, visual evidence, and audio cues."
        image = cls._make_image_bytes(width=64, height=64, color_value=96, format="PNG")
        audio = cls._make_wav_bytes(sample_rate=16000, duration=1.0, frequency=440.0)
        structured = json.dumps([
            {"value": 4.2, "category": "B", "timestamp": "2026-07-16T12:00:00"},
            {"value": 8.4, "category": "A", "timestamp": "2026-07-16T12:05:00"},
        ])
        video_frames = [
            cls._make_image_bytes(width=32, height=32, color_value=10, format="PNG"),
            cls._make_image_bytes(width=32, height=32, color_value=200, format="PNG"),
            cls._make_image_bytes(width=32, height=32, color_value=120, format="PNG"),
        ]
        return MultimodalSample(text=text, image=image, audio=audio, structured=structured, video=video_frames)
