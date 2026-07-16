"""Stage 4 compatibility facade.

New code should import :mod:`hrm.multimodal`. This module preserves the former
``PerceptionPipeline`` entry point without returning placeholder embeddings.
"""
from __future__ import annotations

import io
import wave
from typing import Any

import numpy as np
from PIL import Image

from hrm.multimodal import HRMStateProjector, ModalityInput, MultimodalPipeline


class PerceptionPipeline:
    aliases = {"image": "vision"}

    def __init__(self, latent_dim: int = 32) -> None:
        self.pipeline = MultimodalPipeline(latent_dim)

    def integrate(self, modalities: dict[str, Any], *, metadata: dict[str, dict[str, Any]] | None = None,
                  timestamps: dict[str, float] | None = None) -> dict[str, Any]:
        metadata, timestamps = metadata or {}, timestamps or {}
        values = []
        for name, payload in modalities.items():
            canonical = self.aliases.get(name, name)
            values.append(ModalityInput(canonical, metadata.get(name, {}).get("source_id", name), payload,
                                        timestamps.get(name), metadata.get(name, {})))
        representations, fusion = self.pipeline.process(values)
        return {
            "representations": [{"modality": item.modality, "source_id": item.source_id,
                                  "latent": item.latent.tolist(), "confidence": item.confidence,
                                  "timestamp": item.timestamp, "encoder_name": item.encoder_name,
                                  "metadata": item.metadata} for item in representations],
            "combined_embedding": fusion.fused_latent.tolist(),
            "modality_weights": fusion.modality_weights,
            "modality_confidences": fusion.modality_confidences,
            "missing_modalities": fusion.missing_modalities,
            "contradictions": fusion.contradictions,
            "provenance": fusion.provenance,
            "diagnostics": fusion.diagnostics,
            "modalities": [item.modality for item in representations],
        }

    @staticmethod
    def sample_inputs() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        image = np.zeros((32, 32, 3), np.uint8); image[8:24, 8:24] = (240, 80, 20)
        image_stream = io.BytesIO(); Image.fromarray(image).save(image_stream, "PNG")
        t = np.arange(800) / 8000; samples = .6 * np.sin(2 * np.pi * 440 * t)
        audio_stream = io.BytesIO()
        with wave.open(audio_stream, "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(8000)
            wav.writeframes((samples * 32767).astype("<i2").tobytes())
        schema = {"temperature": {"type": "number"}, "alarm": {"type": "boolean"}}
        return ({"vision": image_stream.getvalue(), "audio": audio_stream.getvalue(),
                 "structured": {"temperature": 22.0, "alarm": False}},
                {"vision": {"source_id": "sample-image"}, "audio": {"source_id": "sample-audio"},
                 "structured": {"source_id": "sample-sensor", "schema": schema}})

    @staticmethod
    def project_into_hrm(integrated: dict[str, Any], state: np.ndarray, max_delta_norm: float = 1.0) -> dict[str, Any]:
        from hrm.multimodal.types import FusionResult
        fusion = FusionResult(np.asarray(integrated["combined_embedding"], np.float32),
                              integrated["modality_weights"], integrated["modality_confidences"],
                              tuple(integrated["missing_modalities"]), tuple(integrated["contradictions"]),
                              tuple(integrated["provenance"]), integrated["diagnostics"])
        result = HRMStateProjector(state.size, max_delta_norm).project(fusion, state)
        return {"state": result.state.tolist(), "delta_norm": result.delta_norm,
                "input_norm": result.input_norm, "gate": result.gate, "provenance": result.provenance}
