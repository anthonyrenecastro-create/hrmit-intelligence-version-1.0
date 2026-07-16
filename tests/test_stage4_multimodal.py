from __future__ import annotations

import io
import wave

import numpy as np
from PIL import Image

from hrm.multimodal import HRMStateProjector, ModalityInput, MultimodalPipeline
from hrm.multimodal.benchmarks import complementary_signal_benchmark
from hrm.multimodal.fusion import ConfidenceFusion
from hrm.multimodal.types import ModalityRepresentation
from hrm.perception import PerceptionPipeline


def image_bytes(array: np.ndarray, fmt: str = "PNG") -> bytes:
    stream = io.BytesIO()
    Image.fromarray(array.astype(np.uint8)).save(stream, format=fmt)
    return stream.getvalue()


def wav_bytes(samples: np.ndarray, rate: int = 8000) -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(rate)
        wav.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return stream.getvalue()


def inputs() -> list[ModalityInput]:
    image = np.zeros((32, 32, 3), np.uint8); image[8:24, 8:24] = (255, 80, 20)
    audio = np.sin(np.linspace(0, 20 * np.pi, 800, endpoint=False)).astype(np.float32) * .6
    schema = {"temperature": {"type": "number"}, "alarm": {"type": "boolean"},
              "zone": {"type": "category", "choices": ["north", "south"]}}
    return [
        ModalityInput("vision", "camera-1", image_bytes(image), 10.0),
        ModalityInput("audio", "microphone-1", wav_bytes(audio), 10.1),
        ModalityInput("structured", "sensor-1", {"temperature": 22.5, "alarm": False, "zone": "north"},
                      10.0, {"schema": schema}),
    ]


def test_real_inputs_produce_distinct_provenance_preserving_representations() -> None:
    reps, fusion = MultimodalPipeline().process(inputs())
    assert {r.modality for r in reps} == {"vision", "audio", "structured"}
    assert all(r.latent.shape == (32,) and 0 <= r.confidence <= 1 for r in reps)
    assert not np.allclose(reps[0].latent, reps[1].latent)
    assert fusion.provenance == ("camera-1", "microphone-1", "sensor-1")
    assert abs(sum(fusion.modality_weights.values()) - 1.0) < 1e-6
    assert fusion.missing_modalities == ()


def test_grayscale_jpeg_is_normalized_to_rgb() -> None:
    gray = np.arange(1024, dtype=np.uint8).reshape(32, 32)
    rep = MultimodalPipeline().encode(ModalityInput("vision", "gray", image_bytes(gray, "JPEG")))
    assert rep.metadata["original_mode"] == "L"
    assert rep.metadata["format"] == "JPEG"


def test_missing_modality_is_reported_and_pipeline_remains_functional() -> None:
    _, fusion = MultimodalPipeline().process(inputs()[:2])
    assert fusion.missing_modalities == ("structured",)
    assert np.isfinite(fusion.fused_latent).all()


def test_structured_schema_order_and_missing_mask_are_preserved() -> None:
    value = inputs()[2]
    payload = {"temperature": 22.5, "alarm": None, "zone": "south"}
    rep = MultimodalPipeline().encode(ModalityInput("structured", value.source_id, payload, metadata=value.metadata))
    assert rep.metadata["field_order"] == ("temperature", "alarm", "zone")
    assert rep.mask.tolist() == [True, False, True]
    assert rep.confidence == 2 / 3


def test_hrm_projection_is_bounded_and_changes_state() -> None:
    _, fusion = MultimodalPipeline().process(inputs())
    result = HRMStateProjector(16, max_delta_norm=.25).project(fusion, np.zeros(16, np.float32))
    assert 0 < result.delta_norm <= .25
    assert result.state.shape == (16,)
    assert result.provenance == fusion.provenance


def test_audio_noise_reduces_confidence_when_clipped() -> None:
    clean = inputs()[1]
    clipped = np.ones(800, np.float32)
    pipeline = MultimodalPipeline()
    clean_rep = pipeline.encode(clean)
    clipped_rep = pipeline.encode(ModalityInput("audio", "clipped", wav_bytes(clipped)))
    assert clean_rep.confidence > clipped_rep.confidence


def test_contradictory_representations_are_preserved_in_diagnostics() -> None:
    left = ModalityRepresentation("vision", "v", np.ones(32, np.float32), .8, None, None, "test", {})
    right = ModalityRepresentation("structured", "s", -np.ones(32, np.float32), .9, None, None, "test", {})
    result = ConfidenceFusion().fuse([left, right])
    assert len(result.contradictions) == 1
    assert result.contradictions[0]["left"] == "vision"
    assert result.contradictions[0]["cosine_similarity"] < -.99


def test_multimodal_fusion_improves_complementary_signal_task() -> None:
    report = complementary_signal_benchmark()
    assert report["fusion_accuracy"] >= .9
    assert report["fusion_improvement"] >= .25


def test_stage4_compatibility_facade_returns_inspectable_hrm_projection() -> None:
    modalities, metadata = PerceptionPipeline.sample_inputs()
    integrated = PerceptionPipeline().integrate(modalities, metadata=metadata)
    projection = PerceptionPipeline.project_into_hrm(integrated, np.zeros(16, np.float32), .2)
    assert set(integrated["modalities"]) == {"vision", "audio", "structured"}
    assert 0 < projection["delta_norm"] <= .2
    assert tuple(projection["provenance"]) == integrated["provenance"]
