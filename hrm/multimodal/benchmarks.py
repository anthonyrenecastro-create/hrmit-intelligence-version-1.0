from __future__ import annotations

import io
import wave

import numpy as np
from PIL import Image

from .pipeline import MultimodalPipeline
from .types import ModalityInput


def _png(pattern: int, noise: int) -> bytes:
    rng = np.random.default_rng(noise)
    array = rng.integers(0, 12, (32, 32, 3), dtype=np.uint8)
    if pattern == 0:
        array[:, 5:11] += 220
    else:
        array[:, 21:27] += 220
    stream = io.BytesIO(); Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def _wav(tone: int, noise: int) -> bytes:
    rng = np.random.default_rng(noise)
    t = np.arange(800) / 8000.0
    samples = .65 * np.sin(2 * np.pi * (300 if tone == 0 else 1100) * t)
    samples += rng.normal(0, .015, samples.shape)
    stream = io.BytesIO()
    with wave.open(stream, "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(8000)
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return stream.getvalue()


def _nearest(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    distances = ((test_x[:, None, :] - train_x[None, :, :]) ** 2).sum(axis=-1)
    return train_y[distances.argmin(axis=1)]


def complementary_signal_benchmark(seed: int = 0) -> dict[str, float | int | str]:
    """Task label is XOR(image position, audio frequency); each modality alone is ambiguous."""
    pipeline = MultimodalPipeline()
    records = []
    for split, offsets in (("train", range(6)), ("test", range(100, 106))):
        for visual in (0, 1):
            for audible in (0, 1):
                for offset in offsets:
                    key = seed + offset + visual * 1000 + audible * 100
                    reps, fused = pipeline.process([
                        ModalityInput("vision", f"v-{key}", _png(visual, key)),
                        ModalityInput("audio", f"a-{key}", _wav(audible, key)),
                    ])
                    records.append((split, visual ^ audible, reps[0].latent, reps[1].latent, fused.fused_latent))
    train = [r for r in records if r[0] == "train"]
    test = [r for r in records if r[0] == "test"]
    y_train, y_test = np.array([r[1] for r in train]), np.array([r[1] for r in test])
    scores = {}
    for name, index in (("vision", 2), ("audio", 3), ("fusion", 4)):
        prediction = _nearest(np.stack([r[index] for r in train]), y_train,
                              np.stack([r[index] for r in test]))
        scores[f"{name}_accuracy"] = float(np.mean(prediction == y_test))
    scores["best_unimodal_accuracy"] = max(scores["vision_accuracy"], scores["audio_accuracy"])
    scores["fusion_improvement"] = scores["fusion_accuracy"] - scores["best_unimodal_accuracy"]
    return {"task": "complementary_visual_audio_xor", "train_examples": len(train),
            "test_examples": len(test), **scores}
