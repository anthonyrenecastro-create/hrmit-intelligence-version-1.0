from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ModalityOutput:
    modality: str
    normalized_input: Any
    embedding: np.ndarray
    readiness: float
    summary: str


class ModalityAdapter:
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    def adapt(self, input_data: Any) -> ModalityOutput:
        raise NotImplementedError("Subclasses must implement adapt")


class TextAdapter(ModalityAdapter):
    def __init__(self) -> None:
        super().__init__("text", "Normalize and embed text input.")

    @staticmethod
    def _embed_text(text: str, dim: int = 32) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        weights = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        weights = weights[:dim] if weights.size >= dim else np.pad(weights, (0, dim - weights.size))
        norm = np.linalg.norm(weights) + 1e-9
        return np.tanh(weights / norm)

    def adapt(self, input_data: str) -> ModalityOutput:
        normalized = input_data.strip()
        embedding = self._embed_text(normalized)
        readiness = min(1.0, max(0.0, len(normalized) / 128))
        summary = f"Text length {len(normalized)}"
        return ModalityOutput("text", normalized, embedding, readiness, summary)


class ImageAdapter(ModalityAdapter):
    def __init__(self) -> None:
        super().__init__("image", "Normalize and embed image data.")

    @staticmethod
    def _embed_image(image: np.ndarray, dim: int = 32) -> np.ndarray:
        flattened = np.asarray(image, dtype=np.float32).ravel()
        if flattened.size == 0:
            flattened = np.zeros(dim, dtype=np.float32)
        values = flattened[:dim]
        if values.size < dim:
            values = np.pad(values, (0, dim - values.size), mode="constant")
        if np.linalg.norm(values) == 0:
            return values
        return np.tanh(values / (np.linalg.norm(values) + 1e-9))

    def adapt(self, input_data: Any) -> ModalityOutput:
        image = np.asarray(input_data, dtype=np.float32)
        normalized = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-9)
        embedding = self._embed_image(normalized)
        readiness = min(1.0, 0.5 + normalized.size / 1024)
        summary = f"Image shape {normalized.shape}"
        return ModalityOutput("image", normalized.tolist(), embedding, readiness, summary)


class AudioAdapter(ModalityAdapter):
    def __init__(self) -> None:
        super().__init__("audio", "Normalize and embed audio waveform data.")

    @staticmethod
    def _embed_audio(waveform: np.ndarray, dim: int = 32) -> np.ndarray:
        waveform = np.asarray(waveform, dtype=np.float32).ravel()
        if waveform.size == 0:
            waveform = np.zeros(dim, dtype=np.float32)
        values = waveform[:dim]
        if values.size < dim:
            values = np.pad(values, (0, dim - values.size), mode="constant")
        if np.linalg.norm(values) == 0:
            return values
        return np.tanh(values / (np.linalg.norm(values) + 1e-9))

    def adapt(self, input_data: Any) -> ModalityOutput:
        waveform = np.asarray(input_data, dtype=np.float32)
        normalized = waveform / (np.max(np.abs(waveform)) + 1e-9)
        embedding = self._embed_audio(normalized)
        readiness = min(1.0, 0.4 + normalized.size / 512)
        summary = f"Audio length {normalized.size}"
        return ModalityOutput("audio", normalized.tolist(), embedding, readiness, summary)


class VideoAdapter(ModalityAdapter):
    def __init__(self) -> None:
        super().__init__("video", "Normalize and embed video frame data.")
        self.image_adapter = ImageAdapter()

    def adapt(self, input_data: Any) -> ModalityOutput:
        frames = [np.asarray(frame, dtype=np.float32) for frame in input_data]
        embeddings = [self.image_adapter._embed_image(frame) for frame in frames]
        aggregated = np.mean(np.stack(embeddings), axis=0)
        readiness = min(1.0, 0.5 + len(frames) * 0.1)
        summary = f"Video frames {len(frames)}"
        normalized = [frame.tolist() for frame in frames]
        return ModalityOutput("video", normalized, aggregated, readiness, summary)


class PerceptionPipeline:
    def __init__(self) -> None:
        self.adapters = {
            "text": TextAdapter(),
            "image": ImageAdapter(),
            "audio": AudioAdapter(),
            "video": VideoAdapter(),
        }

    def process(self, modality: str, input_data: Any) -> ModalityOutput:
        if modality not in self.adapters:
            raise ValueError(f"Unsupported modality: {modality}")
        adapter = self.adapters[modality]
        return adapter.adapt(input_data)

    def integrate(self, modalities: dict[str, Any]) -> dict[str, Any]:
        outputs = {}
        embeddings = []
        readiness_scores = []
        for modality, data in modalities.items():
            output = self.process(modality, data)
            outputs[modality] = {
                "summary": output.summary,
                "readiness": output.readiness,
                "normalized_input": output.normalized_input,
            }
            embeddings.append(output.embedding)
            readiness_scores.append(output.readiness)

        combined_embedding = np.mean(np.stack(embeddings), axis=0) if embeddings else np.zeros(32, dtype=np.float32)
        overall_readiness = float(np.mean(readiness_scores)) if readiness_scores else 0.0
        return {
            "outputs": outputs,
            "combined_embedding": combined_embedding.tolist(),
            "overall_readiness": overall_readiness,
            "modalities": list(outputs.keys()),
        }

    @staticmethod
    def sample_inputs() -> dict[str, Any]:
        text = "HRM model receives multimodal sensory inputs for perception integration."
        image = np.linspace(0.0, 1.0, num=64, dtype=np.float32).reshape(8, 8)
        audio = np.sin(np.linspace(0.0, 2 * np.pi, 128, dtype=np.float32))
        video = [image, image[::-1], np.rot90(image), np.fliplr(image)]
        return {"text": text, "image": image, "audio": audio, "video": video}
