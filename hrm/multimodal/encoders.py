from __future__ import annotations

import numpy as np

from .types import DecodedModality, ModalityRepresentation


def _project(features: np.ndarray, dim: int, seed: int) -> np.ndarray:
    features = np.asarray(features, np.float32).ravel()
    rng = np.random.default_rng(seed + features.size)
    matrix = rng.normal(0.0, 1.0 / np.sqrt(max(1, features.size)), (features.size, dim)).astype(np.float32)
    return np.tanh(features @ matrix).astype(np.float32)


class VisionEncoder:
    """Deterministic computed baseline using spatial/color statistics and gradients."""
    def __init__(self, latent_dim: int = 32) -> None:
        self.latent_dim = latent_dim

    def encode(self, decoded: DecodedModality) -> ModalityRepresentation:
        image = decoded.tensor.astype(np.float32) / 255.0
        h, w = image.shape[:2]
        ys = np.array_split(image, 4, axis=0)
        pooled = np.concatenate([np.concatenate([x.mean((0, 1)) for x in np.array_split(y, 4, axis=1)]) for y in ys])
        gx, gy = np.diff(image, axis=1), np.diff(image, axis=0)
        features = np.concatenate([pooled, image.mean((0, 1)), image.std((0, 1)),
                                   [np.mean(np.abs(gx)), np.mean(np.abs(gy))]])
        confidence = float(np.clip(min(h, w) / 64.0, 0.2, 1.0) * np.clip(image.std() * 4 + .2, .2, 1.0))
        return ModalityRepresentation("vision", decoded.source_id, _project(features, self.latent_dim, 11),
                                      confidence, decoded.mask, decoded.timestamp, "spatial-statistics-v1", decoded.metadata)


class AudioEncoder:
    def __init__(self, latent_dim: int = 32, bands: int = 16) -> None:
        self.latent_dim, self.bands = latent_dim, bands

    def encode(self, decoded: DecodedModality) -> ModalityRepresentation:
        x = decoded.tensor.astype(np.float32)
        if x.size < 32:
            raise ValueError("Audio requires at least 32 samples")
        windowed = x * np.hanning(x.size)
        spectrum = np.abs(np.fft.rfft(windowed))
        edges = np.linspace(0, spectrum.size, self.bands + 1, dtype=int)
        bands = np.array([np.log1p(spectrum[edges[i]:max(edges[i] + 1, edges[i + 1])].mean()) for i in range(self.bands)])
        zcr = np.mean(np.signbit(x[1:]) != np.signbit(x[:-1]))
        features = np.concatenate([bands, [x.mean(), x.std(), np.sqrt(np.mean(x*x)), zcr]])
        clipping = np.mean(np.abs(x) >= .999)
        confidence = float(np.clip(x.std() * 5, .05, 1.0) * (1.0 - clipping))
        metadata = {**decoded.metadata, "features": "log_spectral_bands+rms+zcr"}
        return ModalityRepresentation("audio", decoded.source_id, _project(features, self.latent_dim, 23),
                                      confidence, decoded.mask, decoded.timestamp, "spectral-baseline-v1", metadata)


class StructuredEncoder:
    def __init__(self, latent_dim: int = 32) -> None:
        self.latent_dim = latent_dim

    def encode(self, decoded: DecodedModality) -> ModalityRepresentation:
        mask = decoded.mask if decoded.mask is not None else np.ones(decoded.tensor.size, bool)
        features = np.concatenate([decoded.tensor, mask.astype(np.float32)])
        confidence = float(mask.mean())
        return ModalityRepresentation("structured", decoded.source_id, _project(features, self.latent_dim, 37),
                                      confidence, mask, decoded.timestamp, "schema-aware-projection-v1", decoded.metadata)
