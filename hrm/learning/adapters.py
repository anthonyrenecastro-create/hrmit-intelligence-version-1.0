from __future__ import annotations

import numpy as np


class LinearAdapter:
    def __init__(self, feature_dim: int, parameters: np.ndarray | None = None) -> None:
        self.feature_dim = feature_dim
        self.parameters = (np.zeros(feature_dim + 1, np.float32) if parameters is None
                           else np.asarray(parameters, np.float32).copy())
        if self.parameters.shape != (feature_dim + 1,):
            raise ValueError("Parameter shape mismatch")

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, np.float32)
        logits = np.clip(features @ self.parameters[:-1] + self.parameters[-1], -30, 30)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.probabilities(features) >= .5).astype(np.int64)

    def trained_candidate(self, x: np.ndarray, y: np.ndarray, *, learning_rate: float,
                          epochs: int, max_update_norm: float) -> tuple[np.ndarray, float]:
        candidate = self.parameters.copy()
        for _ in range(epochs):
            probability = 1.0 / (1.0 + np.exp(-np.clip(x @ candidate[:-1] + candidate[-1], -30, 30)))
            error = probability - y
            gradient = np.concatenate([(x.T @ error) / len(x), [error.mean()]]).astype(np.float32)
            candidate -= learning_rate * gradient
        delta = candidate - self.parameters
        norm = float(np.linalg.norm(delta))
        if norm > max_update_norm:
            candidate = self.parameters + delta * (max_update_norm / norm)
        return candidate, float(np.linalg.norm(candidate - self.parameters))
