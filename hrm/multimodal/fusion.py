from __future__ import annotations

import numpy as np

from hrm.multimodal.types import FusionResult, ModalityContradiction, ModalityRepresentation


class FusionEngine:
    def __init__(self, strategy: str = "confidence_weighted", modality_order: list[str] | None = None) -> None:
        self.strategy = strategy
        self.modality_order = modality_order or []
        self.gating_params: dict[str, float] = {modality: 1.0 for modality in self.modality_order}

    def fuse(self, representations: list[ModalityRepresentation], expected_modalities: list[str] | None = None) -> FusionResult:
        modalities = [rep.modality for rep in representations]
        confidences = {rep.modality: float(rep.confidence) for rep in representations}
        weights = self._compute_weights(representations)
        fused = self._weighted_sum(representations, weights)
        missing = tuple(sorted(set(expected_modalities or []) - set(modalities))) if expected_modalities else tuple()
        contradictions = self._detect_contradictions(representations)
        provenance = tuple(f"{rep.modality}:{rep.source_id}" for rep in representations)
        diagnostics = {
            "strategy": self.strategy,
            "requested_modalities": tuple(expected_modalities) if expected_modalities else tuple(modalities),
            "present_modalities": tuple(modalities),
            "missing_modalities": missing,
            "latent_norms": {rep.modality: float(np.linalg.norm(rep.latent)) for rep in representations},
            "modality_weights": weights,
            "weight_sum": float(sum(weights.values())),
            "contradictions": [c.__dict__ for c in contradictions],
        }
        return FusionResult(
            fused_latent=fused,
            modality_weights=weights,
            modality_confidences=confidences,
            missing_modalities=missing,
            contradictions=tuple(contradictions),
            provenance=provenance,
            diagnostics=diagnostics,
        )

    def _compute_weights(self, representations: list[ModalityRepresentation]) -> dict[str, float]:
        if not representations:
            return {}
        if self.strategy == "confidence_weighted":
            total_confidence = sum(max(rep.confidence, 0.0) for rep in representations)
            if total_confidence > 0.0:
                return {rep.modality: float(max(rep.confidence, 0.0) / total_confidence) for rep in representations}
            return {rep.modality: 1.0 / len(representations) for rep in representations}
        if self.strategy == "uniform":
            return {rep.modality: 1.0 / len(representations) for rep in representations}
        if self.strategy == "learned_gating":
            scores = []
            for rep in representations:
                base = self.gating_params.get(rep.modality, 1.0)
                scores.append(max(rep.confidence, 0.0) * base)
            weights = self._softmax(np.asarray(scores, dtype=np.float32))
            return {rep.modality: float(weights[idx]) for idx, rep in enumerate(representations)}
        return {rep.modality: 1.0 / len(representations) for rep in representations}

    def _weighted_sum(self, representations: list[ModalityRepresentation], weights: dict[str, float]) -> np.ndarray:
        fused: np.ndarray | None = None
        for rep in representations:
            latent = np.asarray(rep.latent, dtype=np.float32)
            weight = weights.get(rep.modality, 0.0)
            scaled = latent * weight
            fused = scaled if fused is None else fused + scaled
        if fused is None:
            fused = np.zeros(1, dtype=np.float32)
        return fused

    def _detect_contradictions(self, representations: list[ModalityRepresentation]) -> list[ModalityContradiction]:
        contradictions: list[ModalityContradiction] = []
        if len(representations) < 2:
            return contradictions
        for i, rep_a in enumerate(representations):
            for rep_b in representations[i + 1 :]:
                if rep_a.modality == rep_b.modality:
                    continue
                similarity = self._cosine_similarity(rep_a.latent, rep_b.latent)
                if similarity < 0.7:
                    contradictions.append(
                        ModalityContradiction(
                            contradiction_id=f"{rep_a.modality}_vs_{rep_b.modality}",
                            modalities=(rep_a.modality, rep_b.modality),
                            contradiction_type="latent_similarity_disagreement",
                            severity=float(1.0 - similarity),
                            confidence=float((rep_a.confidence + rep_b.confidence) / 2.0),
                            evidence={
                                rep_a.modality: float(np.mean(rep_a.latent)),
                                rep_b.modality: float(np.mean(rep_b.latent)),
                                "similarity": float(similarity),
                            },
                            resolution=None,
                        )
                    )
        return contradictions

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a_flat = np.asarray(a, dtype=np.float32).ravel()
        b_flat = np.asarray(b, dtype=np.float32).ravel()
        if a_flat.size == 0 or b_flat.size == 0:
            return 0.0
        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values)
        exp_values = np.exp(shifted)
        total = np.sum(exp_values)
        return exp_values / (total + 1e-9)
