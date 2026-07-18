from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .configuration import HRMTransitionConfig
from .proposals import TransitionCorrection


@dataclass(frozen=True)
class SafetyResult:
    phi: np.ndarray
    cognition_latent: np.ndarray
    corrections: tuple[TransitionCorrection, ...]
    rejected: bool
    rejection_reason: str | None


def apply_safe_transition(
    phi: np.ndarray,
    cognition_latent: np.ndarray,
    *,
    previous_phi: np.ndarray,
    config: HRMTransitionConfig,
) -> SafetyResult:
    corrections: list[TransitionCorrection] = []

    if not np.isfinite(phi).all() or not np.isfinite(cognition_latent).all():
        return SafetyResult(
            phi=previous_phi,
            cognition_latent=np.array(cognition_latent, copy=True),
            corrections=(
                TransitionCorrection(
                    correction_id="reject_non_finite",
                    reason="non_finite_detected",
                    block="Phi",
                    magnitude=0.0,
                ),
            ),
            rejected=True,
            rejection_reason="non_finite_detected",
        )

    bounded_phi = config.field_bound * np.tanh(phi / max(config.field_bound, 1e-9))
    phi_correction = bounded_phi - phi
    if np.any(np.abs(phi_correction) > 0.0):
        corrections.append(
            TransitionCorrection(
                correction_id="phi_bound_projection",
                reason="field_bound_projection",
                block="Phi",
                magnitude=float(np.linalg.norm(phi_correction)),
            )
        )
    phi = bounded_phi

    bounded_latent = config.cognition_bound * np.tanh(cognition_latent / max(config.cognition_bound, 1e-9))
    latent_correction = bounded_latent - cognition_latent
    if np.any(np.abs(latent_correction) > 0.0):
        corrections.append(
            TransitionCorrection(
                correction_id="cognition_bound_projection",
                reason="cognition_bound_projection",
                block="Cognition",
                magnitude=float(np.linalg.norm(latent_correction)),
            )
        )

    field_norm = float(np.linalg.norm(phi))
    if field_norm > config.max_field_norm:
        scale = config.max_field_norm / (field_norm + 1e-9)
        scaled = phi * scale
        corrections.append(
            TransitionCorrection(
                correction_id="phi_norm_scale",
                reason="field_norm_guardrail",
                block="Phi",
                magnitude=float(np.linalg.norm(scaled - phi)),
                metadata={"scale": float(scale)},
            )
        )
        phi = scaled

    variance = float(np.var(phi))
    if variance < config.min_field_variance:
        corrections.append(
            TransitionCorrection(
                correction_id="collapse_warning",
                reason="low_variance_collapse_risk",
                block="Phi",
                magnitude=float(config.min_field_variance - variance),
            )
        )

    return SafetyResult(
        phi=phi,
        cognition_latent=bounded_latent,
        corrections=tuple(corrections),
        rejected=False,
        rejection_reason=None,
    )
