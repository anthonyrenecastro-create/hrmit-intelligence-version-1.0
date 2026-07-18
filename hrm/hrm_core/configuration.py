from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MechanismAblations:
    input_projection: bool = True
    diffusion: bool = True
    reaction: bool = True
    memory: bool = True
    cognition: bool = True
    hierarchy: bool = True


@dataclass(frozen=True)
class HRMTransitionConfig:
    dt: float = 0.1
    field_bound: float = 4.0
    cognition_bound: float = 4.0
    memory_decay: float = 0.08
    diffusion_gain: float = 0.2
    reaction_gain: float = 0.5
    reaction_saturation: float = 0.35
    memory_gain: float = 0.15
    cognition_gain: float = 0.12
    guidance_gain: float = 0.08
    hierarchy_gain: float = 0.1
    ledger_relative_tolerance: float = 1e-5
    max_field_norm: float = 128.0
    min_field_variance: float = 1e-7
    deterministic_mode: bool = True
    ablations: MechanismAblations = field(default_factory=MechanismAblations)
