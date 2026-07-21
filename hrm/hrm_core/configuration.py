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
    topology: bool = True

    def is_full_arm(self) -> bool:
        return (
            self.input_projection
            and self.diffusion
            and self.reaction
            and self.memory
            and self.cognition
            and self.hierarchy
            and self.topology
        )


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
    topology_enabled: bool = False
    topology_max_add_nodes: int = 1
    topology_max_remove_nodes: int = 1
    topology_metric_scale_step: float = 0.05
    ablations: MechanismAblations = field(default_factory=MechanismAblations)


def topology_enabled_config(**overrides: object) -> HRMTransitionConfig:
    return HRMTransitionConfig(topology_enabled=True, **overrides)
