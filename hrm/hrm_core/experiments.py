from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .configuration import HRMTransitionConfig
from .mechanisms import (
    CognitionMechanism,
    DiffusionMechanism,
    HierarchyMechanism,
    InputProjectionMechanism,
    MemoryMechanism,
    ReactionMechanism,
    RegionalMechanism,
)
from .mechanisms.base import HRMInput
from .state import HRMState, make_initial_state
from .transition import CanonicalTransitionEngine


@dataclass(frozen=True)
class ExperimentReport:
    final_state: HRMState
    metrics: dict[str, float]
    ledger_errors: dict[str, float]
    activations: dict[str, float]
    elapsed_seconds: float


def build_engine(config: HRMTransitionConfig) -> CanonicalTransitionEngine:
    mechanisms = [
        InputProjectionMechanism(enabled=config.ablations.input_projection, gain=0.35),
        DiffusionMechanism(enabled=config.ablations.diffusion, gain=config.diffusion_gain),
        ReactionMechanism(enabled=config.ablations.reaction, gain=config.reaction_gain, saturation=config.reaction_saturation),
        RegionalMechanism(enabled=True, gain=0.05),
        MemoryMechanism(enabled=config.ablations.memory, gain=config.memory_gain, decay=config.memory_decay),
        CognitionMechanism(enabled=config.ablations.cognition, gain=config.cognition_gain, guidance_gain=config.guidance_gain),
        HierarchyMechanism(enabled=config.ablations.hierarchy, gain=config.hierarchy_gain),
    ]
    return CanonicalTransitionEngine(config=config, mechanisms=mechanisms)


def _ring_drive(step: int, node_count: int, channels: int, horizon: int) -> np.ndarray:
    idx = np.arange(node_count, dtype=np.float32)
    phase = 2.0 * np.pi * (step / max(1, horizon))
    base = np.sin((idx / max(1, node_count - 1)) * 2.0 * np.pi + phase)
    drive = np.repeat(base[:, None], channels, axis=1)
    return drive.astype(np.float32)


def run_spatial_reconstruction(seed: int = 0, steps: int = 32, config: HRMTransitionConfig | None = None) -> ExperimentReport:
    config = config or HRMTransitionConfig()
    state = make_initial_state(node_count=32, channels=8, latent_dim=16, memory_capacity=16, seed=seed)
    engine = build_engine(config)
    last_ledger = None
    start = time.perf_counter()
    for step in range(steps):
        drive = _ring_drive(step, node_count=32, channels=8, horizon=steps)
        result = engine.step(
            state,
            HRMInput(field_drive=drive, metadata={"memory_query": f"spatial_step_{step}"}),
        )
        state = result.state
        last_ledger = result.ledger

    assert last_ledger is not None
    return ExperimentReport(
        final_state=state,
        metrics={k: float(v) for k, v in last_ledger.metrics.items()},
        ledger_errors={
            "max_abs": float(last_ledger.max_abs_reconstruction_error),
            "max_rel": float(last_ledger.max_rel_reconstruction_error),
        },
        activations={k: float(v) for k, v in last_ledger.proposal_activations.items()},
        elapsed_seconds=time.perf_counter() - start,
    )


def run_sequence_memory(seed: int = 0, steps: int = 24, config: HRMTransitionConfig | None = None) -> ExperimentReport:
    config = config or HRMTransitionConfig()
    state = make_initial_state(node_count=16, channels=6, latent_dim=12, memory_capacity=16, seed=seed)
    engine = build_engine(config)
    last_ledger = None
    start = time.perf_counter()
    for step in range(steps):
        token = (step % 4) / 3.0
        drive = np.zeros((16, 6), dtype=np.float32)
        drive[:, step % 6] = token
        result = engine.step(
            state,
            HRMInput(field_drive=drive, metadata={"memory_query": f"sequence_token_{step % 4}"}),
        )
        state = result.state
        last_ledger = result.ledger

    assert last_ledger is not None
    return ExperimentReport(
        final_state=state,
        metrics={k: float(v) for k, v in last_ledger.metrics.items()},
        ledger_errors={
            "max_abs": float(last_ledger.max_abs_reconstruction_error),
            "max_rel": float(last_ledger.max_rel_reconstruction_error),
        },
        activations={k: float(v) for k, v in last_ledger.proposal_activations.items()},
        elapsed_seconds=time.perf_counter() - start,
    )
