from __future__ import annotations

import numpy as np

from hrm.hrm_core import (
    HRMTransitionConfig,
    MechanismAblations,
    build_engine,
    freeze_state,
    make_initial_state,
    state_digest,
    state_from_dict,
    state_to_dict,
)
from hrm.hrm_core.mechanisms.base import HRMInput


def _drive(step: int, nodes: int, channels: int, horizon: int) -> np.ndarray:
    idx = np.arange(nodes, dtype=np.float32)
    phase = 2.0 * np.pi * step / max(1, horizon)
    base = np.sin((idx / max(1, nodes - 1)) * 2.0 * np.pi + phase)
    return np.repeat(base[:, None], channels, axis=1).astype(np.float32)


def test_state_round_trip_and_digest_stability() -> None:
    state = make_initial_state(node_count=12, channels=4, latent_dim=8, memory_capacity=6, seed=7)
    wire = state_to_dict(state)
    restored = state_from_dict(wire)

    assert state_digest(state) == state_digest(restored)
    assert restored.phi.phi.shape == (12, 4)
    assert restored.memory.capacity == 6


def test_state_digest_is_hex_and_rng_state_is_stable_under_round_trip() -> None:
    state = make_initial_state(node_count=6, channels=3, latent_dim=4, memory_capacity=4, seed=17)
    restored = state_from_dict(state_to_dict(state))

    digest = state_digest(state)
    assert len(digest) == 64
    int(digest, 16)
    assert state.rng_state == restored.rng_state


def test_snapshot_is_immutable_for_proposal_path() -> None:
    state = make_initial_state(node_count=8, channels=3, latent_dim=6, memory_capacity=4, seed=3)
    snapshot = freeze_state(state)

    try:
        snapshot.state.phi.phi[0, 0] = 1.0
        mutated = True
    except ValueError:
        mutated = False

    assert mutated is False


def test_deterministic_replay_same_seed_and_inputs() -> None:
    config = HRMTransitionConfig()
    engine = build_engine(config)

    initial = make_initial_state(node_count=10, channels=4, latent_dim=6, memory_capacity=6, seed=11)
    cloned = state_from_dict(state_to_dict(initial))

    state_a = initial
    state_b = cloned
    for step in range(12):
        drive = _drive(step, nodes=10, channels=4, horizon=12)
        inp = HRMInput(field_drive=drive, metadata={"memory_query": f"step_{step}"})
        state_a = engine.step(state_a, inp).state
        state_b = engine.step(state_b, inp).state

    assert state_digest(state_a) == state_digest(state_b)


def test_ledger_reconstruction_error_is_bounded() -> None:
    config = HRMTransitionConfig()
    engine = build_engine(config)
    state = make_initial_state(node_count=14, channels=5, latent_dim=8, memory_capacity=8, seed=21)

    for step in range(8):
        drive = _drive(step, nodes=14, channels=5, horizon=8)
        result = engine.step(state, HRMInput(field_drive=drive, metadata={"memory_query": "ledger"}))
        state = result.state

    assert result.ledger.max_abs_reconstruction_error >= 0.0
    assert result.ledger.max_rel_reconstruction_error <= config.ledger_relative_tolerance
    assert np.allclose(result.ledger.block_residuals["M"], 0.0)
    assert np.allclose(result.ledger.block_residuals["C"], 0.0)
    assert np.allclose(result.ledger.block_residuals["H"], 0.0)


def test_event_handling_advances_memory_write_head_and_records_transport() -> None:
    config = HRMTransitionConfig()
    engine = build_engine(config)
    state = make_initial_state(node_count=12, channels=4, latent_dim=8, memory_capacity=6, seed=23)
    result = engine.step(state, HRMInput(field_drive=_drive(1, nodes=12, channels=4, horizon=8), metadata={"memory_query": "event_path"}))

    assert result.ledger.proposed_events
    assert result.ledger.accepted_events
    assert any(event.event_type == "memory_write_head" and event.accepted for event in result.ledger.accepted_events)
    assert result.state.memory.write_index == (state.memory.write_index + 1) % state.memory.capacity
    assert all(record.mode == "identity" for record in result.ledger.transport_records)


def test_diffusion_ablation_changes_trajectory() -> None:
    base = make_initial_state(node_count=16, channels=6, latent_dim=10, memory_capacity=8, seed=5)

    enabled_engine = build_engine(HRMTransitionConfig())
    disabled_engine = build_engine(HRMTransitionConfig(ablations=MechanismAblations(diffusion=False)))

    state_on = state_from_dict(state_to_dict(base))
    state_off = state_from_dict(state_to_dict(base))

    for step in range(10):
        drive = _drive(step, nodes=16, channels=6, horizon=10)
        inp = HRMInput(field_drive=drive, metadata={"memory_query": "ablation"})
        state_on = enabled_engine.step(state_on, inp).state
        state_off = disabled_engine.step(state_off, inp).state

    divergence = float(np.linalg.norm(state_on.phi.phi - state_off.phi.phi))
    assert divergence > 1e-4


def test_reaction_ablation_changes_trajectory() -> None:
    base = make_initial_state(node_count=16, channels=6, latent_dim=10, memory_capacity=8, seed=9)

    enabled_engine = build_engine(HRMTransitionConfig())
    disabled_engine = build_engine(HRMTransitionConfig(ablations=MechanismAblations(reaction=False)))

    state_on = state_from_dict(state_to_dict(base))
    state_off = state_from_dict(state_to_dict(base))

    for step in range(10):
        drive = _drive(step, nodes=16, channels=6, horizon=10)
        inp = HRMInput(field_drive=drive, metadata={"memory_query": "reaction_ablation"})
        state_on = enabled_engine.step(state_on, inp).state
        state_off = disabled_engine.step(state_off, inp).state

    divergence = float(np.linalg.norm(state_on.phi.phi - state_off.phi.phi))
    assert divergence > 1e-4


def test_field_stays_bounded_in_operating_range() -> None:
    config = HRMTransitionConfig(field_bound=3.0, max_field_norm=64.0)
    engine = build_engine(config)
    state = make_initial_state(node_count=20, channels=4, latent_dim=8, memory_capacity=8, seed=13)

    for step in range(20):
        drive = 2.0 * _drive(step, nodes=20, channels=4, horizon=20)
        state = engine.step(state, HRMInput(field_drive=drive, metadata={"memory_query": "bounded"})).state

    assert np.max(np.abs(state.phi.phi)) <= 3.0 + 1e-4
