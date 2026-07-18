from __future__ import annotations

from hrm.hrm_core import HRMTransitionConfig, run_sequence_memory, run_spatial_reconstruction


def test_spatial_reconstruction_experiment_runs_with_ledger_validity() -> None:
    config = HRMTransitionConfig()
    report = run_spatial_reconstruction(seed=0, steps=12, config=config)

    assert report.ledger_errors["max_rel"] <= config.ledger_relative_tolerance
    assert report.metrics["field_norm"] > 0.0
    assert report.activations["diffusion"] >= 0.0
    assert report.activations["reaction"] >= 0.0


def test_sequence_memory_experiment_activates_memory_and_cognition() -> None:
    config = HRMTransitionConfig()
    report = run_sequence_memory(seed=2, steps=12, config=config)

    assert report.ledger_errors["max_rel"] <= config.ledger_relative_tolerance
    assert report.activations["memory"] > 0.0
    assert report.activations["cognition"] > 0.0
