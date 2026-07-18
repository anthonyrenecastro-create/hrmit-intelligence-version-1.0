import json
from pathlib import Path

import pytest

from hrm.theory import HRMTheory


def test_stage1_baseline_runtime_message(tmp_path: Path) -> None:
    theory = HRMTheory()
    result = theory.run_stage(1, seed=1, output_dir=tmp_path / "hrm_baseline_outputs")

    assert result["stage"] == "HRM reasoning core"
    assert "runtime_message" in result["result"]
    assert result["result"]["runtime_message"] in {
        "Stage 1 executed the real JAX baseline pipeline.",
        "Stage 1 used the placeholder baseline path because JAX was unavailable.",
    }


def test_stage1_baseline_training_mode(tmp_path: Path) -> None:
    theory = HRMTheory()
    result = theory.run_stage(
        1,
        seed=1,
        output_dir=tmp_path / "hrm_baseline_outputs",
        train=True,
        train_epochs=1,
        train_learning_rate=0.01,
    )

    assert result["stage"] == "HRM reasoning core"
    assert result["result"]["runtime_message"] in {
        "Stage 1 executed the JAX baseline training pipeline.",
        "Stage 1 used the placeholder baseline path because JAX was unavailable.",
    }
    assert "training_history" in result["result"]["baseline_record"] or "training_epochs" not in result["result"]["baseline_record"]
    if "training_epochs" in result["result"]["baseline_record"]:
        assert result["result"]["baseline_record"]["training_epochs"] == 1


def test_stage1_training_reduces_validation_loss_and_exceeds_recurrent_baseline(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    from hrm.baseline import BASELINE_CONFIG, evaluate_non_hrm_baseline, train_baseline_pipeline

    record = train_baseline_pipeline(seed=0, epochs=3, learning_rate=0.02, save_artifacts=False, output_dir=tmp_path)

    assert record["training_epochs"] == 3
    assert len(record["training_history"]) == 3
    assert record["validation_loss_after"] <= record["validation_loss_before"]
    assert record["validation_improvement"] >= 0.0
    assert record["hrm_vs_non_hrm"] is True
    assert record["ledger_pass"] is True
    assert record["bounded_pass"] is True
    assert record["finite"] is True
    assert float(record["non_hrm_baseline_loss"]) >= 0.0
    assert record["validation_loss_after"] <= float(record["non_hrm_baseline_loss"])


def test_non_hrm_recurrent_baseline_has_valid_loss() -> None:
    pytest.importorskip("jax")
    from hrm.baseline import BASELINE_CONFIG, evaluate_non_hrm_baseline

    loss = evaluate_non_hrm_baseline(BASELINE_CONFIG, seed=7)
    assert loss >= 0.0


def test_stage2_and_stage3_smoke_flow() -> None:
    theory = HRMTheory()
    stage2 = theory.run_stage(2, baseline_record=None, plan_query="Smoke test planning")
    stage3 = theory.run_stage(
        3,
        baseline_record=None,
        api_endpoint="status",
        api_payload={"health": True},
        inference_provider="mock_deterministic",
    )

    assert stage2["stage"] == "Long-term memory and planning"
    assert stage2["result"]["plan"]["query"] == "Smoke test planning"
    assert stage2["result"]["memory_persistence"]["saved"] is True
    assert stage2["result"]["memory_persistence"]["loaded_entries"] > 0
    assert stage3["stage"] == "Tool use and verification"
    assert stage3["result"]["verification"]["overall_verified"] is True
    assert stage3["result"]["verification"]["schema_valid"] is True
    assert "governed_contract" in stage3["result"]
    assert stage3["result"]["governed_contract"]["provider_used"] == "mock_deterministic"
    assert stage3["result"]["governed_contract"]["state_version_after"] > stage3["result"]["governed_contract"]["state_version_before"]
    assert stage3["result"]["governed_contract"]["response_envelope"]["tool_results_used"]


def test_stage4_multimodal_perception_smoke() -> None:
    theory = HRMTheory()
    result = theory.run_stage(4, modality_query="HRM sensory check", include_modalities=["vision", "audio", "structured"])

    assert result["stage"] == "Multimodal perception"
    assert result["result"]["phase"] == "Stage 4"
    assert result["result"]["hrm_state_projection"]["delta_norm"] > 0.0
    assert set(result["result"]["modalities"]) == {"vision", "audio", "structured"}
    assert result["result"]["fusion"]["provenance"]
    assert "combined_embedding_summary" in result["result"]


def test_stage4_multimodal_governed_contract() -> None:
    theory = HRMTheory()
    result = theory.run_stage(
        4,
        modality_query="Governed multimodal check",
        include_modalities=["vision", "audio", "structured"],
        use_governed_multimodal=True,
        inference_provider="mock_deterministic",
    )

    governed = result["result"]["governed_contract"]
    assert governed["provider_used"] == "mock_deterministic"
    assert governed["state_version_after"] > governed["state_version_before"]
    assert governed["response_envelope"]["safety_status"] == "ok"


def test_stage5_distributed_cognition_contains_expected_keys() -> None:
    theory = HRMTheory()
    result = theory.run_stage(
        5,
        baseline_record=None,
        consensus_query="Coordinate distributed HRM reasoning",
        agent_roles=["safety", "efficiency", "planning", "recovery"],
    )

    assert result["stage"] == "Distributed cognition"
    coordination = result["result"]["coordination"]
    assert coordination["agent_count"] == 4
    assert "reasoning_traces" in coordination
    assert len(coordination["reasoning_traces"]) == 4
    assert coordination["consensus"]["agreement_score"] >= 0.0
    assert set(coordination["consensus"]["role_influence"].keys()) == {"safety", "efficiency", "planning", "recovery"}
    assert all(value >= 0.0 for value in coordination["consensus"]["role_influence"].values())
    assert coordination["distributed_plan"]["plan_steps"]


def test_stage5_distributed_cognition_governed_contract() -> None:
    theory = HRMTheory()
    result = theory.run_stage(
        5,
        baseline_record=None,
        consensus_query="Coordinate distributed HRM reasoning",
        agent_roles=["safety", "efficiency", "planning", "recovery"],
        use_governed_consensus=True,
        inference_provider="mock_deterministic",
    )

    governed = result["result"]["governed_contract"]
    assert governed["provider_used"] == "mock_deterministic"
    assert governed["state_version_after"] > governed["state_version_before"]
    assert governed["response_envelope"]["safety_status"] == "ok"


def test_stage6_learning_systems_prefers_expected_structure() -> None:
    theory = HRMTheory()
    result = theory.run_stage(
        6,
        baseline_record=None,
        learning_rate=0.08,
        starting_preferences={"exploration": 0.3, "safety": 0.4, "efficiency": 0.3, "bias": 0.1},
    )

    assert result["stage"] == "Learning systems"
    learning_result = result["result"]["learning_result"]
    assert result["result"]["adaptation_mode"] == "controlled_heuristic"
    assert learning_result["signals"]["baseline_phase"] in {"Stage 1 placeholder", "Stage 1 baseline"}
    assert learning_result["preference_model"]["weights"]["safety"] == pytest.approx(0.432, rel=1e-3)
    assert learning_result["adaptation_metrics"]["memory_growth"] >= 1
    assert learning_result["adaptation_metrics"]["adaptation_score"] >= 0.0


def test_runner_stage5_and_stage6_end_to_end(tmp_path: Path) -> None:
    theory = HRMTheory()
    stage5 = theory.run_stage(
        5,
        baseline_record=None,
        consensus_query="Coordinate distributed HRM reasoning",
        agent_roles=["safety", "efficiency", "planning", "recovery"],
    )
    stage6 = theory.run_stage(
        6,
        baseline_record=None,
        learning_rate=0.08,
        starting_preferences={"exploration": 0.3, "safety": 0.4, "efficiency": 0.3, "bias": 0.1},
    )

    assert stage5["result"]["coordination"]["agent_count"] == 4
    assert stage6["result"]["learning_result"]["preference_model"]["bias"] == pytest.approx(0.18, rel=1e-3)
    assert json.dumps(stage5)
    assert json.dumps(stage6)
