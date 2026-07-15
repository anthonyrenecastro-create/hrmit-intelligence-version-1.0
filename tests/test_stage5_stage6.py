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


def test_stage2_and_stage3_smoke_flow() -> None:
    theory = HRMTheory()
    stage2 = theory.run_stage(2, baseline_record=None, plan_query="Smoke test planning")
    stage3 = theory.run_stage(3, baseline_record=None, api_endpoint="status", api_payload={"health": True})

    assert stage2["stage"] == "Long-term memory and planning"
    assert stage2["result"]["plan"]["query"] == "Smoke test planning"
    assert stage3["stage"] == "Tool use and verification"
    assert stage3["result"]["verification"]["overall_verified"] is True


def test_stage4_multimodal_perception_smoke() -> None:
    theory = HRMTheory()
    result = theory.run_stage(4, modality_query="HRM sensory check", include_modalities=["text", "image", "audio", "video"])

    assert result["stage"] == "Multimodal perception"
    assert result["result"]["phase"] == "Stage 4"
    assert result["result"]["readiness"] >= 0.0
    assert set(result["result"]["modalities"]) == {"text", "image", "audio", "video"}
    assert "combined_embedding_summary" in result["result"]


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
    assert coordination["distributed_plan"]["plan_steps"]


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
