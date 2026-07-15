import json
from pathlib import Path

import pytest

from hrm.theory import HRMTheory


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
    assert learning_result["signals"]["baseline_phase"] == "Stage 1 placeholder"
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
