from __future__ import annotations

import json
from pathlib import Path

from hrm.integration import IntegratedRuntime, MockDeterministicProvider, OperatingMode


def test_governed_text_vertical_slice_commits_state_once_and_persists_checkpoint(tmp_path: Path) -> None:
    runtime = IntegratedRuntime(
        mode=OperatingMode.SOVEREIGN_LOCAL,
        preferred_provider="mock_deterministic",
        session_id="test-session",
        checkpoint_dir=tmp_path,
    )
    runtime.register_provider(MockDeterministicProvider())

    before_version = runtime.state.version
    result = runtime.process_text_request("Provide a safe deterministic summary.")

    assert result.state_version_before == before_version
    assert result.state_version_after == before_version + 1
    assert runtime.state.version == before_version + 1
    assert result.provider_used == "mock_deterministic"
    assert result.ledger_max_rel_error <= runtime.config.ledger_relative_tolerance
    assert "[confidence=" in result.response_text

    checkpoint_path = Path(result.checkpoint_path)
    assert checkpoint_path.exists()
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert payload["state_version_before"] == before_version
    assert payload["state_version_after"] == before_version + 1
    assert payload["inference"]["provider"] == "mock_deterministic"
    assert payload["ledger"]["max_rel"] <= runtime.config.ledger_relative_tolerance
    assert payload["response_envelope"]["answer_content"]


def test_stage2_governed_text_path_is_exposed_via_theory() -> None:
    from hrm.theory import HRMTheory

    theory = HRMTheory()
    output = theory.run_stage(
        2,
        plan_query="Assess memory and planning",
        use_governed_text=True,
        governed_input="Respond with governed path only.",
        inference_provider="mock_deterministic",
    )
    stage_result = output["result"]

    assert "governed_text_vertical_slice" in stage_result
    governed = stage_result["governed_text_vertical_slice"]
    assert governed["provider_used"] == "mock_deterministic"
    assert governed["state_version_after"] > governed["state_version_before"]
