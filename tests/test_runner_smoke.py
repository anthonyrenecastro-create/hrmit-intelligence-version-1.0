from __future__ import annotations

import subprocess
from pathlib import Path


def test_runner_stage4_and_stage5_smoke() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    result4 = subprocess.run(
        ["python3", "hrm/runner.py", "--stage", "4"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Multimodal perception" in result4.stdout
    assert result4.returncode == 0

    result5 = subprocess.run(
        ["python3", "hrm/runner.py", "--stage", "5"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Distributed cognition" in result5.stdout
    assert result5.returncode == 0
