from __future__ import annotations

import subprocess
from pathlib import Path


def test_runner_stage4_and_stage5_smoke() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    python_executable = repo_root / ".venv" / "bin" / "python"

    result4 = subprocess.run(
        [str(python_executable), "hrm/runner.py", "--stage", "4"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert "Multimodal perception" in result4.stdout
    assert result4.returncode == 0

    result5 = subprocess.run(
        [str(python_executable), "hrm/runner.py", "--stage", "5"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert "Distributed cognition" in result5.stdout
    assert result5.returncode == 0

    result6 = subprocess.run(
        [str(python_executable), "hrm/runner.py", "--stage", "6", "--starting-preferences", '{"exploration": 0.1, "safety": 0.4, "efficiency": 0.4, "bias": 0.1}'],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert "Learning systems" in result6.stdout
    assert result6.returncode == 0
