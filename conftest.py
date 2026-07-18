from __future__ import annotations

import sys
from pathlib import Path


def _inject_venv_site_packages() -> None:
    repo_root = Path(__file__).resolve().parent
    candidate = repo_root / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if candidate.exists():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


_inject_venv_site_packages()