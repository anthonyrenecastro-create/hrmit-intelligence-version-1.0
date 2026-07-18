from __future__ import annotations

import numpy as np


def compute_interactions(phi_shape: tuple[int, ...]) -> np.ndarray:
    # Interactions are explicit and deterministic; currently zero in fixed-carrier phase.
    return np.zeros(phi_shape, dtype=np.float32)
