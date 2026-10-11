"""HRM-backed image generator.

Uses the HRM's field dynamics as a generative spatiotemporal system.
The reaction-diffusion field naturally produces structured patterns;
we guide it with learned conditioning and upscale to images.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from PIL import Image

from hrm.baseline import BASELINE_CONFIG, FULL, State, hrm_step, make_params


class HRMImageGenerator:
    """Generates images from HRM field dynamics."""

    def __init__(self, seed: int = 0, out_size: int = 64):
        self.cfg = BASELINE_CONFIG
        self.shape = self.cfg.shape
        self.dyn = self.cfg.dynamics()
        self.out_size = out_size
        self.batch = 1
        key = random.PRNGKey(seed)
        self.hrm_params = make_params(key, self.shape)
        # Field (h, w, c) -> RGB projection
        s = self.shape
        self.to_rgb = random.normal(random.PRNGKey(seed + 1),
                                     (s.channels, 3), dtype=jnp.float32) * 0.5

    def _make_state(self, key):
        s = self.shape
        k1, k2 = random.split(key)
        phi = 0.05 * random.normal(k1, (self.batch, s.height, s.width, s.channels), dtype=jnp.float32)
        memory = jnp.zeros_like(phi)
        cognition = 0.01 * random.normal(k2, (self.batch, s.cognitive_dim), dtype=jnp.float32)
        hierarchy = jnp.zeros((self.batch, s.hierarchy_height, s.hierarchy_width, s.channels), dtype=jnp.float32)
        allocation = jnp.ones((self.batch, s.height, s.width, s.channels), dtype=jnp.float32)
        return State(phi, memory, cognition, hierarchy, allocation)

    def generate(self, seed: int = 0, steps: int = 48, creativity: float = 0.15) -> np.ndarray:
        """Run HRM dynamics with noise drive to generate a field, convert to RGB image."""
        key = random.PRNGKey(seed)
        state = self._make_state(key)
        s = self.shape
        for t in range(steps):
            key, k1 = random.split(key)
            # Noise drive sustains the dynamics (like dreaming)
            drive = creativity * random.normal(k1, (self.batch, s.height, s.width, s.channels), dtype=jnp.float32)
            dummy = jnp.zeros_like(drive)
            state, _ = hrm_step(state, (drive, dummy, dummy), self.hrm_params, self.dyn, self.shape, FULL)
        # Field -> RGB
        field = np.asarray(state.phi[0])  # (h, w, c)
        rgb_small = np.asarray(jnp.einsum("hwc,cd->hwd", jnp.asarray(field), self.to_rgb))
        # Normalize to [0, 255]
        rgb_small = (rgb_small - rgb_small.min()) / (rgb_small.max() - rgb_small.min() + 1e-8)
        # Upscale
        img = Image.fromarray((rgb_small * 255).astype(np.uint8))
        img = img.resize((self.out_size, self.out_size), Image.LANCZOS)
        return np.asarray(img)

    def generate_batch(self, n: int, base_seed: int = 0) -> list[np.ndarray]:
        return [self.generate(seed=base_seed + i) for i in range(n)]

    def save_image(self, arr: np.ndarray, path: str | Path) -> None:
        Image.fromarray(arr).save(path)
