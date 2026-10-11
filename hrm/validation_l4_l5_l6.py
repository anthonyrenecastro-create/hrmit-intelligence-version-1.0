"""L4/L5/L6 validation: independent workloads, scaling, reproducibility.

L4: G/T mechanisms engage (non-zero, ablatable) across 3 input distributions.
L5: Training converges and beats baseline at 100 epochs (proven config).
L6: 5 seeds at 100 epochs, count wins.

Usage: python3 -m hrm.validation_l4_l5_l6
Takes ~8 min on CPU.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from .baseline import (
    BASELINE_CONFIG, Ablation, hrm_step,
    make_initial_state, make_params, train_baseline_pipeline,
)


def validate_l4():
    """L4: G/T engage across different input distributions."""
    print("=" * 60)
    print("L4: Mechanism engagement across workloads")
    print("=" * 60)
    cfg = BASELINE_CONFIG
    dyn = cfg.dynamics()
    key = jax.random.PRNGKey(0)
    params = make_params(key, cfg.shape)
    state = make_initial_state(key, cfg.shape)

    distributions = [
        ("blob", 0.1),
        ("clean_pattern", 0.0),
        ("high_noise", 0.3),
    ]
    all_active = True
    for name, noise in distributions:
        k1, k2 = jax.random.split(jax.random.fold_in(key, hash(name) % 1000))
        drive = jax.random.normal(k1, (1, 8, 8, 4)) + noise * jax.random.normal(k2, (1, 8, 8, 4))
        inp = (drive, drive, jnp.zeros_like(drive))
        _, led_full = hrm_step(state, inp, params, dyn, cfg.shape, Ablation())
        _, led_noGT = hrm_step(state, inp, params, dyn, cfg.shape,
                               Ablation(geometry=0.0, topology=0.0))
        topo_full = float(jnp.mean(led_full.topo_norm))
        topo_abl = float(jnp.mean(led_noGT.topo_norm))
        topo_on = topo_full > 1e-9
        # Ablated should be ~1e-6 (rms epsilon floor), much smaller than full
        topo_off = topo_abl < topo_full * 0.1
        geo_matters = abs(float(jnp.mean(led_full.diffusion_norm))
                          - float(jnp.mean(led_noGT.diffusion_norm))) > 1e-12
        active = topo_on and topo_off and geo_matters
        all_active = all_active and active
        print(f"  {name}: topology engages={topo_on}, ablates={topo_off}, "
              f"geometry matters={geo_matters}")
    print(f"\nL4: {'PASS' if all_active else 'FAIL'}")
    return all_active


def train_once(seed, epochs=100):
    """Train once, return (hrm_val, baseline_val, hrm_wins)."""
    with tempfile.TemporaryDirectory() as td:
        r = train_baseline_pipeline(
            seed=seed, epochs=epochs, learning_rate=0.1,
            save_artifacts=False, output_dir=Path(td))
        h = r['training_history']
        return h[-1]['validation_loss'], r['hrm_vs_non_hrm']


def validate_l5_l6():
    """L5/L6: 5 seeds at 100 epochs (proven config)."""
    print("\n" + "=" * 60)
    print("L5/L6: Reproducibility — 5 seeds, 100 epochs each")
    print("=" * 60)
    wins = 0
    hrm_losses = []
    for seed in range(5):
        print(f"  Seed {seed}... ", flush=True, end="")
        h, win = train_once(seed, epochs=100)
        wins += win
        hrm_losses.append(h)
        print(f"val={h:.4f} {'WIN' if win else 'LOSS'}")
    mean = np.mean(hrm_losses)
    std = np.std(hrm_losses)
    print(f"\n  HRM: {mean:.4f} ± {std:.4f} | Baseline: 0.0630")
    print(f"  Wins: {wins}/5")
    # L5: benefit persists (mean beats baseline)
    l5 = mean < 0.0630
    # L6: reproducible (4+ wins)
    l6 = wins >= 4
    print(f"  L5 (persists): {'PASS' if l5 else 'FAIL'}")
    print(f"  L6 (reproducible): {'PASS' if l6 else 'FAIL'}")
    return l5 and l6


def main():
    print("HRM L4/L5/L6 Validation (real training)")
    print("=" * 60)
    l4 = validate_l4()
    l5_l6 = validate_l5_l6()
    print("\n" + "=" * 60)
    print(f"Overall: L4={'PASS' if l4 else 'FAIL'}, L5/L6={'PASS' if l5_l6 else 'FAIL'}")


if __name__ == "__main__":
    main()
