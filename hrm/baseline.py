from __future__ import annotations

import csv
import dataclasses
import hashlib
import importlib.util
import json
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np

jax = None
jnp = None
lax = None
random = None

if importlib.util.find_spec("jax") is not None:
    import jax
    import jax.numpy as jnp
    from jax import lax, random

BACKEND = jax.default_backend() if jax is not None else "cpu"
RUN_PROFILE = "tpu_baseline" if BACKEND == "tpu" else "smoke"

PROFILE = {
    "batch": 32,
    "height": 16,
    "width": 16,
    "channels": 8,
    "cognitive_dim": 32,
    "hierarchy_height": 8,
    "hierarchy_width": 8,
    "steps": 256,
    "seed_count": 5,
    "long_horizons": (1024, 2048),
} if RUN_PROFILE == "tpu_baseline" else {
    "batch": 2,
    "height": 8,
    "width": 8,
    "channels": 4,
    "cognitive_dim": 16,
    "hierarchy_height": 4,
    "hierarchy_width": 4,
    "steps": 32,
    "seed_count": 2,
    "long_horizons": (64, 128),
}

OUTPUT_DIR = Path("hrm_baseline_outputs")


class BaselineError(Exception):
    """Base class for Stage 1 baseline execution errors."""


class BaselineImportError(BaselineError, ImportError):
    pass


class BaselineConfigurationError(BaselineError, ValueError):
    pass


class BaselineRuntimeError(BaselineError, RuntimeError):
    pass


def ensure_output_dir(path: Path = OUTPUT_DIR) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_jax() -> None:
    if jax is None:
        raise BaselineImportError(
            "JAX is required to execute the HRM baseline pipeline. Install jax to run Stage 1."
        )


@dataclass(frozen=True)
class ShapeConfig:
    batch: int
    height: int
    width: int
    channels: int
    cognitive_dim: int
    hierarchy_height: int
    hierarchy_width: int

    def validate(self) -> "ShapeConfig":
        if self.height % self.hierarchy_height != 0:
            raise BaselineConfigurationError(
                "ShapeConfig height must be divisible by hierarchy_height"
            )
        if self.width % self.hierarchy_width != 0:
            raise BaselineConfigurationError(
                "ShapeConfig width must be divisible by hierarchy_width"
            )
        if self.cognitive_dim < self.channels:
            raise BaselineConfigurationError(
                "ShapeConfig cognitive_dim must be greater than or equal to channels"
            )
        if min(asdict(self).values()) <= 0:
            raise BaselineConfigurationError("ShapeConfig values must all be positive")
        return self


class Dynamics(NamedTuple):
    dt: jax.Array
    diffusion: jax.Array
    reaction_gain: jax.Array
    reaction_saturation: jax.Array
    input_gain: jax.Array
    memory_gain: jax.Array
    memory_decay: jax.Array
    cognitive_gain: jax.Array
    guidance_gain: jax.Array
    hierarchy_gain: jax.Array
    allocation_floor: jax.Array
    field_bound: jax.Array
    latent_bound: jax.Array


@dataclass(frozen=True)
class ExperimentConfig:
    shape: ShapeConfig
    steps: int = 256
    dt: float = 0.08
    diffusion: float = 0.18
    reaction_gain: float = 0.55
    reaction_saturation: float = 0.35
    input_gain: float = 0.35
    memory_gain: float = 0.18
    memory_decay: float = 0.08
    cognitive_gain: float = 0.10
    guidance_gain: float = 0.16
    hierarchy_gain: float = 0.12
    allocation_floor: float = 0.20
    field_bound: float = 4.0
    latent_bound: float = 4.0
    perturb_step: int = 128
    perturb_strength: float = 1.5
    recovery_threshold_fraction: float = 0.10
    recovery_weight: float = 0.25
    safety_weight: float = 0.05
    ledger_tolerance: float = 5e-4

    def replace(self, **changes) -> "ExperimentConfig":
        return dataclasses.replace(self, **changes)

    def validate(self) -> "ExperimentConfig":
        self.shape.validate()
        if self.steps < 4:
            raise BaselineConfigurationError("ExperimentConfig.steps must be at least 4")
        if not (0 <= self.perturb_step < self.steps):
            raise BaselineConfigurationError(
                "ExperimentConfig.perturb_step must be between 0 and steps - 1"
            )
        if self.dt < 0:
            raise BaselineConfigurationError("ExperimentConfig.dt must be non-negative")
        if self.field_bound <= 0 or self.latent_bound <= 0:
            raise BaselineConfigurationError(
                "ExperimentConfig.field_bound and latent_bound must be positive"
            )
        if not (0 <= self.allocation_floor <= 1):
            raise BaselineConfigurationError(
                "ExperimentConfig.allocation_floor must be between 0 and 1"
            )
        return self

    def as_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def run_id(self, seed: int = 0, tag: str = "run") -> str:
        return f"{tag}-{self.config_hash()[:12]}-s{seed}"

    def dynamics(self) -> Dynamics:
        vals = (
            self.dt,
            self.diffusion,
            self.reaction_gain,
            self.reaction_saturation,
            self.input_gain,
            self.memory_gain,
            self.memory_decay,
            self.cognitive_gain,
            self.guidance_gain,
            self.hierarchy_gain,
            self.allocation_floor,
            self.field_bound,
            self.latent_bound,
        )
        return Dynamics(*(jnp.asarray(v, dtype=jnp.float32) for v in vals))


def _assert_shape(value: jax.Array, expected: tuple[int, ...], name: str) -> None:
    if value.shape != expected:
        raise BaselineConfigurationError(
            f"{name} shape mismatch: expected {expected}, got {value.shape}"
        )


def _assert_finite(value: jax.Array, name: str) -> None:
    if not bool(jnp.all(jnp.isfinite(value))):
        raise BaselineRuntimeError(f"{name} contains non-finite values")


def _assert_bounded(value: jax.Array, bound: float, name: str) -> None:
    max_abs = float(jnp.max(jnp.abs(value)))
    if max_abs > bound + 1e-6:
        raise BaselineRuntimeError(
            f"{name} exceeds bound {bound}: max abs {max_abs:.6f}"
        )


def _assert_probability(value: jax.Array, name: str) -> None:
    if not bool(jnp.all((value >= 0.0) & (value <= 1.0))):
        raise BaselineRuntimeError(f"{name} contains out-of-range probability values")


def validate_state(state: State, cfg: ExperimentConfig) -> None:
    _assert_shape(
        state.phi,
        (cfg.shape.batch, cfg.shape.height, cfg.shape.width, cfg.shape.channels),
        "phi",
    )
    _assert_shape(
        state.memory,
        (cfg.shape.batch, cfg.shape.height, cfg.shape.width, cfg.shape.channels),
        "memory",
    )
    _assert_shape(
        state.cognition,
        (cfg.shape.batch, cfg.shape.cognitive_dim),
        "cognition",
    )
    _assert_shape(
        state.hierarchy,
        (
            cfg.shape.batch,
            cfg.shape.hierarchy_height,
            cfg.shape.hierarchy_width,
            cfg.shape.channels,
        ),
        "hierarchy",
    )
    _assert_shape(
        state.allocation,
        (cfg.shape.batch, 1, 1, cfg.shape.channels),
        "allocation",
    )
    _assert_finite(state.phi, "phi")
    _assert_finite(state.memory, "memory")
    _assert_finite(state.cognition, "cognition")
    _assert_finite(state.hierarchy, "hierarchy")
    _assert_finite(state.allocation, "allocation")
    _assert_bounded(state.phi, cfg.field_bound, "phi")
    _assert_bounded(state.memory, cfg.field_bound, "memory")
    _assert_bounded(state.cognition, cfg.latent_bound, "cognition")
    _assert_bounded(state.hierarchy, cfg.field_bound, "hierarchy")
    _assert_probability(state.allocation, "allocation")


def validate_ledger(ledger: StepLedger, cfg: ExperimentConfig) -> None:
    for field_name, value in ledger._asdict().items():
        _assert_finite(value, field_name)
    if float(jnp.max(ledger.ledger_relative_error)) > cfg.ledger_tolerance + 1e-12:
        raise BaselineRuntimeError(
            f"Ledger reconstruction exceeded tolerance: max relative error "
            f"{float(jnp.max(ledger.ledger_relative_error)):.6e} > {cfg.ledger_tolerance:.6e}"
        )
    if float(jnp.min(ledger.field_variance)) < -1e-6:
        raise BaselineRuntimeError("Ledger field_variance is negative")
    _assert_probability(ledger.active_fraction, "active_fraction")

    coherence = float(jnp.min(ledger.coherence))
    if coherence <= 0.0:
        raise BaselineRuntimeError(
            f"Ledger coherence must be positive, got {coherence:.6f}"
        )

    if float(jnp.min(ledger.prediction_mse)) < 0.0:
        raise BaselineRuntimeError(
            "Ledger prediction_mse contains negative values"
        )

    if float(jnp.min(ledger.correction_norm)) < 0.0:
        raise BaselineRuntimeError(
            "Ledger correction_norm contains negative values"
        )

    if float(jnp.min(ledger.active_fraction)) < 0.0:
        raise BaselineRuntimeError(
            "Ledger active_fraction contains negative values"
        )

    if float(jnp.max(ledger.active_fraction)) > 1.0 + 1e-6:
        raise BaselineRuntimeError(
            f"Ledger active_fraction exceeds 1.0: max {float(jnp.max(ledger.active_fraction)):.6f}"
        )

    if float(jnp.min(ledger.field_variance)) < -1e-6:
        raise BaselineRuntimeError(
            f"Ledger field_variance has invalid values: min {float(jnp.min(ledger.field_variance)):.6f}"
        )

    if float(jnp.min(ledger.input_norm)) < 0.0:
        raise BaselineRuntimeError("Ledger input_norm contains negative values")


class State(NamedTuple):
    phi: jax.Array
    memory: jax.Array
    cognition: jax.Array
    hierarchy: jax.Array
    allocation: jax.Array


class Params(NamedTuple):
    field_to_cognition: jax.Array
    cognition_to_field: jax.Array
    hierarchy_mix: jax.Array
    predictor: jax.Array


class Inputs(NamedTuple):
    drive: jax.Array
    target: jax.Array
    perturb: jax.Array


class Ablation(NamedTuple):
    diffusion: float = 1.0
    reaction: float = 1.0
    memory: float = 1.0
    cognition: float = 1.0
    guidance: float = 1.0
    hierarchy: float = 1.0
    allocation: float = 1.0


class StepLedger(NamedTuple):
    input_norm: jax.Array
    diffusion_norm: jax.Array
    reaction_norm: jax.Array
    memory_norm: jax.Array
    cognition_norm: jax.Array
    guidance_norm: jax.Array
    hierarchy_norm: jax.Array
    perturb_norm: jax.Array
    correction_norm: jax.Array
    ledger_relative_error: jax.Array
    prediction_mse: jax.Array
    field_variance: jax.Array
    coherence: jax.Array
    active_fraction: jax.Array


SHAPE = ShapeConfig(
    batch=PROFILE["batch"],
    height=PROFILE["height"],
    width=PROFILE["width"],
    channels=PROFILE["channels"],
    cognitive_dim=PROFILE["cognitive_dim"],
    hierarchy_height=PROFILE["hierarchy_height"],
    hierarchy_width=PROFILE["hierarchy_width"],
).validate()

BASELINE_CONFIG = ExperimentConfig(
    shape=SHAPE,
    steps=PROFILE["steps"],
    perturb_step=PROFILE["steps"] // 2,
).validate()

FULL = Ablation()
ALL_RECORDS: list[dict] = []
RUN_ARTIFACTS: dict = {}


def bounded(x: jax.Array, bound: float) -> jax.Array:
    return bound * jnp.tanh(x / bound)


def laplacian_2d(x: jax.Array) -> jax.Array:
    return (
        4.0 * x
        - jnp.roll(x, 1, axis=1)
        - jnp.roll(x, -1, axis=1)
        - jnp.roll(x, 1, axis=2)
        - jnp.roll(x, -1, axis=2)
    )


def pool_to_hierarchy(x: jax.Array, shape: ShapeConfig) -> jax.Array:
    b, h, w, d = x.shape
    sh = h // shape.hierarchy_height
    sw = w // shape.hierarchy_width
    return x.reshape(b, shape.hierarchy_height, sh, shape.hierarchy_width, sw, d).mean(axis=(2, 4))


def prolong_from_hierarchy(hx: jax.Array, shape: ShapeConfig) -> jax.Array:
    sh = shape.height // shape.hierarchy_height
    sw = shape.width // shape.hierarchy_width
    return jnp.repeat(jnp.repeat(hx, sh, axis=1), sw, axis=2)


def rms_batch(x: jax.Array) -> jax.Array:
    axes = tuple(range(1, x.ndim))
    return jnp.sqrt(jnp.mean(x * x, axis=axes) + 1e-12)


def make_params(key: jax.Array, shape: ShapeConfig) -> Params:
    k1, k2, k3, k4 = random.split(key, 4)
    scale = 1.0 / jnp.sqrt(shape.channels)
    return Params(
        field_to_cognition=random.normal(k1, (shape.channels, shape.cognitive_dim), dtype=jnp.float32) * scale,
        cognition_to_field=random.normal(k2, (shape.cognitive_dim, shape.channels), dtype=jnp.float32) / jnp.sqrt(shape.cognitive_dim),
        hierarchy_mix=random.normal(k3, (shape.channels, shape.channels), dtype=jnp.float32) * scale,
        predictor=random.normal(k4, (shape.cognitive_dim, shape.channels), dtype=jnp.float32) / jnp.sqrt(shape.cognitive_dim),
    )


def make_initial_state(key: jax.Array, shape: ShapeConfig) -> State:
    k1, k2 = random.split(key)
    phi = 0.05 * random.normal(k1, (shape.batch, shape.height, shape.width, shape.channels), dtype=jnp.float32)
    memory = jnp.zeros_like(phi)
    cognition = 0.01 * random.normal(k2, (shape.batch, shape.cognitive_dim), dtype=jnp.float32)
    hierarchy = pool_to_hierarchy(phi, shape)
    allocation = jnp.ones((shape.batch, 1, 1, shape.channels), dtype=jnp.float32)
    return State(phi, memory, cognition, hierarchy, allocation)


def make_inputs(key: jax.Array, cfg: ExperimentConfig, perturb_strength: jax.Array | float | None = None) -> Inputs:
    shape = cfg.shape
    noise_key, perturb_key = random.split(key)
    yy = jnp.linspace(-1.0, 1.0, shape.height, dtype=jnp.float32)
    xx = jnp.linspace(-1.0, 1.0, shape.width, dtype=jnp.float32)
    Y, X = jnp.meshgrid(yy, xx, indexing="ij")

    t = jnp.arange(cfg.steps, dtype=jnp.float32)[:, None]
    b = jnp.arange(shape.batch, dtype=jnp.float32)[None, :]
    phase = 2.0 * jnp.pi * (t / float(cfg.steps) + b / float(shape.batch))

    cx = 0.45 * jnp.sin(phase)
    cy = 0.45 * jnp.cos(phase * 1.37)
    sigma = 0.18 + 0.04 * jnp.sin(phase * 0.73)

    X4 = X[None, None, :, :]
    Y4 = Y[None, None, :, :]
    target_scalar = jnp.exp(-((X4 - cx[:, :, None, None]) ** 2 + (Y4 - cy[:, :, None, None]) ** 2) / (2.0 * sigma[:, :, None, None] ** 2))
    channel_phase = jnp.linspace(0.7, 1.3, shape.channels, dtype=jnp.float32)[None, None, None, None, :]
    target = jnp.sin(channel_phase * jnp.pi * target_scalar[..., None])

    noise = 0.35 * random.normal(noise_key, target.shape, dtype=jnp.float32)
    drive = target + noise

    perturb_strength_value = jnp.asarray(cfg.perturb_strength, dtype=jnp.float32) if perturb_strength is None else jnp.asarray(perturb_strength, dtype=jnp.float32)
    perturb = jnp.zeros_like(target)
    damage = perturb_strength_value * random.normal(perturb_key, (shape.batch, shape.height, shape.width, shape.channels), dtype=jnp.float32)
    perturb = perturb.at[cfg.perturb_step].set(damage)
    return Inputs(drive=drive, target=target, perturb=perturb)


def seeded_components(seed: int, cfg: ExperimentConfig, perturb_strength: jax.Array | float | None = None) -> tuple[Params, State, Inputs]:
    root = random.PRNGKey(seed)
    pkey, skey, ikey = random.split(root, 3)
    return make_params(pkey, cfg.shape), make_initial_state(skey, cfg.shape), make_inputs(ikey, cfg, perturb_strength=perturb_strength)


def seeded_state_inputs(seed: int, cfg: ExperimentConfig, perturb_strength: jax.Array | float | None = None) -> tuple[State, Inputs]:
    root = random.PRNGKey(seed)
    skey, ikey = random.split(root, 2)
    return make_initial_state(skey, cfg.shape), make_inputs(ikey, cfg, perturb_strength=perturb_strength)


def hrm_step(state: State, inp: tuple[jax.Array, jax.Array, jax.Array], params: Params, dyn: Dynamics, shape: ShapeConfig, ablation: Ablation) -> tuple[State, StepLedger]:
    drive_t, target_t, perturb_t = inp
    phi, memory, cognition, hierarchy, allocation = state

    pooled_phi = phi.mean(axis=(1, 2))
    predicted_channels = cognition @ params.predictor

    observed_channels = drive_t.mean(axis=(1, 2))
    prediction_error = observed_channels - predicted_channels

    utility = jnp.abs(prediction_error) + 0.25 * jnp.mean(jnp.abs(phi), axis=(1, 2))
    priority = jax.nn.softmax(utility, axis=-1)
    priority = priority / (jnp.max(priority, axis=-1, keepdims=True) + 1e-8)
    soft_alloc = dyn.allocation_floor + (1.0 - dyn.allocation_floor) * priority
    next_allocation = ablation.allocation * soft_alloc[:, None, None, :] + (1.0 - ablation.allocation) * jnp.ones_like(allocation)

    a_input = dyn.dt * dyn.input_gain * drive_t * next_allocation
    a_diff = -dyn.dt * dyn.diffusion * laplacian_2d(phi) * next_allocation * ablation.diffusion
    reaction = dyn.reaction_gain * phi - dyn.reaction_saturation * phi ** 3
    a_react = dyn.dt * reaction * next_allocation * ablation.reaction
    a_memory = dyn.dt * dyn.memory_gain * (memory - phi) * next_allocation * ablation.memory
    cognitive_field = cognition @ params.cognition_to_field
    a_cognition = dyn.dt * dyn.cognitive_gain * cognitive_field[:, None, None, :] * next_allocation * ablation.cognition
    a_guidance = dyn.dt * dyn.guidance_gain * jnp.tanh(prediction_error[:, None, None, :]) * next_allocation * ablation.guidance
    h_prolong = prolong_from_hierarchy(hierarchy, shape)
    h_mixed = jnp.einsum("bhwd,df->bhwf", h_prolong, params.hierarchy_mix)
    a_hierarchy = dyn.dt * dyn.hierarchy_gain * (h_mixed - phi) * next_allocation * ablation.hierarchy
    a_perturb = perturb_t

    proposed_phi = phi + a_input + a_diff + a_react + a_memory + a_cognition + a_guidance + a_hierarchy + a_perturb
    safe_phi = bounded(proposed_phi, dyn.field_bound)
    a_correction = safe_phi - proposed_phi

    next_memory = bounded((1.0 - dyn.memory_decay) * memory + dyn.memory_decay * safe_phi, dyn.field_bound)
    error_pad = jnp.pad(prediction_error, ((0, 0), (0, shape.cognitive_dim - shape.channels)))
    cognitive_proposal = cognition + dyn.dt * jnp.einsum("bd,dc->bc", pooled_phi, params.field_to_cognition) + dyn.dt * error_pad
    next_cognition = bounded(cognitive_proposal, dyn.latent_bound)
    next_hierarchy = pool_to_hierarchy(safe_phi, shape)

    next_state = State(safe_phi, next_memory, next_cognition, next_hierarchy, next_allocation)

    observed_delta = safe_phi - phi
    reconstructed_delta = a_input + a_diff + a_react + a_memory + a_cognition + a_guidance + a_hierarchy + a_perturb + a_correction
    residual = observed_delta - reconstructed_delta
    rel_err = rms_batch(residual) / (rms_batch(observed_delta) + 1e-8)

    disagreement = 0.5 * (jnp.mean((safe_phi - jnp.roll(safe_phi, 1, axis=1)) ** 2, axis=(1, 2, 3)) + jnp.mean((safe_phi - jnp.roll(safe_phi, 1, axis=2)) ** 2, axis=(1, 2, 3)))

    ledger = StepLedger(
        input_norm=rms_batch(a_input),
        diffusion_norm=rms_batch(a_diff),
        reaction_norm=rms_batch(a_react),
        memory_norm=rms_batch(a_memory),
        cognition_norm=rms_batch(a_cognition),
        guidance_norm=rms_batch(a_guidance),
        hierarchy_norm=rms_batch(a_hierarchy),
        perturb_norm=rms_batch(a_perturb),
        correction_norm=rms_batch(a_correction),
        ledger_relative_error=rel_err,
        prediction_mse=jnp.mean((safe_phi - target_t) ** 2, axis=(1, 2, 3)),
        field_variance=jnp.var(safe_phi, axis=(1, 2, 3)),
        coherence=1.0 / (1.0 + disagreement),
        active_fraction=jnp.mean(next_allocation, axis=(1, 2, 3)),
    )
    return next_state, ledger


if jax is not None:
    @partial(jax.jit, static_argnames=("shape",))
    def run_hrm(initial_state: State, inputs: Inputs, params: Params, dyn: Dynamics, shape: ShapeConfig, ablation: Ablation = FULL) -> tuple[State, StepLedger]:
        scan_inputs = (inputs.drive, inputs.target, inputs.perturb)

        def body(state: State, inp: tuple[jax.Array, jax.Array, jax.Array]):
            return hrm_step(state, inp, params, dyn, shape, ablation)

        return lax.scan(body, initial_state, scan_inputs)
else:
    def run_hrm(*args, **kwargs):
        raise ImportError("JAX is required to execute the HRM recurrent model. Install jax to run baseline functions.")


def execute_config_with_params(phase: str, candidate: str, cfg: ExperimentConfig, params: Params, seed: int = 0, keep_artifact: bool = False) -> dict[str, float | int | str | bool]:
    cfg.validate()
    initial, inputs = seeded_state_inputs(seed, cfg, perturb_strength=cfg.perturb_strength)
    t0 = time.perf_counter()
    final_state, ledger = run_hrm(initial, inputs, params, cfg.dynamics(), cfg.shape, FULL)
    block_until_ready((final_state, ledger))
    validate_state(final_state, cfg)
    validate_ledger(ledger, cfg)
    elapsed = time.perf_counter() - t0
    metrics = score_ledger(ledger, final_state, cfg)

    record = {
        "phase": phase,
        "candidate": candidate,
        "run_id": cfg.run_id(seed=seed, tag=phase.lower().replace(" ", "_")),
        "config_hash": cfg.config_hash(),
        "seed": seed,
        "backend": BACKEND,
        "profile": RUN_PROFILE,
        "steps": cfg.steps,
        "batch": cfg.shape.batch,
        "perturb_step": cfg.perturb_step,
        "perturb_strength": cfg.perturb_strength,
        "elapsed_seconds": elapsed,
        **metrics,
        **{f"mechanism_{k}": v for k, v in mechanism_means(ledger).items()},
    }
    if keep_artifact:
        RUN_ARTIFACTS[(phase, candidate, seed)] = (cfg, final_state, ledger)
    return record


class SimpleRecurrentParams(NamedTuple):
    W_input: jax.Array
    W_hidden: jax.Array
    W_output: jax.Array


def make_simple_recurrent_params(key: jax.Array, shape: ShapeConfig) -> SimpleRecurrentParams:
    k1, k2, k3 = random.split(key, 3)
    return SimpleRecurrentParams(
        W_input=random.normal(k1, (shape.channels, shape.cognitive_dim), dtype=jnp.float32) / jnp.sqrt(shape.channels),
        W_hidden=random.normal(k2, (shape.cognitive_dim, shape.cognitive_dim), dtype=jnp.float32) / jnp.sqrt(shape.cognitive_dim),
        W_output=random.normal(k3, (shape.cognitive_dim, shape.channels), dtype=jnp.float32) / jnp.sqrt(shape.cognitive_dim),
    )


def run_simple_recurrent_baseline(hidden: jax.Array, inputs: Inputs, params: SimpleRecurrentParams, cfg: ExperimentConfig) -> jax.Array:
    def baseline_step(hidden_state: jax.Array, inp: tuple[jax.Array, jax.Array, jax.Array]):
        drive_t, target_t, _ = inp
        drive_embed = jnp.mean(drive_t, axis=(1, 2, 3))
        hidden_state = jnp.tanh(jnp.einsum("bd,df->bf", drive_embed, params.W_input) + hidden_state @ params.W_hidden)
        output = hidden_state @ params.W_output
        prediction = jnp.tile(output[:, None, None, :], (1, cfg.shape.height, cfg.shape.width, 1))
        mse = jnp.mean((prediction - target_t) ** 2, axis=(1, 2, 3))
        return hidden_state, mse

    _, mse_history = lax.scan(baseline_step, hidden, (inputs.drive, inputs.target, inputs.perturb))
    return jnp.mean(mse_history)


def evaluate_non_hrm_baseline(cfg: ExperimentConfig, seed: int = 0) -> float:
    key = random.PRNGKey(seed)
    hidden_key, param_key = random.split(key)
    initial_hidden = 0.01 * random.normal(hidden_key, (cfg.shape.batch, cfg.shape.cognitive_dim), dtype=jnp.float32)
    baseline_params = make_simple_recurrent_params(param_key, cfg.shape)
    _, inputs = seeded_state_inputs(seed, cfg, perturb_strength=cfg.perturb_strength)
    return float(np.asarray(run_simple_recurrent_baseline(initial_hidden, inputs, baseline_params, cfg)))


def block_until_ready(tree):
    return jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree)


def mechanism_means(ledger: StepLedger) -> dict[str, float]:
    names = ("input", "diffusion", "reaction", "memory", "cognition", "guidance", "hierarchy", "perturb", "correction")
    arrays = (ledger.input_norm, ledger.diffusion_norm, ledger.reaction_norm, ledger.memory_norm, ledger.cognition_norm, ledger.guidance_norm, ledger.hierarchy_norm, ledger.perturb_norm, ledger.correction_norm)
    return {name: float(np.asarray(jnp.mean(arr))) for name, arr in zip(names, arrays)}


def score_ledger(ledger: StepLedger, final_state: State, cfg: ExperimentConfig) -> dict[str, float | int | bool]:
    mse_curve = np.asarray(jnp.mean(ledger.prediction_mse, axis=1), dtype=np.float64)
    t_steps = len(mse_curve)
    p = int(np.clip(cfg.perturb_step, 0, t_steps - 1))
    window = max(2, min(16, t_steps // 8))
    pre_start = max(0, p - window)
    pre = float(np.mean(mse_curve[pre_start:p])) if p > pre_start else float(mse_curve[p])

    if cfg.perturb_strength <= 0.0:
        peak = pre
        threshold = pre
        did_recover = True
        recovery_time = 0
        amplitude_reduction = 0.0
        l_recovery = 0.0
    else:
        response_end = min(t_steps, p + window)
        peak = float(np.max(mse_curve[p:response_end]))
        peak_delta = max(peak - pre, 1e-12)
        threshold = pre + cfg.recovery_threshold_fraction * peak_delta
        after = mse_curve[p:]
        recovered_indices = np.flatnonzero(after <= threshold)
        did_recover = bool(recovered_indices.size)
        recovery_time = int(recovered_indices[0]) if did_recover else int(len(after))
        final_level = float(np.mean(mse_curve[-window:]))
        amplitude_reduction = float(np.clip((peak - final_level) / peak_delta, 0.0, 1.0))
        recovery_fraction = recovery_time / max(len(after) - 1, 1)
        l_recovery = recovery_fraction + (1.0 - amplitude_reduction)
        if not did_recover:
            l_recovery += 1.0

    max_ledger_error = float(np.asarray(jnp.max(ledger.ledger_relative_error)))
    mean_correction = float(np.asarray(jnp.mean(ledger.correction_norm)))
    final_phi = np.asarray(final_state.phi)
    finite = bool(np.isfinite(final_phi).all() and np.isfinite(mse_curve).all())
    max_abs_phi = float(np.max(np.abs(final_phi)))
    bounded_pass = max_abs_phi <= cfg.field_bound + 1e-4
    ledger_pass = max_ledger_error <= cfg.ledger_tolerance

    l_task = float(np.mean(mse_curve))
    l_safety = mean_correction + 10.0 * max(max_ledger_error - cfg.ledger_tolerance, 0.0)
    if not finite:
        l_safety += 1000.0
    if not bounded_pass:
        l_safety += 100.0
    l_total = l_task + cfg.recovery_weight * l_recovery + cfg.safety_weight * l_safety

    return {
        "L_total": l_total,
        "L_task": l_task,
        "L_recovery": float(l_recovery),
        "L_safety": float(l_safety),
        "did_recover": did_recover,
        "recovery_time": recovery_time,
        "recovery_amplitude_reduction": amplitude_reduction,
        "pre_mse": pre,
        "peak_mse": peak,
        "recovery_threshold": threshold,
        "final_prediction_mse": float(mse_curve[-1]),
        "max_ledger_relative_error": max_ledger_error,
        "mean_coherence": float(np.asarray(jnp.mean(ledger.coherence))),
        "min_field_variance": float(np.asarray(jnp.min(ledger.field_variance))),
        "mean_active_fraction": float(np.asarray(jnp.mean(ledger.active_fraction))),
        "mean_correction_norm": mean_correction,
        "max_abs_phi": max_abs_phi,
        "finite": finite,
        "bounded_pass": bounded_pass,
        "ledger_pass": ledger_pass,
    }


def reset_run_history() -> None:
    ALL_RECORDS.clear()
    RUN_ARTIFACTS.clear()


def execute_config(phase: str, candidate: str, cfg: ExperimentConfig, seed: int = 0, ablation: Ablation = FULL, keep_artifact: bool = False) -> dict[str, float | int | str | bool]:
    cfg.validate()
    params, initial, inputs = seeded_components(seed, cfg)
    t0 = time.perf_counter()
    final_state, ledger = run_hrm(initial, inputs, params, cfg.dynamics(), cfg.shape, ablation)
    block_until_ready((final_state, ledger))
    elapsed = time.perf_counter() - t0
    metrics = score_ledger(ledger, final_state, cfg)

    record = {
        "phase": phase,
        "candidate": candidate,
        "run_id": cfg.run_id(seed=seed, tag=phase.lower().replace(" ", "_")),
        "config_hash": cfg.config_hash(),
        "seed": seed,
        "backend": BACKEND,
        "profile": RUN_PROFILE,
        "steps": cfg.steps,
        "batch": cfg.shape.batch,
        "perturb_step": cfg.perturb_step,
        "perturb_strength": cfg.perturb_strength,
        "elapsed_seconds": elapsed,
        **metrics,
        **{f"mechanism_{k}": v for k, v in mechanism_means(ledger).items()},
    }
    ALL_RECORDS.append(record)
    if keep_artifact:
        RUN_ARTIFACTS[(phase, candidate, seed)] = (cfg, final_state, ledger)
    return record


def baseline_training_loss(ledger: StepLedger) -> jax.Array:
    return jnp.mean(ledger.prediction_mse) + 0.1 * jnp.mean(ledger.ledger_relative_error)


def train_baseline_pipeline(
    seed: int = 0,
    epochs: int = 3,
    learning_rate: float = 0.02,
    save_artifacts: bool = False,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, float | int | str | bool]:
    require_jax()
    reset_run_history()
    output_dir = ensure_output_dir(output_dir)

    cfg = BASELINE_CONFIG
    params, _, _ = seeded_components(seed, cfg, perturb_strength=cfg.perturb_strength)
    train_seeds = [seed, seed + 1]
    validation_seeds = [seed + 10, seed + 11]

    def hrm_loss_for_params(params: Params, seed_value: int) -> jax.Array:
        initial, inputs = seeded_state_inputs(seed_value, cfg, perturb_strength=cfg.perturb_strength)
        _, ledger = run_hrm(initial, inputs, params, cfg.dynamics(), cfg.shape, FULL)
        return baseline_training_loss(ledger)

    def training_loss(params: Params) -> jax.Array:
        losses = jnp.stack([hrm_loss_for_params(params, s) for s in train_seeds])
        return jnp.mean(losses)

    def validation_loss(params: Params) -> float:
        losses = [float(np.asarray(hrm_loss_for_params(params, s))) for s in validation_seeds]
        return float(np.mean(losses))

    initial_validation_loss = validation_loss(params)
    baseline_comparison_loss = float(np.mean([evaluate_non_hrm_baseline(cfg, s) for s in validation_seeds]))

    grad_fn = jax.grad(training_loss)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        current_train_loss = float(np.asarray(training_loss(params)))
        grads = grad_fn(params)
        params = jax.tree_util.tree_map(lambda p, g: p - jnp.asarray(learning_rate, dtype=jnp.float32) * g, params, grads)
        current_validation_loss = validation_loss(params)
        history.append({
            "epoch": int(epoch),
            "training_loss": current_train_loss,
            "validation_loss": current_validation_loss,
        })

    final_validation_loss = validation_loss(params)
    validation_improvement = initial_validation_loss - final_validation_loss
    record = execute_config_with_params("Stage 1 baseline training", "trained_baseline", cfg, params, seed=seed, keep_artifact=True)
    if save_artifacts:
        save_records(output_dir)

    record["training_history"] = history
    record["validation_loss_before"] = float(initial_validation_loss)
    record["validation_loss_after"] = float(final_validation_loss)
    record["validation_improvement"] = float(validation_improvement)
    record["non_hrm_baseline_loss"] = float(baseline_comparison_loss)
    record["hrm_vs_non_hrm"] = float(final_validation_loss) < float(baseline_comparison_loss)
    record["training_epochs"] = epochs
    record["training_learning_rate"] = learning_rate
    return record


def run_sweep(phase: str, candidates: dict[str, ExperimentConfig], seed: int = 0, update_baseline: bool = True) -> tuple[ExperimentConfig | None, list[dict[str, float | int | str | bool]]]:
    print(f"\n=== {phase} ===")
    phase_records = []
    for name, cfg in candidates.items():
        record = execute_config(phase, name, cfg, seed=seed)
        phase_records.append(record)
        print(f"{name:28s} L_total={record['L_total']:.6f} task={record['L_task']:.6f} recovery={record['L_recovery']:.4f} ledger={'PASS' if record['ledger_pass'] else 'FAIL'} finite={record['finite']}")
    eligible = [r for r in phase_records if r["finite"] and r["ledger_pass"] and r["bounded_pass"]]
    pool = eligible if eligible else phase_records
    best = min(pool, key=lambda r: r["L_total"])
    label = "Selected best" if update_baseline else "Lowest measured loss (evaluation only)"
    print(label + ":", best["candidate"], "L_total=", f"{best['L_total']:.6f}")
    best_cfg = candidates[best["candidate"]]
    return (best_cfg if update_baseline else None), phase_records


def summarize_records(records: list[dict[str, float | int | str | bool]], label: str) -> dict[str, float | int | str]:
    fields = (
        "L_total",
        "L_task",
        "L_recovery",
        "recovery_time",
        "recovery_amplitude_reduction",
        "max_ledger_relative_error",
        "mean_coherence",
        "mean_active_fraction",
    )
    summary = {"label": label, "count": len(records)}
    for field in fields:
        values = np.asarray([r[field] for r in records], dtype=np.float64)
        summary[f"{field}_mean"] = float(np.mean(values))
        summary[f"{field}_std"] = float(np.std(values))
    summary["recovery_rate"] = float(np.mean([r["did_recover"] for r in records]))
    summary["ledger_pass_rate"] = float(np.mean([r["ledger_pass"] for r in records]))
    summary["bounded_pass_rate"] = float(np.mean([r["bounded_pass"] for r in records]))
    return summary


def save_records(output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path, Path]:
    output_dir = ensure_output_dir(output_dir)
    json_path = output_dir / "hrm_tpu_baseline_summary.json"
    csv_path = output_dir / "hrm_tpu_baseline_records.csv"
    config_path = output_dir / "hrm_tpu_selected_config.json"

    final_cfg = BASELINE_CONFIG
    summary = {
        "schema_version": "hrm-tpu-baseline-v2",
        "backend": BACKEND,
        "run_profile": RUN_PROFILE,
        "jax_version": jax.__version__,
        "final_config": final_cfg.as_dict(),
        "final_config_hash": final_cfg.config_hash(),
        "record_count": len(ALL_RECORDS),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(final_cfg.as_dict(), indent=2), encoding="utf-8")

    fieldnames = sorted({key for record in ALL_RECORDS for key in record.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ALL_RECORDS)

    return json_path, csv_path, config_path


def run_baseline_pipeline(seed: int = 0, save_artifacts: bool = True, output_dir: Path = OUTPUT_DIR) -> dict[str, float | int | str]:
    require_jax()
    reset_run_history()
    ensure_output_dir(output_dir)
    record = execute_config("Stage 1 baseline", "active_baseline", BASELINE_CONFIG, seed=seed, keep_artifact=True)
    if save_artifacts:
        save_records(output_dir)
    return record


if __name__ == "__main__":
    run_baseline_pipeline()
