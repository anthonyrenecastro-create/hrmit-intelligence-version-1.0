from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FieldState:
    phi: np.ndarray


@dataclass(frozen=True)
class GeometryState:
    laplacian: np.ndarray
    metric_scale: float = 1.0


@dataclass(frozen=True)
class TopologyState:
    node_count: int
    edge_count: int
    version: int = 0


@dataclass(frozen=True)
class MemoryState:
    working: np.ndarray
    associative_keys: np.ndarray
    associative_values: np.ndarray
    capacity: int
    write_index: int = 0


@dataclass(frozen=True)
class CognitionState:
    latent: np.ndarray
    prediction: np.ndarray
    residual: np.ndarray
    uncertainty: np.ndarray


@dataclass(frozen=True)
class HierarchyState:
    coarse: np.ndarray
    restriction: np.ndarray
    prolongation: np.ndarray
    gain: float = 0.1


@dataclass(frozen=True)
class BudgetState:
    total_budget: float
    remaining_budget: float
    active_width: int
    event_allowance: int
    cumulative_cost: float = 0.0


@dataclass(frozen=True)
class HRMState:
    version: int
    step: int
    dtype: str
    device: str
    rng_state: dict[str, Any]
    phi: FieldState
    geometry: GeometryState
    topology: TopologyState
    memory: MemoryState
    cognition: CognitionState
    hierarchy: HierarchyState
    budget: BudgetState


def _to_np(value: np.ndarray | list[float], dtype: str = "float32") -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    return arr


def make_initial_state(
    *,
    node_count: int,
    channels: int,
    latent_dim: int,
    memory_capacity: int,
    seed: int,
    dtype: str = "float32",
    total_budget: float = 1000.0,
) -> HRMState:
    rng = np.random.default_rng(seed)
    phi = 0.05 * rng.normal(size=(node_count, channels)).astype(dtype)
    memory = np.zeros((node_count, channels), dtype=dtype)
    keys = np.zeros((memory_capacity, channels), dtype=dtype)
    values = np.zeros((memory_capacity, channels), dtype=dtype)
    latent = 0.01 * rng.normal(size=(latent_dim,)).astype(dtype)
    prediction = np.zeros((channels,), dtype=dtype)
    residual = np.zeros((channels,), dtype=dtype)
    uncertainty = np.ones((channels,), dtype=dtype)

    laplacian = np.eye(node_count, dtype=dtype) * 2.0
    off_diag = np.eye(node_count, k=1, dtype=dtype) + np.eye(node_count, k=-1, dtype=dtype)
    laplacian = laplacian - off_diag
    laplacian[0, -1] = -1.0
    laplacian[-1, 0] = -1.0

    coarse_count = max(1, node_count // 2)
    restriction = np.zeros((coarse_count, node_count), dtype=dtype)
    for i in range(coarse_count):
        left = 2 * i
        right = min(node_count - 1, left + 1)
        restriction[i, left] = 0.5
        restriction[i, right] = 0.5
    prolongation = restriction.T * 2.0
    coarse = restriction @ phi

    return HRMState(
        version=0,
        step=0,
        dtype=dtype,
        device="cpu",
        rng_state=rng.bit_generator.state,
        phi=FieldState(phi=phi),
        geometry=GeometryState(laplacian=laplacian, metric_scale=1.0),
        topology=TopologyState(node_count=node_count, edge_count=node_count, version=0),
        memory=MemoryState(
            working=memory,
            associative_keys=keys,
            associative_values=values,
            capacity=memory_capacity,
            write_index=0,
        ),
        cognition=CognitionState(
            latent=latent,
            prediction=prediction,
            residual=residual,
            uncertainty=uncertainty,
        ),
        hierarchy=HierarchyState(
            coarse=coarse,
            restriction=restriction,
            prolongation=prolongation,
            gain=0.1,
        ),
        budget=BudgetState(
            total_budget=total_budget,
            remaining_budget=total_budget,
            active_width=channels,
            event_allowance=max(1, channels // 2),
            cumulative_cost=0.0,
        ),
    )


def state_to_dict(state: HRMState) -> dict[str, Any]:
    return {
        "version": state.version,
        "step": state.step,
        "dtype": state.dtype,
        "device": state.device,
        "rng_state": state.rng_state,
        "phi": {"phi": state.phi.phi.tolist()},
        "geometry": {
            "laplacian": state.geometry.laplacian.tolist(),
            "metric_scale": state.geometry.metric_scale,
        },
        "topology": {
            "node_count": state.topology.node_count,
            "edge_count": state.topology.edge_count,
            "version": state.topology.version,
        },
        "memory": {
            "working": state.memory.working.tolist(),
            "associative_keys": state.memory.associative_keys.tolist(),
            "associative_values": state.memory.associative_values.tolist(),
            "capacity": state.memory.capacity,
            "write_index": state.memory.write_index,
        },
        "cognition": {
            "latent": state.cognition.latent.tolist(),
            "prediction": state.cognition.prediction.tolist(),
            "residual": state.cognition.residual.tolist(),
            "uncertainty": state.cognition.uncertainty.tolist(),
        },
        "hierarchy": {
            "coarse": state.hierarchy.coarse.tolist(),
            "restriction": state.hierarchy.restriction.tolist(),
            "prolongation": state.hierarchy.prolongation.tolist(),
            "gain": state.hierarchy.gain,
        },
        "budget": {
            "total_budget": state.budget.total_budget,
            "remaining_budget": state.budget.remaining_budget,
            "active_width": state.budget.active_width,
            "event_allowance": state.budget.event_allowance,
            "cumulative_cost": state.budget.cumulative_cost,
        },
    }


def state_from_dict(payload: dict[str, Any]) -> HRMState:
    dtype = payload["dtype"]
    return HRMState(
        version=int(payload["version"]),
        step=int(payload["step"]),
        dtype=dtype,
        device=str(payload["device"]),
        rng_state=dict(payload["rng_state"]),
        phi=FieldState(phi=_to_np(payload["phi"]["phi"], dtype=dtype)),
        geometry=GeometryState(
            laplacian=_to_np(payload["geometry"]["laplacian"], dtype=dtype),
            metric_scale=float(payload["geometry"]["metric_scale"]),
        ),
        topology=TopologyState(
            node_count=int(payload["topology"]["node_count"]),
            edge_count=int(payload["topology"]["edge_count"]),
            version=int(payload["topology"]["version"]),
        ),
        memory=MemoryState(
            working=_to_np(payload["memory"]["working"], dtype=dtype),
            associative_keys=_to_np(payload["memory"]["associative_keys"], dtype=dtype),
            associative_values=_to_np(payload["memory"]["associative_values"], dtype=dtype),
            capacity=int(payload["memory"]["capacity"]),
            write_index=int(payload["memory"]["write_index"]),
        ),
        cognition=CognitionState(
            latent=_to_np(payload["cognition"]["latent"], dtype=dtype),
            prediction=_to_np(payload["cognition"]["prediction"], dtype=dtype),
            residual=_to_np(payload["cognition"]["residual"], dtype=dtype),
            uncertainty=_to_np(payload["cognition"]["uncertainty"], dtype=dtype),
        ),
        hierarchy=HierarchyState(
            coarse=_to_np(payload["hierarchy"]["coarse"], dtype=dtype),
            restriction=_to_np(payload["hierarchy"]["restriction"], dtype=dtype),
            prolongation=_to_np(payload["hierarchy"]["prolongation"], dtype=dtype),
            gain=float(payload["hierarchy"]["gain"]),
        ),
        budget=BudgetState(
            total_budget=float(payload["budget"]["total_budget"]),
            remaining_budget=float(payload["budget"]["remaining_budget"]),
            active_width=int(payload["budget"]["active_width"]),
            event_allowance=int(payload["budget"]["event_allowance"]),
            cumulative_cost=float(payload["budget"]["cumulative_cost"]),
        ),
    )


def state_digest(state: HRMState) -> str:
    data = json.dumps(state_to_dict(state), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
