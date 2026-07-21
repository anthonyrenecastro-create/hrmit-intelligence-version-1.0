from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from hrm.hrm_core import (
    HRMTransitionConfig,
    MechanismAblations,
    build_engine,
    freeze_state,
    make_initial_state,
    state_digest,
    state_from_dict,
    state_to_dict,
    topology_enabled_config,
)
from hrm.hrm_core.invariants import validate_state
from hrm.hrm_core.mechanisms.base import HRMInput
from hrm.integration import IntegratedRuntime, MockDeterministicProvider, OperatingMode
from hrm.integration.providers import InferenceProvider
from hrm.integration.types import InferenceContext, InferenceRequest, InferenceResult


@dataclass(frozen=True)
class ValidationConfig:
    request_text: str
    seed: int
    node_count: int
    channels: int
    latent_dim: int
    memory_capacity: int
    mode: str
    preferred_provider: str
    session_id: str
    ledger_tolerance: float


class FailingProvider:
    provider_id = "mock_timeout"

    async def infer(self, request: InferenceRequest, context: InferenceContext) -> InferenceResult:
        return InferenceResult(
            raw_output="",
            parsed_output=None,
            prompt_tokens=0,
            completion_tokens=0,
            latency_seconds=0.0,
            model_id=request.model_id,
            provider_id=self.provider_id,
            sampling={"temperature": request.temperature},
            seed=request.seed,
            parse_status="error",
            safety_status="unknown",
            retry_history=("simulated_failure",),
            provenance={"simulated": True, "mode": context.mode.value},
            error_state="simulated_failure",
        )


def parse_simple_yaml(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            payload[key] = ""
            continue
        lowered = value.lower()
        if lowered in {"true", "false"}:
            payload[key] = lowered == "true"
            continue
        if lowered in {"null", "none"}:
            payload[key] = None
            continue
        try:
            payload[key] = json.loads(value)
            continue
        except Exception:
            pass
        try:
            if "." in value or "e" in lowered:
                payload[key] = float(value)
            else:
                payload[key] = int(value)
            continue
        except ValueError:
            payload[key] = value
    return payload


def load_config(path: Path | None) -> ValidationConfig:
    if path is None:
        data = {
            "request_text": "Provide a safe deterministic summary.",
            "seed": 7,
            "node_count": 16,
            "channels": 6,
            "latent_dim": 12,
            "memory_capacity": 16,
            "mode": "sovereign_local",
            "preferred_provider": "mock_deterministic",
            "session_id": "release-validation",
            "ledger_tolerance": 1e-5,
        }
    else:
        data = parse_simple_yaml(path)
    return ValidationConfig(
        request_text=str(data.get("request_text", "Provide a safe deterministic summary.")),
        seed=int(data.get("seed", 7)),
        node_count=int(data.get("node_count", 16)),
        channels=int(data.get("channels", 6)),
        latent_dim=int(data.get("latent_dim", 12)),
        memory_capacity=int(data.get("memory_capacity", 16)),
        mode=str(data.get("mode", "sovereign_local")),
        preferred_provider=str(data.get("preferred_provider", "mock_deterministic")),
        session_id=str(data.get("session_id", "release-validation")),
        ledger_tolerance=float(data.get("ledger_tolerance", 1e-5)),
    )


def _git(*args: str) -> str:
    try:
        result = subprocess.check_output(["git", *args], cwd=Path(__file__).resolve().parent.parent, text=True)
    except Exception:
        return "unknown"
    return result.strip()


def _dependency_versions() -> dict[str, str]:
    names = ["numpy", "pytest", "httpx", "Pillow"]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except Exception:
            versions[name] = "unavailable"
    return versions


def _environment_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git_commit": _git("rev-parse", "HEAD"),
        "dirty_tree": bool(_git("status", "--porcelain")),
        "dependencies": _dependency_versions(),
    }


def _text_drive(user_text: str, node_count: int, channels: int) -> np.ndarray:
    from hrm.memory import _embed_text

    projection = _embed_text(user_text, dim=channels).astype(np.float32)
    return np.repeat(projection[None, :], node_count, axis=0)


def _make_runtime(config: ValidationConfig, checkpoint_dir: Path) -> IntegratedRuntime:
    runtime = IntegratedRuntime(
        mode=OperatingMode(config.mode),
        preferred_provider=config.preferred_provider,
        session_id=config.session_id,
        checkpoint_dir=checkpoint_dir,
    )
    runtime.providers.setdefault("mock_deterministic", MockDeterministicProvider())
    return runtime


def _run_governed_text(config: ValidationConfig, artifact_dir: Path) -> dict[str, Any]:
    runtime = _make_runtime(config, artifact_dir / "checkpoints")
    initial_state = make_initial_state(
        node_count=config.node_count,
        channels=config.channels,
        latent_dim=config.latent_dim,
        memory_capacity=config.memory_capacity,
        seed=config.seed,
    )
    runtime.state = state_from_dict(state_to_dict(initial_state))
    initial_hash = state_digest(runtime.state)
    result_error: dict[str, Any] | None = None
    try:
        result = runtime.process_text_request(config.request_text)
    except Exception as error:
        result_error = {"error": type(error).__name__, "message": str(error)}
        result = None
    final_hash = state_digest(runtime.state)

    replay_runtime = _make_runtime(config, artifact_dir / "replay_checkpoints")
    replay_runtime.state = state_from_dict(state_to_dict(initial_state))
    replay_error: dict[str, Any] | None = None
    try:
        replay_result = replay_runtime.process_text_request(config.request_text)
        replay_hash = state_digest(replay_runtime.state)
    except Exception as error:
        replay_error = {"error": type(error).__name__, "message": str(error)}
        replay_result = None
        replay_hash = ""

    drive = _text_drive(config.request_text, config.node_count, config.channels)
    full_topology_config = topology_enabled_config()
    base_engine = build_engine(full_topology_config)
    ablated_engine = build_engine(
        topology_enabled_config(ablations=MechanismAblations(diffusion=False))
    )
    ablation_error: dict[str, Any] | None = None
    try:
        base_transition = base_engine.step(initial_state, HRMInput(field_drive=drive, metadata={"memory_query": "validation"}))
        ablated_transition = ablated_engine.step(state_from_dict(state_to_dict(initial_state)), HRMInput(field_drive=drive, metadata={"memory_query": "validation"}))
        divergence = float(np.linalg.norm(base_transition.state.phi.phi - ablated_transition.state.phi.phi))
        base_activation = float(base_transition.ledger.proposal_activations.get("diffusion", 0.0))
        ablated_activation = float(ablated_transition.ledger.proposal_activations.get("diffusion", 0.0))
    except Exception as error:
        ablation_error = {"error": type(error).__name__, "message": str(error)}
        divergence = 0.0
        base_activation = 0.0
        ablated_activation = 0.0

    topology_probe_error: dict[str, Any] | None = None
    topology_probe: dict[str, Any]
    try:
        topology_step = base_engine.step(
            state_from_dict(state_to_dict(initial_state)),
            HRMInput(
                field_drive=drive,
                metadata={
                    "memory_query": "validation_topology",
                    "topology_add_nodes": 1,
                    "topology_rewire": True,
                },
            ),
        )
        topology_phi_transport = next(
            (record for record in topology_step.ledger.transport_records if record.block == "Phi"),
            None,
        )
        topology_probe = {
            "before_nodes": initial_state.topology.node_count,
            "after_nodes": topology_step.state.topology.node_count,
            "before_topology_version": initial_state.topology.version,
            "after_topology_version": topology_step.state.topology.version,
            "accepted_topology_events": [
                event.event_type
                for event in topology_step.ledger.accepted_events
                if event.block == "T"
            ],
            "geometry_shape_after": list(topology_step.state.geometry.laplacian.shape),
            "ledger_max_rel_error": float(topology_step.ledger.max_rel_reconstruction_error),
            "phi_transport_after_shape": list(topology_phi_transport.after_shape) if topology_phi_transport else None,
            "phi_transport_new_to_old": list(topology_phi_transport.new_to_old_index_map) if topology_phi_transport else None,
            "full_arm_mechanisms": [mech.mechanism_id for mech in base_engine.mechanisms],
            "ablated_arm_mechanisms": [mech.mechanism_id for mech in ablated_engine.mechanisms],
        }
    except Exception as error:
        topology_probe_error = {"error": type(error).__name__, "message": str(error)}
        topology_probe = {
            "before_nodes": initial_state.topology.node_count,
            "after_nodes": initial_state.topology.node_count,
            "before_topology_version": initial_state.topology.version,
            "after_topology_version": initial_state.topology.version,
            "accepted_topology_events": [],
            "geometry_shape_after": list(initial_state.geometry.laplacian.shape),
            "ledger_max_rel_error": float("inf"),
            "phi_transport_after_shape": None,
            "phi_transport_new_to_old": None,
            "full_arm_mechanisms": [mech.mechanism_id for mech in base_engine.mechanisms],
            "ablated_arm_mechanisms": [mech.mechanism_id for mech in ablated_engine.mechanisms],
        }

    failing_runtime = _make_runtime(config, artifact_dir / "failure_checkpoints")
    failing_runtime.providers["mock_timeout"] = FailingProvider()
    failing_runtime.preferred_provider = "mock_timeout"
    failing_runtime.state = state_from_dict(state_to_dict(initial_state))
    failure_error: dict[str, Any] | None = None
    try:
        failing_result = failing_runtime.process_text_request(config.request_text)
    except Exception as error:
        failure_error = {"error": type(error).__name__, "message": str(error)}
        failing_result = None

    sovereign_runtime = _make_runtime(config, artifact_dir / "sovereign_checkpoints")
    sovereign_runtime.state = state_from_dict(state_to_dict(initial_state))
    sovereign_error: dict[str, Any] | None = None
    try:
        sovereign_result = sovereign_runtime.process_text_request(config.request_text)
    except Exception as error:
        sovereign_error = {"error": type(error).__name__, "message": str(error)}
        sovereign_result = None

    invalid_state = state_from_dict(state_to_dict(initial_state))
    invalid_state.phi.phi[0, 0] = np.nan
    invalid_state_result = None
    invalid_state_detected = False
    try:
        invalid_state_result = base_engine.step(invalid_state, HRMInput(field_drive=drive, metadata={"memory_query": "validation"}))
    except Exception as error:
        invalid_state_detected = True
        invalid_state_result = {"error": type(error).__name__, "message": str(error)}

    return {
        "run_kind": "governed_text_slice",
        "config": asdict(config),
        "initial_state_hash": initial_hash,
        "final_state_hash": final_hash,
        "state_version_before": getattr(result, "state_version_before", None),
        "state_version_after": getattr(result, "state_version_after", None),
        "ledger_max_rel_error": getattr(result, "ledger_max_rel_error", None),
        "ledger_tolerance": config.ledger_tolerance,
        "replay": {
            "final_state_hash": replay_hash,
            "matches": replay_result is not None and replay_hash == final_hash,
            "response_text_match": bool(replay_result is not None and result is not None and replay_result.response_text == result.response_text),
            "error": replay_error,
        },
        "activations": dict(sorted(getattr(result.response_envelope, "__dict__", {}).items())) if result else {},
        "mechanism_activations": getattr(result, "observability", {}).get("mechanism_activations", {}) if result else {},
        "observability": getattr(result, "observability", {}),
        "provider_used": getattr(result, "provider_used", None),
        "response_text": getattr(result, "response_text", ""),
        "ablation": {
            "diffusion_disabled_divergence": divergence,
            "base_activation": base_activation,
            "ablated_activation": ablated_activation,
            "error": ablation_error,
        },
        "topology_probe": {
            **topology_probe,
            "error": topology_probe_error,
        },
        "failure_injection": {
            "model_failure_response": getattr(failing_result, "response_text", ""),
            "model_failure_provider": getattr(failing_result, "provider_used", None),
            "model_failure_state_version_after": getattr(failing_result, "state_version_after", None),
            "model_failure_error": failure_error,
            "state_failure_detected": invalid_state_detected,
            "state_failure_result": invalid_state_result,
        },
        "sovereignty": {
            "mode": config.mode,
            "provider_used": getattr(sovereign_result, "provider_used", None),
            "remote_fallback_used": getattr(sovereign_result, "provider_used", None) not in {None, "mock_deterministic"},
            "error": sovereign_error,
        },
    }


def _write_report_files(run_dir: Path, report: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    final_json = run_dir.parent.parent / "FINAL_VALIDATION_REPORT.json"
    final_md = run_dir.parent.parent / "FINAL_VALIDATION_REPORT.md"
    final_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    final_md.write_text(
        "# Final Validation Report\n\n"
        f"- run_id: {report['run_id']}\n"
        f"- evidence_level: {report['evidence_level']}\n"
        f"- ledger_ok: {report['checks']['ledger_ok']}\n"
        f"- replay_ok: {report['checks']['replay_ok']}\n"
        f"- activation_ok: {report['checks']['activation_ok']}\n"
        f"- divergence_ok: {report['checks']['divergence_ok']}\n"
        f"- model_failure_safe: {report['checks']['model_failure_safe']}\n"
        f"- state_failure_safe: {report['checks']['state_failure_safe']}\n"
        f"- sovereignty_ok: {report['checks']['sovereignty_ok']}\n",
        encoding="utf-8",
    )


def run_validation(config: ValidationConfig) -> dict[str, Any]:
    run_id = time.strftime("run-%Y%m%d-%H%M%S")
    config_digest = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode("utf-8")).hexdigest()
    artifact_root = Path("validation") / "artifacts" / run_id
    report = _run_governed_text(config, artifact_root)
    report.update(
        {
            "run_id": run_id,
            "config_digest": config_digest,
            "environment": _environment_snapshot(),
            "artifact_root": str(artifact_root),
            "evidence_level": 2,
            "checks": {
                "ledger_ok": report.get("ledger_max_rel_error") is not None and report["ledger_max_rel_error"] <= config.ledger_tolerance,
                "replay_ok": report["replay"]["matches"],
                "activation_ok": report["ablation"]["base_activation"] > 0.0,
                "divergence_ok": report["ablation"]["diffusion_disabled_divergence"] > 1e-6,
                "ablation_integrity_ok": (
                    "topology" in report["topology_probe"]["full_arm_mechanisms"]
                    and "topology" not in report["topology_probe"]["ablated_arm_mechanisms"]
                ),
                "G1_topology_mutation_ok": (
                    report["topology_probe"]["after_nodes"] > report["topology_probe"]["before_nodes"]
                    and report["topology_probe"]["after_topology_version"] > report["topology_probe"]["before_topology_version"]
                    and bool(report["topology_probe"]["accepted_topology_events"])
                ),
                "G2_transport_map_ok": (
                    report["topology_probe"]["phi_transport_after_shape"] is not None
                    and report["topology_probe"]["phi_transport_after_shape"][0] == report["topology_probe"]["after_nodes"]
                    and report["topology_probe"]["phi_transport_new_to_old"] is not None
                    and any(index is None for index in report["topology_probe"]["phi_transport_new_to_old"])
                ),
                "structural_ledger_gate_ok": (
                    report["topology_probe"]["ledger_max_rel_error"] <= config.ledger_tolerance
                    and report["topology_probe"]["geometry_shape_after"][0] == report["topology_probe"]["after_nodes"]
                ),
                "model_failure_safe": bool(report["failure_injection"]["model_failure_response"]),
                "state_failure_safe": report["failure_injection"]["state_failure_detected"],
                "sovereignty_ok": report["sovereignty"]["provider_used"] == "mock_deterministic",
            },
        }
    )
    _write_report_files(artifact_root, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reproducible HRMIT validation.")
    parser.add_argument("--config", type=Path, default=Path("validation/configs/release_text_slice.yaml"), help="Validation config YAML file")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    report = run_validation(config)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
