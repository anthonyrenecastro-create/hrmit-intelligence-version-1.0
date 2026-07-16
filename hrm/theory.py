from __future__ import annotations

import json
import time

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hrm.baseline import baseline_training_loss, run_baseline_pipeline, train_baseline_pipeline, BASELINE_CONFIG, RUN_PROFILE
from hrm.distributed import DistributedCoordinator, RoleSpecialistModule, TaskGraph
from hrm.distributed.types import CognitiveTask as DistributedTask
from hrm.memory import LongTermMemory, MemoryQuery, MemorySystem, Planner, summarize_memory
from hrm.perception import PerceptionPipeline
from hrm.tools import APIConnector, AuditLogger, PermissionContext, PathPolicy, SelfVerifier, ToolExecutor, ToolRegistry
from hrm.learning import LearningSystem


@dataclass(frozen=True)
class HRMStage:
    index: int
    name: str
    summary: str
    steps: tuple[str, ...]


class HRMTheory:
    def __init__(self) -> None:
        self.stages = [
            HRMStage(
                1,
                "HRM reasoning core",
                "Framework integration of the baseline TPU HRM model with a staged intelligence pipeline.",
                (
                    "Define HRM configuration",
                    "Initialize core state and parameters",
                    "Execute the recurrent HRM reasoning loop",
                    "Collect baseline metrics",
                    "Persist baseline artifacts",
                ),
            ),
            HRMStage(
                2,
                "Long-term memory and planning",
                "Prepare a retrieval and planning layer on top of baseline HRM state.",
                (
                    "Create memory buffers",
                    "Construct retrieval indexes",
                    "Assemble knowledge graph stubs",
                    "Plan a reasoning trajectory",
                    "Export planning metadata",
                ),
            ),
            HRMStage(
                3,
                "Tool use and verification",
                "Infrastructure to connect the HRM core to external tools, code execution, and verification loops.",
                (
                    "Register available tool wrappers",
                    "Initialize code execution agent",
                    "Configure API connectors",
                    "Run a self-verification pass",
                    "Store verification outcomes",
                ),
            ),
            HRMStage(
                4,
                "Multimodal perception",
                "Foundation for text, image, audio, and video perception modules.",
                (
                    "Load modality adapters",
                    "Normalize input formats",
                    "Encode sensory tokens",
                    "Integrate modality state",
                    "Evaluate perception readiness",
                ),
            ),
            HRMStage(
                5,
                "Distributed cognition",
                "Coordinate multiple reasoning agents and merge insights across nodes.",
                (
                    "Instantiate agent proxies",
                    "Exchange reasoning traces",
                    "Merge belief states",
                    "Resolve consensus",
                    "Output distributed plan",
                ),
            ),
            HRMStage(
                6,
                "Learning systems",
                "Support continual controlled heuristic adaptation, preference optimization, and memory refinement.",
                (
                    "Capture adaptation signals",
                    "Update preference model",
                    "Refine long-term memory",
                    "Adjust reasoning parameters",
                    "Record adaptation metrics",
                ),
            ),
        ]

    def get_stage(self, index: int) -> HRMStage:
        if not 1 <= index <= len(self.stages):
            raise ValueError(f"Stage {index} is not defined")
        return self.stages[index - 1]

    def run_stage(self, index: int, **kwargs: Any) -> dict[str, Any]:
        stage = self.get_stage(index)
        if index == 1:
            result = self._run_stage_1(**kwargs)
        elif index == 2:
            result = self._run_stage_2(**kwargs)
        elif index == 3:
            result = self._run_stage_3(**kwargs)
        elif index == 4:
            result = self._run_stage_4(**kwargs)
        elif index == 5:
            result = self._run_stage_5(**kwargs)
        elif index == 6:
            result = self._run_stage_6(**kwargs)
        else:
            result = self._run_placeholder_stage(stage, **kwargs)
        return {"stage": stage.name, "summary": stage.summary, "steps": stage.steps, "result": result}

    def run_all(self) -> list[dict[str, Any]]:
        results = []
        for stage in self.stages:
            results.append(self.run_stage(stage.index))
        return results

    def _run_stage_1(
        self,
        seed: int = 0,
        output_dir: Path | str = "hrm_baseline_outputs",
        train: bool = False,
        train_epochs: int = 3,
        train_learning_rate: float = 0.02,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        runtime_message = ""
        try:
            if train:
                artifact = train_baseline_pipeline(
                    seed=seed,
                    epochs=train_epochs,
                    learning_rate=train_learning_rate,
                    save_artifacts=True,
                    output_dir=output_dir,
                )
                runtime_message = "Stage 1 executed the JAX baseline training pipeline."
            else:
                artifact = run_baseline_pipeline(seed=seed, save_artifacts=True, output_dir=output_dir)
                runtime_message = "Stage 1 executed the real JAX baseline pipeline."
        except Exception:
            artifact = {
                "phase": "Stage 1 placeholder",
                "candidate": "baseline_placeholder",
                "config_hash": BASELINE_CONFIG.config_hash(),
                "seed": seed,
                "backend": "cpu",
                "profile": RUN_PROFILE,
                "steps": BASELINE_CONFIG.steps,
                "batch": BASELINE_CONFIG.shape.batch,
                "perturb_step": BASELINE_CONFIG.perturb_step,
                "perturb_strength": BASELINE_CONFIG.perturb_strength,
                "L_total": 0.0,
                "did_recover": False,
                "ledger_pass": False,
                "bounded_pass": False,
            }
            runtime_message = "Stage 1 used the placeholder baseline path because JAX was unavailable."
        return {
            "phase": "Stage 1",
            "baseline_record": artifact,
            "output_dir": str(output_dir),
            "runtime_message": runtime_message,
        }

    def _run_stage_2(
        self,
        baseline_record: dict[str, Any] | None = None,
        plan_query: str = "Improve HRM safety and recovery",
        output_dir: Path | str = "hrm_baseline_outputs",
    ) -> dict[str, Any]:
        if baseline_record is None:
            try:
                baseline_record = run_baseline_pipeline(seed=0, save_artifacts=False)
            except Exception as error:
                baseline_record = {
                    "phase": "Stage 1 placeholder",
                    "candidate": "baseline_placeholder",
                    "config_hash": BASELINE_CONFIG.config_hash(),
                    "seed": 0,
                    "backend": "cpu",
                    "profile": RUN_PROFILE,
                    "steps": BASELINE_CONFIG.steps,
                    "batch": BASELINE_CONFIG.shape.batch,
                    "perturb_step": BASELINE_CONFIG.perturb_step,
                    "perturb_strength": BASELINE_CONFIG.perturb_strength,
                    "L_total": 0.0,
                    "did_recover": False,
                    "ledger_pass": False,
                    "bounded_pass": False,
                }
        output_dir = Path(output_dir)
        memory = MemorySystem()
        baseline_memory = LongTermMemory.from_baseline_record(baseline_record)
        for entry in baseline_memory.entries:
            memory.create_episode(
                key=entry.key,
                value=entry.content,
                sequence_step=0,
                source="stage2_baseline",
                importance=0.7,
                confidence=0.9,
                metadata=entry.metadata,
            )
        semantic_consolidation = memory.consolidate()
        memory_store_path = output_dir / "stage2_memory.json"
        memory.save(memory_store_path)
        reloaded_memory = MemorySystem.load(memory_store_path)
        retrieved_snapshot = reloaded_memory.retrieve(MemoryQuery(query=plan_query, limit=3), limit=3)
        planner = Planner(baseline_memory)
        plan = planner.create_plan(plan_query)
        memory_summary = {
            "working_count": len(reloaded_memory.working.items),
            "episodic_count": len(reloaded_memory.episodic.items),
            "semantic_count": len(reloaded_memory.semantic.records),
            "consolidations": len(reloaded_memory.semantic.consolidations),
            "sample_memory_keys": [item.key for item in baseline_memory.entries[:5]],
        }
        loaded_entries = len(reloaded_memory.working.items) + len(reloaded_memory.episodic.items) + len(reloaded_memory.semantic.records)
        return {
            "phase": "Stage 2",
            "memory_summary": memory_summary,
            "memory_store_path": str(memory_store_path),
            "memory_persistence": {
                "saved": True,
                "loaded_entries": loaded_entries,
                "loaded_working": len(reloaded_memory.working.items),
                "loaded_episodic": len(reloaded_memory.episodic.items),
                "loaded_semantic": len(reloaded_memory.semantic.records),
                "retrieved_snapshot": [result.__dict__ for result in retrieved_snapshot],
                "semantic_consolidation": semantic_consolidation.__dict__ if semantic_consolidation else None,
            },
            "plan": {
                "query": plan.query,
                "steps": plan.steps,
                "sources": plan.sources,
            },
        }

    def _run_stage_3(
        self,
        baseline_record: dict[str, Any] | None = None,
        api_endpoint: str = "status",
        api_payload: dict[str, Any] | None = None,
        output_dir: Path | str = "hrm_baseline_outputs",
    ) -> dict[str, Any]:
        if baseline_record is None:
            try:
                baseline_record = run_baseline_pipeline(seed=0, save_artifacts=False)
            except ImportError:
                baseline_record = {
                    "phase": "Stage 1 placeholder",
                    "candidate": "baseline_placeholder",
                    "config_hash": BASELINE_CONFIG.config_hash(),
                    "seed": 0,
                    "backend": "cpu",
                    "profile": RUN_PROFILE,
                    "steps": BASELINE_CONFIG.steps,
                    "batch": BASELINE_CONFIG.shape.batch,
                    "perturb_step": BASELINE_CONFIG.perturb_step,
                    "perturb_strength": BASELINE_CONFIG.perturb_strength,
                    "L_total": 0.0,
                    "did_recover": False,
                    "ledger_pass": False,
                    "bounded_pass": False,
                }

        registry = ToolRegistry()
        registry.register_builtin_tools()

        api_payload = api_payload or {"health": True}
        api_connector = APIConnector(base_url="https://api.placeholder.local")
        verifier = SelfVerifier(registry, api_connector)

        policy = PathPolicy(read_roots=[Path(output_dir).resolve(), Path.cwd().resolve()], write_roots=[Path(output_dir).resolve(), Path.cwd().resolve()])
        audit_log = AuditLogger(Path(output_dir) / "stage3_tool_audit.log")
        tool_executor = ToolExecutor(registry, policy, audit_log)
        permission_context = PermissionContext(
            principal_id="stage3_agent",
            granted_permissions=frozenset({"tool.read", "tool.write", "tool.execute", "tool.network", "tool.delete"}),
            allowed_directories=(Path(output_dir).resolve(), Path.cwd().resolve()),
            allowed_hosts=("localhost", "127.0.0.1", "api.placeholder.local"),
            session_id="stage3_session",
            expires_at=time.time() + 3600,
            valid=True,
        )

        api_request = {"endpoint": api_endpoint, "payload": api_payload}
        try:
            api_response = api_connector.call("POST", api_endpoint, headers={"Content-Type": "application/json"}, body=api_payload, permission_context=permission_context)
        except Exception as error:
            api_response = {
                "connector": api_connector.allowed_hosts,
                "endpoint": api_endpoint,
                "status": "error",
                "payload": api_payload,
                "summary": str(error),
            }

        tool_inputs = {
            "list_directory": json.dumps({"path": str(Path.cwd())}),
            "read_file": json.dumps({"path": str(Path(__file__).resolve())}),
            "execute_code": json.dumps({"code": "print(1 + 2)", "timeout_seconds": 2}),
            "api_request": json.dumps({"method": "GET", "url": "status", "headers": {}, "body": {"health": True}}),
        }

        tool_results = {}
        for name, data in tool_inputs.items():
            try:
                arguments = json.loads(data)
            except Exception:
                arguments = {"input": data}
            tool_results[name] = tool_executor.invoke(name, arguments, permission_context)

        verification = verifier.self_check()
        api_verification = verifier.verify_api_connector(api_endpoint, api_payload)
        return {
            "phase": "Stage 3",
            "baseline_context": {
                "phase": baseline_record.get("phase"),
                "candidate": baseline_record.get("candidate"),
                "config_hash": baseline_record.get("config_hash"),
            },
            "registered_tools": registry.list_tools(),
            "tool_results": {
                name: {
                    "success": result.success,
                    "output": result.output,
                    "error": result.error,
                }
                for name, result in tool_results.items()
            },
            "verification": verification,
            "api_request": api_request,
            "api_response": api_response,
            "api_verification": api_verification,
        }

    def _run_stage_4(
        self,
        modality_query: str = "Inspect sensory inputs",
        include_modalities: list[str] | None = None,
    ) -> dict[str, Any]:
        pipeline = PerceptionPipeline()
        sample_inputs, sample_metadata = pipeline.sample_inputs()
        aliases = {"image": "vision"}
        requested = include_modalities or ["vision", "audio", "structured"]
        modalities = [aliases.get(name, name) for name in requested]
        unsupported = [name for name in modalities if name not in sample_inputs]
        if unsupported:
            raise ValueError(
                "Unsupported completed Stage 4 modalities: " + ", ".join(unsupported)
                + ". Video and text are not claimed by the Stage 4 empirical milestone."
            )
        selected = {name: sample_inputs[name] for name in modalities}
        metadata = {name: sample_metadata[name] for name in modalities}
        integrated = pipeline.integrate(selected, metadata=metadata)
        state_projection = pipeline.project_into_hrm(
            integrated, np.zeros(BASELINE_CONFIG.shape.cognitive_dim, dtype=np.float32), max_delta_norm=0.5
        )
        return {
            "phase": "Stage 4",
            "modality_query": modality_query,
            "modalities": integrated["modalities"],
            "integrated_outputs": integrated["representations"],
            "combined_embedding_summary": f"{len(integrated['combined_embedding'])} dims",
            "fusion": {
                "weights": integrated["modality_weights"],
                "confidences": integrated["modality_confidences"],
                "missing_modalities": integrated["missing_modalities"],
                "contradictions": integrated["contradictions"],
                "provenance": integrated["provenance"],
                "diagnostics": integrated["diagnostics"],
            },
            "hrm_state_projection": state_projection,
            "completion_scope": "vision_audio_structured_fusion; video_experimental",
        }

    def _run_stage_5(
        self,
        baseline_record: dict[str, Any] | None = None,
        consensus_query: str = "Coordinate distributed HRM reasoning and planning",
        agent_roles: list[str] | None = None,
    ) -> dict[str, Any]:
        if baseline_record is None:
            try:
                baseline_record = run_baseline_pipeline(seed=0, save_artifacts=False)
            except ImportError:
                baseline_record = {
                    "phase": "Stage 1 placeholder",
                    "candidate": "baseline_placeholder",
                    "config_hash": BASELINE_CONFIG.config_hash(),
                    "seed": 0,
                    "backend": "cpu",
                    "profile": RUN_PROFILE,
                    "steps": BASELINE_CONFIG.steps,
                    "batch": BASELINE_CONFIG.shape.batch,
                    "perturb_step": BASELINE_CONFIG.perturb_step,
                    "perturb_strength": BASELINE_CONFIG.perturb_strength,
                    "L_total": 0.0,
                    "did_recover": False,
                    "ledger_pass": False,
                    "bounded_pass": False,
                }

        memory = LongTermMemory.from_baseline_record(baseline_record)
        agent_roles = agent_roles or ["safety", "efficiency", "planning", "recovery"]
        modules = [RoleSpecialistModule(role, memory) for role in agent_roles]

        main_task = DistributedTask.create(
            objective=consensus_query,
            inputs={"memory_reads": ["baseline_summary"], "memory_writes": ["stage5_consensus"]},
            required_capabilities=frozenset(agent_roles),
            dependencies=(),
            priority=80,
            deadline=None,
            retry_limit=1,
            metadata={"stage": 5, "query": consensus_query},
        )
        task_graph = TaskGraph([main_task])
        coordinator = DistributedCoordinator(task_graph, modules)
        coordination = coordinator.run()
        memory_summary = summarize_memory(memory)
        return {
            "phase": "Stage 5",
            "consensus_query": consensus_query,
            "agent_roles": agent_roles,
            "memory_summary": memory_summary,
            "coordination": coordination,
        }

    def _run_stage_6(
        self,
        baseline_record: dict[str, Any] | None = None,
        learning_rate: float = 0.05,
        starting_preferences: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        if baseline_record is None:
            try:
                baseline_record = run_baseline_pipeline(seed=0, save_artifacts=False)
            except ImportError:
                baseline_record = {
                    "phase": "Stage 1 placeholder",
                    "candidate": "baseline_placeholder",
                    "config_hash": BASELINE_CONFIG.config_hash(),
                    "seed": 0,
                    "backend": "cpu",
                    "profile": RUN_PROFILE,
                    "steps": BASELINE_CONFIG.steps,
                    "batch": BASELINE_CONFIG.shape.batch,
                    "perturb_step": BASELINE_CONFIG.perturb_step,
                    "perturb_strength": BASELINE_CONFIG.perturb_strength,
                    "L_total": 0.0,
                    "did_recover": False,
                    "ledger_pass": False,
                    "bounded_pass": False,
                }

        memory = LongTermMemory.from_baseline_record(baseline_record)
        learner = LearningSystem(memory)
        learning_result = learner.update(baseline_record, starting_preferences=starting_preferences, learning_rate=learning_rate)
        memory_summary = summarize_memory(memory)
        return {
            "phase": "Stage 6",
            "learning_rate": learning_rate,
            "starting_preferences": starting_preferences or {"exploration": 0.2, "safety": 0.3, "efficiency": 0.5, "bias": 0.0},
            "memory_summary": memory_summary,
            "adaptation_mode": learning_result.get("adaptation_mode", "controlled_heuristic"),
            "learning_result": learning_result,
        }

    def _run_placeholder_stage(self, stage: HRMStage, **kwargs: Any) -> dict[str, Any]:
        return {
            "phase": stage.name,
            "message": "Implemented as a structural placeholder for the HRM Theory integration.",
            "expected_steps": stage.steps,
            "details": {
                "status": "pending",
                "note": "This stage is ready to receive future implementation of the described capabilities.",
            },
        }


if __name__ == "__main__":
    theory = HRMTheory()
    print(theory.run_stage(1))
