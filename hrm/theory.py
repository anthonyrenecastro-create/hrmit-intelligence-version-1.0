from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hrm.baseline import (
    BaselineConfigurationError,
    BaselineImportError,
    BaselineRuntimeError,
    baseline_training_loss,
    run_baseline_pipeline,
    train_baseline_pipeline,
    BASELINE_CONFIG,
    RUN_PROFILE,
)
from hrm.distributed import DistributedCoordinator, RoleSpecialistModule, TaskGraph
from hrm.distributed.types import CognitiveTask as DistributedTask
from hrm.memory import LongTermMemory, MemoryQuery, MemorySystem, Planner, summarize_memory
from hrm.multimodal.pipeline import MultimodalPipeline
from hrm.multimodal.types import ModalityInput
from hrm.tools import APIConnector, AuditLogger, PermissionContext, PathPolicy, SelfVerifier, ToolExecutor, ToolRegistry
from hrm.learning import (
    AdaptationCandidate,
    AdaptationProvenance,
    CandidateTrainer,
    EvaluationConfig,
    Evaluator,
    ExperienceRecord,
    ExperienceStore,
    FeedbackCapture,
    PromotionGate,
    ReplayBuffer,
    ReplayConfig,
    RollbackManager,
    TaskOutcome,
    TrainingConfig,
    PreferenceModelBaseline,
)


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
        diagnostic: dict[str, str] | None = None
        artifact: dict[str, Any] | None = None
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
        except BaselineImportError as error:
            runtime_message = "Stage 1 used the placeholder baseline path because JAX was unavailable."
            diagnostic = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "resolution": "Install JAX or configure the runtime with TPU/CPU support.",
            }
        except BaselineConfigurationError as error:
            runtime_message = "Stage 1 failed due to invalid baseline configuration."
            diagnostic = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "resolution": "Review ExperimentConfig values and retry.",
            }
        except BaselineRuntimeError as error:
            runtime_message = "Stage 1 failed due to runtime instability."
            diagnostic = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "resolution": "Inspect ledger diagnostics and numeric stability constraints.",
            }
        except Exception as error:
            runtime_message = "Stage 1 failed due to an unexpected software error."
            diagnostic = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "resolution": "Investigate the execution stack and underlying software issue.",
            }
        if artifact is None:
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
        result: dict[str, Any] = {
            "phase": "Stage 1",
            "baseline_record": artifact,
            "output_dir": str(output_dir),
            "runtime_message": runtime_message,
            "status": "success" if diagnostic is None else "failed",
        }
        if diagnostic is not None:
            result["diagnostic"] = diagnostic
        return result

    def _run_stage_2(
        self,
        baseline_record: dict[str, Any] | None = None,
        plan_query: str = "Improve HRM safety and recovery",
        output_dir: Path | str = "hrm_baseline_outputs",
    ) -> dict[str, Any]:
        if baseline_record is None:
            try:
                baseline_record = run_baseline_pipeline(seed=0, save_artifacts=False)
            except BaselineImportError as error:
                raise RuntimeError(
                    "Stage 2 cannot proceed because Stage 1 baseline execution failed due to missing JAX."
                ) from error
            except BaselineConfigurationError as error:
                raise RuntimeError(
                    f"Stage 2 cannot proceed because Stage 1 baseline configuration is invalid: {error}"
                ) from error
            except BaselineRuntimeError as error:
                raise RuntimeError(
                    f"Stage 2 cannot proceed because Stage 1 runtime check failed: {error}"
                ) from error
            except Exception as error:
                raise RuntimeError(
                    f"Stage 2 cannot proceed because Stage 1 encountered an unexpected error: {error}"
                ) from error
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
            except BaselineImportError as error:
                raise RuntimeError(
                    "Stage 3 cannot proceed because Stage 1 baseline execution failed due to missing JAX."
                ) from error
            except BaselineConfigurationError as error:
                raise RuntimeError(
                    f"Stage 3 cannot proceed because Stage 1 baseline configuration is invalid: {error}"
                ) from error
            except BaselineRuntimeError as error:
                raise RuntimeError(
                    f"Stage 3 cannot proceed because Stage 1 runtime check failed: {error}"
                ) from error
            except Exception as error:
                raise RuntimeError(
                    f"Stage 3 cannot proceed because Stage 1 encountered an unexpected error: {error}"
                ) from error

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
        modality_query: str = "Text",
        include_modalities: list[str] | None = None,
    ) -> dict[str, Any]:
        pipeline = MultimodalPipeline()
        sample_inputs = pipeline.sample_inputs()
        modalities = include_modalities or ["vision", "audio", "structured"]
        selected_inputs = {k: v for k, v in sample_inputs.__dict__.items() if k in modalities}
        if not selected_inputs:
            raise ValueError("No valid modalities were provided to Stage 4.")

        representations = []
        for modality, payload in selected_inputs.items():
            source_id = f"{modality}_{modality_query}"
            decoded = pipeline.decode(ModalityInput(modality=modality, source_id=source_id, payload=payload, timestamp=None))
            preprocessed = pipeline.preprocess(decoded)
            representation = pipeline.represent(preprocessed)
            representations.append(representation)

        fusion_result = pipeline.fuse(representations, expected_modalities=list(selected_inputs.keys()))
        projected = [pipeline.project(rep) for rep in representations]
        modality_names = [rep.modality for rep in representations]
        return {
            "phase": "Stage 4",
            "modality_query": modality_query,
            "modalities": modality_names,
            "readiness": float(np.mean([rep.confidence for rep in representations])),
            "representations": representations,
            "projection": projected,
            "fusion": fusion_result,
            "integration_summary": {
                "fused_shape": tuple(fusion_result.fused_latent.shape),
                "modalities": modality_names,
                "missing_modalities": fusion_result.missing_modalities,
                "contradictions": [c.__dict__ for c in fusion_result.contradictions],
            },
            "combined_embedding_summary": f"{len(fusion_result.fused_latent)} dims",
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
        feedback_capture = FeedbackCapture()
        experience_store = ExperienceStore()
        replay_buffer = ReplayBuffer(experience_store, config=ReplayConfig())
        evaluator = Evaluator()
        promotion_gate = PromotionGate()
        rollback_manager = RollbackManager()
        base_checkpoint = {
            "checkpoint_id": baseline_record.get("candidate", "baseline_placeholder"),
            "held_out_accuracy": float(baseline_record.get("L_total", 0.0)),
            "prior_task_recall": float(baseline_record.get("bounded_pass", 1.0)),
            "safety_bias": float(starting_preferences.get("safety", 0.0) if starting_preferences else 0.0),
        }
        training_config = TrainingConfig(
            seed=0,
            steps=5,
            learning_rate=learning_rate,
            max_update_norm=0.5,
            max_relative_change=0.05,
            gradient_clip_norm=1.0,
            frozen_scopes=("core",),
            trainable_scopes=("fusion", "calibration", "memory_retrieval"),
        )

        if starting_preferences is None:
            starting_preferences = {"exploration": 0.2, "safety": 0.3, "efficiency": 0.5, "bias": 0.0}

        feedback = feedback_capture.capture_from_task(
            TaskOutcome(
                task_id="baseline_signal",
                task_type="baseline",
                inputs={"baseline_record": baseline_record},
                output=None,
                expected_output=None,
                success=True,
                score=1.0,
                completion_time=0.0,
                module_ids=("learning",),
                tool_audit_ids=(),
                memory_ids=(),
                metadata={"source": "baseline"},
            ),
            source="verifier",
            feedback_type="numeric_score",
            value=float(baseline_record.get("L_total", 0.0)),
            confidence=1.0,
            scope="baseline",
            objective=True,
            suitable_for_training=True,
        )
        task_outcome = TaskOutcome(
            task_id="baseline_signal",
            task_type="baseline",
            inputs={"baseline_record": baseline_record},
            output=None,
            expected_output=None,
            success=True,
            score=1.0,
            completion_time=0.0,
            module_ids=("learning",),
            tool_audit_ids=(),
            memory_ids=(),
            metadata={"source": "baseline"},
        )
        experience = ExperienceRecord.create(
            task_outcome=task_outcome,
            feedback=(feedback,),
            reward=1.0,
            priority=1.0,
            provenance={"baseline_candidate": baseline_record.get("candidate")},
        )
        experience_store.add(experience)
        replay_buffer.add(experience)

        trainer = CandidateTrainer(base_checkpoint=base_checkpoint, trainable_scopes=training_config.trainable_scopes, config=training_config)
        provenance = AdaptationProvenance.create(
            candidate_id="pending",
            parent_checkpoint_id=base_checkpoint["checkpoint_id"],
            active_checkpoint_before=base_checkpoint["checkpoint_id"],
            experience_ids=(experience.experience_id,),
            feedback_ids=(feedback.feedback_id,),
            replay_strategy="prioritized",
            training_seed=training_config.seed,
            training_config=training_config.__dict__,
            trainable_parameter_names=training_config.trainable_scopes,
            update_metrics={"update_norm": 0.0},
            held_out_metrics={},
            regression_metrics={},
            safety_metrics={},
            promotion_decision="pending",
            rejection_reasons=(),
        )
        candidate = trainer.create_candidate(parent_checkpoint_id=base_checkpoint["checkpoint_id"], provenance_id=provenance.provenance_id, scope_names=training_config.trainable_scopes)

        try:
            candidate_checkpoint = trainer.train([experience])
            checkpoint_path = trainer.checkpoint(candidate.checkpoint_path)
            candidate = AdaptationCandidate(
                **{**candidate.__dict__, "checkpoint_path": str(checkpoint_path), "update_norm": trainer.update_norm, "state": "trained"}
            )
        except ValueError as error:
            return {
                "phase": "Stage 6",
                "learning_rate": learning_rate,
                "starting_preferences": starting_preferences,
                "memory_summary": summarize_memory(memory),
                "adaptation_mode": "controlled_heuristic",
                "learning_result": {
                    "status": "candidate_rejected",
                    "reason": str(error),
                },
            }

        held_out_data = [{"difficulty": 1.0} for _ in range(3)]
        report = evaluator.evaluate_parent_and_candidate(base_checkpoint, candidate_checkpoint, held_out_data, EvaluationConfig())
        promotion = promotion_gate.evaluate(report, candidate_checkpoint)

        if not report.accepted:
            restored = rollback_manager.rollback(base_checkpoint, base_checkpoint, candidate.candidate_id, reason=promotion["decision"])
            provenance = AdaptationProvenance(
                **{**provenance.__dict__, "active_checkpoint_after": restored.get("checkpoint_id", base_checkpoint["checkpoint_id"]), "promotion_decision": promotion["decision"], "rejection_reasons": promotion["rejection_reasons"], "rollback_id": f"rb_{candidate.candidate_id}"}
            )
            return {
                "phase": "Stage 6",
                "learning_rate": learning_rate,
                "starting_preferences": starting_preferences,
                "memory_summary": summarize_memory(memory),
                "adaptation_mode": "controlled_heuristic",
                "learning_result": {
                    "status": promotion["decision"],
                    "evaluation": report.__dict__,
                    "provenance": provenance.__dict__,
                },
            }

        provenance = AdaptationProvenance(
            **{**provenance.__dict__, "active_checkpoint_after": candidate_checkpoint.get("checkpoint_id", candidate.checkpoint_path), "promotion_decision": "accepted"}
        )
        return {
            "phase": "Stage 6",
            "learning_rate": learning_rate,
            "starting_preferences": starting_preferences,
            "memory_summary": summarize_memory(memory),
            "adaptation_mode": "controlled_heuristic",
            "learning_result": {
                "status": promotion["decision"],
                "evaluation": report.__dict__,
                "provenance": provenance.__dict__,
            },
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
