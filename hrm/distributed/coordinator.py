from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

from hrm.distributed.execution.sequential import SequentialExecutor
from hrm.distributed.modules import SpecialistModule
from hrm.distributed.scheduler import Scheduler
from hrm.distributed.task_graph import TaskGraph
from hrm.distributed.types import CognitiveTask, ConsensusResult, DelegationRecord, ModuleProposal
from hrm.distributed.shared_memory import SharedMemory


@dataclass
class CoordinatorState:
    task_graph: TaskGraph
    shared_memory: SharedMemory
    delegation_history: list[DelegationRecord]
    proposals: list[ModuleProposal]
    disagreements: list[dict[str, Any]]
    consensus_records: list[ConsensusResult]
    failures: list[dict[str, Any]]


class DistributedCoordinator:
    def __init__(
        self,
        task_graph: TaskGraph,
        modules: list[SpecialistModule],
        scheduler: Scheduler | None = None,
        executor: SequentialExecutor | None = None,
    ) -> None:
        self.task_graph = task_graph
        self.modules = modules
        self.scheduler = scheduler or Scheduler()
        self.shared_memory = SharedMemory()
        self.executor = executor or SequentialExecutor()
        self.state = CoordinatorState(
            task_graph=task_graph,
            shared_memory=self.shared_memory,
            delegation_history=[],
            proposals=[],
            disagreements=[],
            consensus_records=[],
            failures=[],
        )
        self.module_lookup = {module.module_id: module for module in modules}

    def run(self) -> dict[str, Any]:
        validation = self.task_graph.validate()
        if not validation["valid"]:
            return {"status": "invalid", "validation": validation}

        results: list[dict[str, Any]] = []
        while True:
            ready_tasks = self.task_graph.get_ready_tasks()
            if not ready_tasks:
                break
            assignments = [self.scheduler.assign(task, self.modules, self.state) for task in ready_tasks]
            for assignment in assignments:
                if assignment is None:
                    continue
                task = assignment.task
                self.task_graph.set_status(task.task_id, "running")
                proposal = assignment.module.execute(task, self)
                self.state.proposals.append(proposal)
                self.task_graph.set_result(task.task_id, {"proposal_id": proposal.proposal_id, "confidence": proposal.confidence})
                self.task_graph.set_status(task.task_id, "completed")
                results.append({"task_id": task.task_id, "module_id": assignment.module.module_id, "proposal_id": proposal.proposal_id, "confidence": proposal.confidence})

        consensus = self._resolve_consensus(self.state.proposals)
        distributed_plan = self._build_distributed_plan(self.state.proposals, consensus)
        return {
            "status": "completed",
            "agent_count": len(self.modules),
            "results": results,
            "task_graph": self.task_graph.to_dict(),
            "reasoning_traces": [proposal.__dict__ for proposal in self.state.proposals],
            "consensus": consensus,
            "distributed_plan": distributed_plan,
            "shared_memory_snapshot": self.shared_memory.snapshot(),
            "conflicts": self.shared_memory.conflicts_snapshot(),
        }

    def _resolve_consensus(self, proposals: list[ModuleProposal]) -> dict[str, Any]:
        if not proposals:
            return {
                "summary": "No proposals were generated.",
                "agreement_score": 0.0,
                "top_recommendations": [],
                "role_influence": {},
            }

        recommendation_scores: dict[str, float] = {}
        role_influence: dict[str, float] = {}
        for proposal in proposals:
            recommendation_scores[proposal.answer] = recommendation_scores.get(proposal.answer, 0.0) + proposal.confidence
            role_influence[proposal.module_id.split("_")[-1]] = role_influence.get(proposal.module_id.split("_")[-1], 0.0) + proposal.confidence

        sorted_recommendations = sorted(recommendation_scores.items(), key=lambda item: item[1], reverse=True)
        total = sum(recommendation_scores.values())
        agreement_score = float(sorted_recommendations[0][1]) / max(1.0, total)
        return {
            "summary": "Consensus achieved on distributed recommendations." if agreement_score >= 0.3 else "Partial consensus achieved.",
            "agreement_score": agreement_score,
            "top_recommendations": [rec for rec, _ in sorted_recommendations[:3]],
            "role_influence": {role: float(score) for role, score in role_influence.items()},
        }

    def _build_distributed_plan(self, proposals: list[ModuleProposal], consensus: dict[str, Any]) -> dict[str, Any]:
        plan_steps = [proposal.answer for proposal in proposals]
        unique_steps = []
        for step in plan_steps:
            if step not in unique_steps:
                unique_steps.append(step)
        preferred_agent = max(self.modules, key=lambda module: module.historical_reliability, default=None)
        return {
            "consensus_summary": consensus["summary"],
            "agreement_score": consensus["agreement_score"],
            "plan_steps": unique_steps[:8],
            "preferred_agent": preferred_agent.module_id if preferred_agent else "none",
            "top_recommendations": consensus["top_recommendations"],
        }
