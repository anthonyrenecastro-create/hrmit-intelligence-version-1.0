from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hrm.memory import LongTermMemory, Planner

ROLE_STRATEGIES: dict[str, dict[str, Any]] = {
    "safety": {
        "template": "Safety-aware reasoning for",
        "retrieve_k": 4,
        "plan_steps": 5,
        "weight": 1.3,
        "approach": "minimize risk and enforce stability",
    },
    "efficiency": {
        "template": "Efficiency-focused reasoning for",
        "retrieve_k": 4,
        "plan_steps": 5,
        "weight": 1.1,
        "approach": "optimize resource usage and throughput",
    },
    "planning": {
        "template": "Strategic planning reasoning for",
        "retrieve_k": 5,
        "plan_steps": 6,
        "weight": 1.2,
        "approach": "align long-horizon objectives",
    },
    "recovery": {
        "template": "Recovery-centric reasoning for",
        "retrieve_k": 4,
        "plan_steps": 5,
        "weight": 1.2,
        "approach": "maximize resilience and fallback readiness",
    },
}

ROLE_AFFINITY: dict[tuple[str, str], float] = {
    ("safety", "recovery"): 0.95,
    ("recovery", "safety"): 0.95,
    ("planning", "efficiency"): 0.85,
    ("efficiency", "planning"): 0.85,
    ("safety", "planning"): 0.7,
    ("planning", "safety"): 0.7,
    ("efficiency", "recovery"): 0.65,
    ("recovery", "efficiency"): 0.65,
}


def get_role_profile(role: str) -> dict[str, Any]:
    return ROLE_STRATEGIES.get(role, {
        "template": "General reasoning for",
        "retrieve_k": 3,
        "plan_steps": 4,
        "weight": 1.0,
        "approach": "balance priorities",
    })


def role_affinity(source: str, target: str) -> float:
    if source == target:
        return 1.0
    return ROLE_AFFINITY.get((source, target), 0.45)


@dataclass(frozen=True)
class AgentTrace:
    agent: str
    role: str
    query: str
    memory_keys: list[str]
    recommendations: list[str]
    confidence: float
    summary: str


class AgentProxy:
    def __init__(self, name: str, role: str, memory: LongTermMemory) -> None:
        self.name = name
        self.role = role
        self.memory = memory
        self.planner = Planner(memory)
        self.profile = get_role_profile(role)

    def reason(self, query: str) -> AgentTrace:
        role_query = f"{self.profile['template']} {query}"
        retrieved = self.memory.retrieve(role_query, k=self.profile["retrieve_k"])
        plan = self.planner.create_plan(role_query, n_steps=self.profile["plan_steps"])
        memory_keys = [entry["key"] for entry in retrieved]
        confidence = min(1.0, len(retrieved) / max(1, self.profile["retrieve_k"]))
        summary = (
            f"{self.name} ({self.role}) used a {self.profile['approach']} strategy, "
            f"reasoned over {len(retrieved)} memory entries, and proposed {len(plan.steps)} steps."
        )
        return AgentTrace(
            agent=self.name,
            role=self.role,
            query=role_query,
            memory_keys=memory_keys,
            recommendations=plan.steps,
            confidence=confidence * self.profile["weight"],
            summary=summary,
        )

    def absorb_peer_traces(self, own_trace: AgentTrace, peer_traces: list[AgentTrace]) -> dict[str, Any]:
        peer_summaries = [trace.summary for trace in peer_traces if trace.agent != self.name]
        peer_recommendations = [recommendation for trace in peer_traces if trace.agent != self.name for recommendation in trace.recommendations]
        agreement_count = sum(1 for recommendation in own_trace.recommendations if recommendation in peer_recommendations)
        agreement_ratio = agreement_count / max(1.0, len(own_trace.recommendations))
        recommendation_count = sum(len(trace.recommendations) for trace in peer_traces)
        affinity_scores = [role_affinity(self.role, trace.role) for trace in peer_traces if trace.agent != self.name]
        average_affinity = float(sum(affinity_scores) / max(1.0, len(affinity_scores)))
        return {
            "agent": self.name,
            "role": self.role,
            "peer_count": len(peer_summaries),
            "peer_summary": "; ".join(peer_summaries[:3]) or "No peer traces available",
            "agreement_ratio": float(agreement_ratio),
            "role_affinity": average_affinity,
            "adjusted_confidence": min(1.0, own_trace.confidence + 0.25 * agreement_ratio * average_affinity),
            "adjustments": [
                f"Consider peer insight from {trace.agent} ({trace.role})"
                for trace in peer_traces
                if trace.agent != self.name
            ][:2],
            "recommendation_count": recommendation_count,
        }


class DistributedCognitionCoordinator:
    def __init__(self, agents: list[AgentProxy]) -> None:
        self.agents = agents

    def coordinate(self, query: str) -> dict[str, Any]:
        reasoning_traces = [agent.reason(query) for agent in self.agents]
        peer_exchanges = [agent.absorb_peer_traces(trace, reasoning_traces) for agent, trace in zip(self.agents, reasoning_traces)]
        belief_states = self._merge_belief_states(reasoning_traces, peer_exchanges)
        consensus = self._resolve_consensus(reasoning_traces)
        distributed_plan = self._build_distributed_plan(consensus, belief_states)
        return {
            "query": query,
            "agent_count": len(self.agents),
            "reasoning_traces": [trace.__dict__ for trace in reasoning_traces],
            "peer_exchanges": peer_exchanges,
            "belief_states": belief_states,
            "consensus": consensus,
            "distributed_plan": distributed_plan,
        }

    def _merge_belief_states(self, traces: list[AgentTrace], peer_exchanges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        shared_memory = set(traces[0].memory_keys) if traces else set()
        for trace in traces[1:]:
            shared_memory &= set(trace.memory_keys)

        merged = []
        for trace, exchange in zip(traces, peer_exchanges):
            merged.append(
                {
                    "agent": trace.agent,
                    "role": trace.role,
                    "memory_keys": trace.memory_keys,
                    "shared_memory_keys": sorted(shared_memory),
                    "recommendations": trace.recommendations,
                    "peer_adjustments": exchange["adjustments"],
                    "confidence": trace.confidence,
                    "peer_summary": exchange["peer_summary"],
                    "agreement_ratio": exchange["agreement_ratio"],
                }
            )
        return merged

    def _resolve_consensus(self, traces: list[AgentTrace]) -> dict[str, Any]:
        recommendation_scores: dict[str, float] = {}
        role_weights: dict[str, float] = {}
        for trace in traces:
            weight = min(1.0, trace.confidence) or 1.0
            role_weights[trace.role] = role_weights.get(trace.role, 0.0) + weight
            for recommendation in trace.recommendations:
                recommendation_scores[recommendation] = recommendation_scores.get(recommendation, 0.0) + weight

        if not recommendation_scores:
            return {"summary": "No recommendations available", "agreement_score": 0.0, "top_recommendations": [], "shared_recommendations": [], "role_influence": {}}

        top_recommendations = sorted(recommendation_scores.items(), key=lambda item: item[1], reverse=True)
        total_weight = sum(recommendation_scores.values())
        agreement_score = float(top_recommendations[0][1]) / max(1.0, total_weight)
        shared_recommendations = [rec for rec, score in top_recommendations if score >= len(traces) * 0.8]
        return {
            "summary": "Consensus achieved on core recommendations." if agreement_score >= 0.5 else "Partial agreement across agent recommendations.",
            "agreement_score": agreement_score,
            "top_recommendations": [rec for rec, _ in top_recommendations[:3]],
            "shared_recommendations": shared_recommendations,
            "recommendation_scores": {rec: score for rec, score in top_recommendations[:6]},
            "role_influence": {role: float(weight) for role, weight in role_weights.items()},
        }

    def _build_distributed_plan(self, consensus: dict[str, Any], belief_states: list[dict[str, Any]]) -> dict[str, Any]:
        combined_steps = []
        for state in belief_states:
            combined_steps.extend(state["recommendations"][:2])

        unique_steps = []
        for step in combined_steps:
            if step not in unique_steps:
                unique_steps.append(step)

        preferred_agent = max(belief_states, key=lambda state: state["confidence"], default={"agent": "none"})["agent"] if belief_states else "none"
        return {
            "consensus_summary": consensus["summary"],
            "agreement_score": consensus["agreement_score"],
            "plan_steps": unique_steps[:8],
            "preferred_agent": preferred_agent,
            "top_recommendations": consensus["top_recommendations"],
            "shared_recommendations": consensus.get("shared_recommendations", []),
        }
