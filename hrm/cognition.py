from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hrm.memory import LongTermMemory, Planner


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

    def reason(self, query: str) -> AgentTrace:
        role_query = f"{self.role} {query}"
        retrieved = self.memory.retrieve(role_query, k=3)
        plan = self.planner.create_plan(role_query, n_steps=4)
        memory_keys = [entry["key"] for entry in retrieved]
        confidence = min(1.0, len(retrieved) / 3)
        summary = (
            f"{self.name} ({self.role}) reasoned over {len(retrieved)} memory entries "
            f"and proposed {len(plan.steps)} distributed cognition steps."
        )
        return AgentTrace(
            agent=self.name,
            role=self.role,
            query=role_query,
            memory_keys=memory_keys,
            recommendations=plan.steps,
            confidence=confidence,
            summary=summary,
        )

    def absorb_peer_traces(self, own_trace: AgentTrace, peer_traces: list[AgentTrace]) -> dict[str, Any]:
        peer_summaries = [trace.summary for trace in peer_traces if trace.agent != self.name]
        peer_recommendations = [recommendation for trace in peer_traces if trace.agent != self.name for recommendation in trace.recommendations]
        agreement_count = sum(1 for recommendation in own_trace.recommendations if recommendation in peer_recommendations)
        agreement_ratio = agreement_count / max(1.0, len(own_trace.recommendations))
        recommendation_count = sum(len(trace.recommendations) for trace in peer_traces)
        return {
            "agent": self.name,
            "role": self.role,
            "peer_count": len(peer_summaries),
            "peer_summary": "; ".join(peer_summaries[:3]) or "No peer traces available",
            "agreement_ratio": float(agreement_ratio),
            "adjusted_confidence": min(1.0, own_trace.confidence + 0.25 * agreement_ratio),
            "adjustments": [
                f"Consider peer insight from {trace.agent}"
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
        for trace in traces:
            weight = trace.confidence or 1.0
            for recommendation in trace.recommendations:
                recommendation_scores[recommendation] = recommendation_scores.get(recommendation, 0.0) + weight

        if not recommendation_scores:
            return {"summary": "No recommendations available", "agreement_score": 0.0, "top_recommendations": [], "shared_recommendations": []}

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
