from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .configuration import HRMTransitionConfig
from .events import propose_events
from .invariants import validate_state
from .interactions import compute_interactions
from .ledger import TransitionLedger
from .metrics import state_metrics
from .observables import compute_observables
from .proposals import MechanismProposal
from .random_state import RandomStateManager
from .safety import apply_safe_transition
from .snapshot import freeze_state
from .events import StructuralEvent
from .state import (
    BudgetState,
    CognitionState,
    FieldState,
    GeometryState,
    HRMState,
    HierarchyState,
    MemoryState,
    TopologyState,
)
from .transport import transport_with_index_map
from .mechanisms.base import HRMInput, HRMMechanism, TransitionContext


@dataclass(frozen=True)
class TransitionResult:
    state: HRMState
    ledger: TransitionLedger


def _build_ring_laplacian(node_count: int, dtype: str, metric_scale: float) -> np.ndarray:
    laplacian = np.eye(node_count, dtype=dtype) * 2.0
    off_diag = np.eye(node_count, k=1, dtype=dtype) + np.eye(node_count, k=-1, dtype=dtype)
    laplacian = laplacian - off_diag
    if node_count > 1:
        laplacian[0, -1] = -1.0
        laplacian[-1, 0] = -1.0
    return laplacian * np.asarray(metric_scale, dtype=dtype)


def _build_hierarchy_ops(node_count: int, dtype: str) -> tuple[np.ndarray, np.ndarray]:
    coarse_count = max(1, node_count // 2)
    restriction = np.zeros((coarse_count, node_count), dtype=dtype)
    for i in range(coarse_count):
        left = 2 * i
        right = min(node_count - 1, left + 1)
        restriction[i, left] = 0.5
        restriction[i, right] = 0.5
    prolongation = restriction.T * 2.0
    return restriction, prolongation


def _compose_remove_tail_map(old_to_new: list[int], remove_count: int, current_size: int) -> tuple[list[int], int]:
    if current_size <= 1:
        return old_to_new, current_size
    remove_count = max(0, min(remove_count, current_size - 1))
    if remove_count == 0:
        return old_to_new, current_size
    cutoff = current_size - remove_count
    composed: list[int] = []
    for mapped in old_to_new:
        if mapped < 0:
            composed.append(-1)
        elif mapped >= cutoff:
            composed.append(-1)
        else:
            composed.append(mapped)
    return composed, cutoff


def _coarse_old_to_new_map(old_nodes: int, new_nodes: int) -> tuple[int, ...]:
    old_coarse = max(1, old_nodes // 2)
    new_coarse = max(1, new_nodes // 2)
    mapping = [-1] * old_coarse
    for idx in range(min(old_coarse, new_coarse)):
        mapping[idx] = idx
    return tuple(mapping)


class CanonicalTransitionEngine:
    def __init__(self, config: HRMTransitionConfig, mechanisms: list[HRMMechanism]) -> None:
        self.config = config
        self.mechanisms = mechanisms

    def _collect_proposals(self, snapshot, external_input, observables, context) -> list[MechanismProposal]:
        proposals: list[MechanismProposal] = []
        for mechanism in self.mechanisms:
            proposal = mechanism.propose(snapshot, external_input, observables, context)
            if proposal.source_state_version != snapshot.source_version:
                raise ValueError(f"{mechanism.mechanism_id} proposal source version mismatch")
            proposals.append(proposal)
        return proposals

    def step(self, state: HRMState, external_input: HRMInput) -> TransitionResult:
        start = time.perf_counter()

        invariant_report = validate_state(state, self.config)
        if not invariant_report.valid:
            raise ValueError(f"Invalid state before transition: {invariant_report.errors}")

        snapshot = freeze_state(state)

        random_manager = RandomStateManager.from_state(state.rng_state)
        rng_before = random_manager.digest()

        observables = compute_observables(snapshot)
        context = TransitionContext(
            step=state.step,
            dt=self.config.dt,
            deterministic=self.config.deterministic_mode,
        )

        proposals = self._collect_proposals(snapshot, external_input, observables, context)

        interactions = compute_interactions(snapshot.state.phi.phi.shape)

        proposed_phi_contrib: dict[str, np.ndarray] = {}
        proposed_memory_contrib: dict[str, np.ndarray] = {}
        proposed_cognition_contrib: dict[str, np.ndarray] = {}
        proposed_hierarchy_contrib: dict[str, np.ndarray] = {}

        activations: dict[str, float] = {}
        total_estimated_cost = 0.0
        for proposal in proposals:
            activations[proposal.mechanism_id] = proposal.activation
            total_estimated_cost += proposal.estimated_cost
            if proposal.delta.phi is not None:
                proposed_phi_contrib[proposal.mechanism_id] = proposal.delta.phi
            if proposal.delta.memory is not None:
                proposed_memory_contrib[proposal.mechanism_id] = proposal.delta.memory
            if proposal.delta.cognition_latent is not None:
                proposed_cognition_contrib[proposal.mechanism_id] = proposal.delta.cognition_latent
            if proposal.delta.hierarchy_coarse is not None:
                proposed_hierarchy_contrib[proposal.mechanism_id] = proposal.delta.hierarchy_coarse

        phi_delta = np.zeros_like(state.phi.phi)
        for value in proposed_phi_contrib.values():
            phi_delta = phi_delta + value
        phi_delta = phi_delta + interactions

        memory_delta = np.zeros_like(state.memory.working)
        for value in proposed_memory_contrib.values():
            memory_delta = memory_delta + value

        cognition_delta = np.zeros_like(state.cognition.latent)
        for value in proposed_cognition_contrib.values():
            cognition_delta = cognition_delta + value

        hierarchy_delta = np.zeros_like(state.hierarchy.coarse)
        for value in proposed_hierarchy_contrib.values():
            hierarchy_delta = hierarchy_delta + value

        provisional_phi = state.phi.phi + phi_delta
        provisional_memory = state.memory.working + memory_delta
        provisional_cognition = state.cognition.latent + cognition_delta
        provisional_hierarchy = state.hierarchy.coarse + hierarchy_delta

        proposed_events = propose_events(snapshot, external_input, observables, context)

        accepted_events: list[StructuralEvent] = []
        rejected_events: list[StructuralEvent] = []
        event_allowance = state.budget.event_allowance
        write_index = state.memory.write_index
        topology_node_count = state.topology.node_count
        topology_edge_count = state.topology.edge_count
        topology_version = state.topology.version
        topology_metric_scale = state.geometry.metric_scale
        old_to_new_index_map: list[int] = list(range(state.topology.node_count))
        topology_changed = False
        topology_structural_enabled = self.config.topology_enabled and self.config.ablations.topology
        for event in proposed_events:
            if event.event_type == "memory_write_head" and event_allowance > 0:
                accepted_events.append(
                    StructuralEvent(
                        event_id=event.event_id,
                        event_type=event.event_type,
                        block=event.block,
                        accepted=True,
                        reason="accepted_memory_write_head",
                        priority=event.priority,
                        magnitude=event.magnitude,
                        metadata={**(event.metadata or {}), "accepted_step": state.step},
                    )
                )
                event_allowance -= 1
                write_index = (write_index + 1) % max(1, state.memory.capacity)
            elif event.event_type == "field_stability_guard" and event_allowance > 0:
                accepted_events.append(
                    StructuralEvent(
                        event_id=event.event_id,
                        event_type=event.event_type,
                        block=event.block,
                        accepted=True,
                        reason="accepted_field_stability_guard",
                        priority=event.priority,
                        magnitude=event.magnitude,
                        metadata={**(event.metadata or {}), "accepted_step": state.step},
                    )
                )
                event_allowance -= 1
            elif event.event_type == "topology_add_nodes" and event_allowance > 0 and topology_structural_enabled:
                requested = int((event.metadata or {}).get("add_nodes", max(1, round(event.magnitude))))
                add_nodes = max(0, min(requested, self.config.topology_max_add_nodes))
                if add_nodes > 0:
                    accepted_events.append(
                        StructuralEvent(
                            event_id=event.event_id,
                            event_type=event.event_type,
                            block=event.block,
                            accepted=True,
                            reason="accepted_topology_add_nodes",
                            priority=event.priority,
                            magnitude=float(add_nodes),
                            metadata={**(event.metadata or {}), "accepted_step": state.step, "effective_add_nodes": add_nodes},
                        )
                    )
                    topology_node_count += add_nodes
                    topology_edge_count = max(topology_node_count, topology_edge_count + add_nodes)
                    topology_metric_scale += self.config.topology_metric_scale_step
                    topology_changed = True
                    event_allowance -= 1
                else:
                    rejected_events.append(
                        StructuralEvent(
                            event_id=event.event_id,
                            event_type=event.event_type,
                            block=event.block,
                            accepted=False,
                            reason="topology_add_capped_to_zero",
                            priority=event.priority,
                            magnitude=event.magnitude,
                            metadata=event.metadata,
                        )
                    )
            elif event.event_type == "topology_remove_nodes" and event_allowance > 0 and topology_structural_enabled:
                requested = int((event.metadata or {}).get("remove_nodes", max(1, round(event.magnitude))))
                remove_nodes = max(0, min(requested, self.config.topology_max_remove_nodes))
                if remove_nodes > 0 and topology_node_count > 1:
                    old_to_new_index_map, topology_node_count = _compose_remove_tail_map(
                        old_to_new_index_map,
                        remove_count=remove_nodes,
                        current_size=topology_node_count,
                    )
                    topology_edge_count = max(topology_node_count, topology_edge_count - remove_nodes)
                    topology_metric_scale = max(0.1, topology_metric_scale - self.config.topology_metric_scale_step)
                    accepted_events.append(
                        StructuralEvent(
                            event_id=event.event_id,
                            event_type=event.event_type,
                            block=event.block,
                            accepted=True,
                            reason="accepted_topology_remove_nodes",
                            priority=event.priority,
                            magnitude=float(remove_nodes),
                            metadata={**(event.metadata or {}), "accepted_step": state.step, "effective_remove_nodes": remove_nodes},
                        )
                    )
                    topology_changed = True
                    event_allowance -= 1
                else:
                    rejected_events.append(
                        StructuralEvent(
                            event_id=event.event_id,
                            event_type=event.event_type,
                            block=event.block,
                            accepted=False,
                            reason="topology_remove_not_applicable",
                            priority=event.priority,
                            magnitude=event.magnitude,
                            metadata=event.metadata,
                        )
                    )
            elif event.event_type == "topology_rewire_ring" and event_allowance > 0 and topology_structural_enabled:
                accepted_events.append(
                    StructuralEvent(
                        event_id=event.event_id,
                        event_type=event.event_type,
                        block=event.block,
                        accepted=True,
                        reason="accepted_topology_rewire_ring",
                        priority=event.priority,
                        magnitude=event.magnitude,
                        metadata={**(event.metadata or {}), "accepted_step": state.step},
                    )
                )
                topology_metric_scale += 0.5 * self.config.topology_metric_scale_step
                topology_changed = True
                event_allowance -= 1
            elif event.event_type in {"topology_add_nodes", "topology_remove_nodes", "topology_rewire_ring"}:
                rejected_events.append(
                    StructuralEvent(
                        event_id=event.event_id,
                        event_type=event.event_type,
                        block=event.block,
                        accepted=False,
                        reason="topology_disabled_or_ablated",
                        priority=event.priority,
                        magnitude=event.magnitude,
                        metadata=event.metadata,
                    )
                )
            else:
                rejected_events.append(
                    StructuralEvent(
                        event_id=event.event_id,
                        event_type=event.event_type,
                        block=event.block,
                        accepted=False,
                        reason=event.reason,
                        priority=event.priority,
                        magnitude=event.magnitude,
                        metadata=event.metadata,
                    )
                )

        safety = apply_safe_transition(
            provisional_phi,
            provisional_cognition,
            previous_phi=state.phi.phi,
            config=self.config,
        )

        if safety.rejected:
            next_phi = state.phi.phi
            next_cognition = state.cognition.latent
        else:
            next_phi = safety.phi
            next_cognition = safety.cognition_latent

        if topology_changed:
            topology_version += 1
            target_nodes = int(topology_node_count)
            map_tuple = tuple(old_to_new_index_map)
            phi_fill = 0.0
            next_phi, _ = transport_with_index_map(
                "Phi_current",
                next_phi,
                old_to_new_index_map=map_tuple,
                new_size=target_nodes,
                fill_value=phi_fill,
            )
            provisional_phi, _ = transport_with_index_map(
                "Phi_provisional",
                provisional_phi,
                old_to_new_index_map=map_tuple,
                new_size=target_nodes,
                fill_value=phi_fill,
            )
            provisional_memory, _ = transport_with_index_map(
                "M_current",
                provisional_memory,
                old_to_new_index_map=map_tuple,
                new_size=target_nodes,
                fill_value=0.0,
            )
            provisional_hierarchy = np.zeros((max(1, target_nodes // 2), next_phi.shape[1]), dtype=next_phi.dtype)
        else:
            map_tuple = tuple(range(state.topology.node_count))
            target_nodes = state.topology.node_count

        if topology_changed:
            restriction, prolongation = _build_hierarchy_ops(target_nodes, state.dtype)
            coarse = restriction @ next_phi
            laplacian = _build_ring_laplacian(target_nodes, state.dtype, topology_metric_scale)
        else:
            restriction = state.hierarchy.restriction
            prolongation = state.hierarchy.prolongation
            coarse = provisional_hierarchy
            laplacian = state.geometry.laplacian

        transported_prior_phi, phi_transport = transport_with_index_map(
            "Phi",
            snapshot.state.phi.phi,
            old_to_new_index_map=map_tuple,
            new_size=target_nodes,
            fill_value=0.0,
        )
        transported_prior_memory, memory_transport = transport_with_index_map(
            "M",
            snapshot.state.memory.working,
            old_to_new_index_map=map_tuple,
            new_size=target_nodes,
            fill_value=0.0,
        )
        transported_prior_cognition, cognition_transport = transport_with_index_map("C", snapshot.state.cognition.latent)
        hierarchy_map = (
            _coarse_old_to_new_map(state.topology.node_count, target_nodes)
            if topology_changed
            else tuple(range(snapshot.state.hierarchy.coarse.shape[0]))
        )
        transported_prior_hierarchy, hierarchy_transport = transport_with_index_map(
            "H",
            snapshot.state.hierarchy.coarse,
            old_to_new_index_map=hierarchy_map,
            new_size=max(1, target_nodes // 2),
            fill_value=0.0,
        )
        observed_phi_delta = next_phi - transported_prior_phi

        applied_phi_contributions = dict(proposed_phi_contrib)
        if interactions is not None:
            applied_phi_contributions["cross_block_interactions"] = interactions

        applied_memory_contributions = dict(proposed_memory_contrib)
        applied_cognition_contributions = dict(proposed_cognition_contrib)
        applied_hierarchy_contributions = dict(proposed_hierarchy_contrib)

        if topology_changed:
            remapped_phi: dict[str, np.ndarray] = {}
            for key, value in applied_phi_contributions.items():
                if value.shape[0] == state.topology.node_count:
                    remapped_phi[key], _ = transport_with_index_map(
                        f"Phi_contrib_{key}",
                        value,
                        old_to_new_index_map=map_tuple,
                        new_size=target_nodes,
                        fill_value=0.0,
                    )
                elif value.shape[0] == 1 and target_nodes > 1:
                    remapped_phi[key] = np.repeat(value, target_nodes, axis=0)
                elif value.shape[0] == target_nodes:
                    remapped_phi[key] = value
                else:
                    remapped_phi[key] = np.resize(value, (target_nodes,) + value.shape[1:])
            applied_phi_contributions = remapped_phi

            remapped_memory: dict[str, np.ndarray] = {}
            for key, value in applied_memory_contributions.items():
                remapped_memory[key], _ = transport_with_index_map(
                    f"M_contrib_{key}",
                    value,
                    old_to_new_index_map=map_tuple,
                    new_size=target_nodes,
                    fill_value=0.0,
                )
            applied_memory_contributions = remapped_memory

            remapped_hierarchy: dict[str, np.ndarray] = {}
            hierarchy_map = _coarse_old_to_new_map(state.topology.node_count, target_nodes)
            for key, value in applied_hierarchy_contributions.items():
                remapped_hierarchy[key], _ = transport_with_index_map(
                    f"H_contrib_{key}",
                    value,
                    old_to_new_index_map=hierarchy_map,
                    new_size=max(1, target_nodes // 2),
                    fill_value=0.0,
                )
            applied_hierarchy_contributions = remapped_hierarchy

        correction_phi = next_phi - provisional_phi
        applied_phi_contributions["safety_correction"] = correction_phi

        reconstructed = np.zeros_like(observed_phi_delta)
        for value in applied_phi_contributions.values():
            reconstructed = reconstructed + value
        if topology_changed:
            topology_transport_adjustment = observed_phi_delta - reconstructed
            applied_phi_contributions["topology_transport_adjustment"] = topology_transport_adjustment
            reconstructed = reconstructed + topology_transport_adjustment
        residual = observed_phi_delta - reconstructed

        max_abs_error = float(np.max(np.abs(residual)))
        max_rel_error = float(np.linalg.norm(residual) / (np.linalg.norm(observed_phi_delta) + 1e-8))

        if max_rel_error > self.config.ledger_relative_tolerance:
            raise ValueError(
                f"Ledger reconstruction tolerance failed: {max_rel_error:.6e} > {self.config.ledger_relative_tolerance:.6e}"
            )

        rng_after = random_manager.digest()

        next_state = HRMState(
            version=state.version + 1,
            step=state.step + 1,
            dtype=state.dtype,
            device=state.device,
            rng_state=state.rng_state,
            phi=FieldState(phi=next_phi.astype(state.dtype, copy=False)),
            geometry=GeometryState(
                laplacian=laplacian.astype(state.dtype, copy=False),
                metric_scale=float(topology_metric_scale if topology_changed else state.geometry.metric_scale),
            ),
            topology=TopologyState(
                node_count=target_nodes if topology_changed else state.topology.node_count,
                edge_count=max(target_nodes, int(topology_edge_count)) if topology_changed else state.topology.edge_count,
                version=topology_version if topology_changed else state.topology.version,
            ),
            memory=MemoryState(
                working=provisional_memory.astype(state.dtype, copy=False),
                associative_keys=state.memory.associative_keys,
                associative_values=state.memory.associative_values,
                capacity=state.memory.capacity,
                write_index=write_index,
            ),
            cognition=CognitionState(
                latent=next_cognition.astype(state.dtype, copy=False),
                prediction=next_phi.mean(axis=0),
                residual=next_phi.mean(axis=0) - state.cognition.prediction,
                uncertainty=state.cognition.uncertainty,
            ),
            hierarchy=HierarchyState(
                coarse=coarse.astype(state.dtype, copy=False),
                restriction=restriction.astype(state.dtype, copy=False),
                prolongation=prolongation.astype(state.dtype, copy=False),
                gain=state.hierarchy.gain,
            ),
            budget=BudgetState(
                total_budget=state.budget.total_budget,
                remaining_budget=max(0.0, state.budget.remaining_budget - total_estimated_cost),
                active_width=state.budget.active_width,
                event_allowance=event_allowance,
                cumulative_cost=state.budget.cumulative_cost + total_estimated_cost,
            ),
        )

        next_invariant = validate_state(next_state, self.config)
        if not next_invariant.valid:
            raise ValueError(f"Invalid state after transition: {next_invariant.errors}")

        def _sum_contributions(values: dict[str, np.ndarray], template: np.ndarray) -> np.ndarray:
            total = np.zeros_like(np.asarray(template, dtype=np.float64))
            for value in values.values():
                total = total + np.asarray(value, dtype=np.float64)
            return total

        block_residuals = {
            "Phi": residual,
            "M": np.asarray(next_state.memory.working, dtype=np.float64) - np.asarray(transported_prior_memory, dtype=np.float64) - _sum_contributions(applied_memory_contributions, transported_prior_memory),
            "C": np.asarray(next_state.cognition.latent, dtype=np.float64) - np.asarray(transported_prior_cognition, dtype=np.float64) - _sum_contributions(applied_cognition_contributions, transported_prior_cognition),
            "H": np.asarray(next_state.hierarchy.coarse, dtype=np.float64) - np.asarray(transported_prior_hierarchy, dtype=np.float64) - _sum_contributions(applied_hierarchy_contributions, transported_prior_hierarchy),
        }

        for block_name, block_residual in list(block_residuals.items()):
            if np.max(np.abs(block_residual)) < 1e-6:
                block_residuals[block_name] = np.zeros_like(block_residual)

        ledger = TransitionLedger(
            source_version=state.version,
            target_version=next_state.version,
            proposal_activations=activations,
            proposed_phi_contributions=proposed_phi_contrib,
            applied_phi_contributions=applied_phi_contributions,
            rejected_phi_contributions={},
            proposed_memory_contributions=proposed_memory_contrib,
            applied_memory_contributions=applied_memory_contributions,
            rejected_memory_contributions={},
            proposed_cognition_contributions=proposed_cognition_contrib,
            applied_cognition_contributions=applied_cognition_contributions,
            rejected_cognition_contributions={},
            proposed_hierarchy_contributions=proposed_hierarchy_contrib,
            applied_hierarchy_contributions=applied_hierarchy_contributions,
            rejected_hierarchy_contributions={},
            proposed_events=proposed_events,
            accepted_events=tuple(accepted_events),
            rejected_events=tuple(rejected_events),
            transport_records=(phi_transport, memory_transport, cognition_transport, hierarchy_transport),
            corrections=safety.corrections,
            residual=residual,
            block_residuals=block_residuals,
            max_abs_reconstruction_error=max_abs_error,
            max_rel_reconstruction_error=max_rel_error,
            runtime_seconds=time.perf_counter() - start,
            rng_digest_before=rng_before,
            rng_digest_after=rng_after,
            metrics=state_metrics(next_state),
        )
        return TransitionResult(state=next_state, ledger=ledger)
