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
    HRMState,
    HierarchyState,
    MemoryState,
)
from .transport import record_identity_transport
from .mechanisms.base import HRMInput, HRMMechanism, TransitionContext


@dataclass(frozen=True)
class TransitionResult:
    state: HRMState
    ledger: TransitionLedger


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

        transported_prior_phi, phi_transport = record_identity_transport("Phi", snapshot.state.phi.phi)
        transported_prior_memory, memory_transport = record_identity_transport("M", snapshot.state.memory.working)
        transported_prior_cognition, cognition_transport = record_identity_transport("C", snapshot.state.cognition.latent)
        transported_prior_hierarchy, hierarchy_transport = record_identity_transport("H", snapshot.state.hierarchy.coarse)
        observed_phi_delta = next_phi - transported_prior_phi

        applied_phi_contributions = dict(proposed_phi_contrib)
        if interactions is not None:
            applied_phi_contributions["cross_block_interactions"] = interactions

        applied_memory_contributions = dict(proposed_memory_contrib)
        applied_cognition_contributions = dict(proposed_cognition_contrib)
        applied_hierarchy_contributions = dict(proposed_hierarchy_contrib)

        zero_memory_residual = np.zeros_like(provisional_memory)
        zero_cognition_residual = np.zeros_like(provisional_cognition)
        zero_hierarchy_residual = np.zeros_like(provisional_hierarchy)

        correction_phi = next_phi - provisional_phi
        applied_phi_contributions["safety_correction"] = correction_phi

        reconstructed = np.zeros_like(observed_phi_delta)
        for value in applied_phi_contributions.values():
            reconstructed = reconstructed + value
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
            geometry=state.geometry,
            topology=state.topology,
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
                coarse=provisional_hierarchy.astype(state.dtype, copy=False),
                restriction=state.hierarchy.restriction,
                prolongation=state.hierarchy.prolongation,
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
