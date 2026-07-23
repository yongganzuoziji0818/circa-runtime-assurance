"""Executable G0 composition of validity, filtering, recovery, and audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .audit import HashChainAuditLog
from .contracts import ActionEnvelope, AgentObservation
from .fallback_mpc import InvariantBoundFallbackResult
from .filtering import AffineSafetyFilter, FilterResult, FilterStatus
from .latency import HandoverLatencyCertificate
from .recoverability import AssuredRecoverabilityResult
from .state_machine import AssuranceMode, RuntimeAssuranceStateMachine, Transition
from .validity import ActionValidityCertificate


@dataclass(frozen=True)
class AssuranceDecision:
    envelope: ActionEnvelope
    transition: Transition
    filter_result: FilterResult | None
    audit_hash: str


class RuntimeAssuranceSupervisor:
    """Fail-closed runtime composition for a fixed 1-UAV + 1-UGV interface.

    This class provides executable semantics, not a formal system-safety proof.
    It accepts only a feasible fallback tube whose terminal set and first action
    are bound to the same invariant fingerprint used by recoverability.  The
    validity and margin gates must also share one componentwise latency-certificate
    fingerprint.  These are externally evidenced contracts; their type and hash
    binding does not prove the real platform model.  Every branch emits a
    hash-linked audit record.
    """

    def __init__(
        self,
        *,
        validity_certificate: ActionValidityCertificate,
        safety_filter: AffineSafetyFilter,
        state_machine: RuntimeAssuranceStateMachine,
        audit_log: HashChainAuditLog,
        max_neighbor_age: float,
        backup_validity: float,
    ):
        if max_neighbor_age < 0.0 or backup_validity <= 0.0:
            raise ValueError("freshness threshold must be non-negative and backup validity positive")
        self.validity_certificate = validity_certificate
        self.safety_filter = safety_filter
        self.state_machine = state_machine
        self.audit_log = audit_log
        self.max_neighbor_age = float(max_neighbor_age)
        self.backup_validity = float(backup_validity)

    def decide(
        self,
        *,
        observations: Sequence[AgentObservation],
        nominal_action: np.ndarray,
        fallback_plan: InvariantBoundFallbackResult,
        A: np.ndarray,
        b: np.ndarray,
        issued_at: float,
        predicted_horizon: float,
        latency_certificate: HandoverLatencyCertificate,
        recoverability: AssuredRecoverabilityResult,
    ) -> AssuranceDecision:
        if not isinstance(recoverability, AssuredRecoverabilityResult):
            raise TypeError("an invariant-bound AssuredRecoverabilityResult is required")
        if not isinstance(fallback_plan, InvariantBoundFallbackResult):
            raise TypeError("an InvariantBoundFallbackResult is required")
        if not isinstance(latency_certificate, HandoverLatencyCertificate):
            raise TypeError("a HandoverLatencyCertificate is required")
        if not fallback_plan.feasible:
            raise ValueError("no certified fallback action is available; decision refused")
        if (
            fallback_plan.backup_invariant_fingerprint
            != recoverability.backup_invariant_fingerprint
        ):
            raise ValueError("fallback plan and recoverability invariant fingerprints differ")
        if latency_certificate.fingerprint() != recoverability.latency_fingerprint:
            raise ValueError("validity and recoverability latency fingerprints differ")
        fallback_action = fallback_plan.action
        backup_recoverable = recoverability.recoverable
        self._validate_team(observations)
        observation_age = max(float(obs.neighbor_age) for obs in observations)
        observations_fresh = (
            observation_age <= self.max_neighbor_age
            and latency_certificate.covers_observation_age(observation_age)
        )

        candidate = self.validity_certificate.issue(
            nominal_action,
            issued_at=issued_at,
            predicted_horizon=predicted_horizon,
            observation_age=observation_age,
            compute_delay=latency_certificate.computation_bound,
            communication_delay=latency_certificate.communication_bound,
            actuation_delay=latency_certificate.actuation_bound,
            guard_time=latency_certificate.execution_guard_bound,
        )
        certificate_valid = candidate.valid_until > candidate.issued_at

        filtered: FilterResult | None = None
        nominal_safe = False
        filter_feasible = False
        if observations_fresh and certificate_valid and backup_recoverable:
            filtered = self.safety_filter.filter(nominal_action, A, b, fallback_action)
            nominal_safe = filtered.status == FilterStatus.PASSTHROUGH
            filter_feasible = filtered.status in {FilterStatus.PASSTHROUGH, FilterStatus.FILTERED}

        transition = self.state_machine.step(
            certificate_valid=certificate_valid,
            nominal_safe=nominal_safe,
            filter_feasible=filter_feasible,
            backup_recoverable=backup_recoverable,
            observations_fresh=observations_fresh,
        )

        if transition.current == AssuranceMode.NOMINAL and filtered is not None:
            envelope = ActionEnvelope(
                filtered.action, issued_at, candidate.valid_until,
                "nominal", "validity_horizon_and_affine_constraints",
            )
        elif transition.current == AssuranceMode.FILTERED and filtered is not None:
            envelope = ActionEnvelope(
                filtered.action, issued_at, candidate.valid_until,
                "safety_filter", "validity_horizon_and_affine_constraints",
            )
        else:
            envelope = ActionEnvelope(
                np.asarray(fallback_action, dtype=float),
                issued_at,
                issued_at + self.backup_validity,
                "verified_backup",
                f"runtime_mode:{transition.current.value}",
            )

        payload = {
            "issued_at": float(issued_at),
            "mode": transition.current.value,
            "reason": transition.reason,
            "source": envelope.source,
            "valid_until": float(envelope.valid_until),
            "observation_age": observation_age,
            "certificate_fingerprint": self.validity_certificate.fingerprint(),
            "filter_status": filtered.status.value if filtered is not None else None,
            "filter_backend": filtered.backend if filtered is not None else None,
            "backup_recoverable": backup_recoverable,
            "recoverability_reason": recoverability.reason,
            "backup_invariant_fingerprint": recoverability.backup_invariant_fingerprint,
            "fallback_plan_reason": fallback_plan.reason,
            "latency_fingerprint": latency_certificate.fingerprint(),
            "handover_total_bound": latency_certificate.handover_total_bound,
            "recoverability_required_margins": recoverability.required_margins.tolist(),
            "recoverability_current_margins": recoverability.current_margins.tolist(),
            "recoverability_critical_constraint": recoverability.critical_constraint,
        }
        audit_hash = self.audit_log.append(payload)
        return AssuranceDecision(envelope, transition, filtered, audit_hash)

    @staticmethod
    def _validate_team(observations: Sequence[AgentObservation]) -> None:
        if len(observations) != 2:
            raise ValueError("P4 G0 supervisor requires exactly 1 UAV + 1 UGV")
        kinds = {obs.agent_kind for obs in observations}
        identifiers = {obs.agent_id for obs in observations}
        if kinds != {"uav", "ugv"} or len(identifiers) != 2:
            raise ValueError("team must contain distinct UAV and UGV agents")
        for observation in observations:
            observation.validate()
