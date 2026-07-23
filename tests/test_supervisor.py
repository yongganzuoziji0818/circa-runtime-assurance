import json

import numpy as np
import pytest

from agc_runtime_assurance.audit import HashChainAuditLog
from agc_runtime_assurance.backup_invariant import LinearFeedbackInvariantBoxVerifier
from agc_runtime_assurance.contracts import AgentObservation
from agc_runtime_assurance.filtering import AffineSafetyFilter
from agc_runtime_assurance.fallback_mpc import (
    InvariantBoundFallbackTubeVerifier,
    LinearBoxFallbackTubeVerifier,
)
from agc_runtime_assurance.latency import HandoverLatencyCertificate
from agc_runtime_assurance.recoverability import (
    MarginErosionRecoverabilityGate,
    VerifiedBackupRecoverabilityGate,
)
from agc_runtime_assurance.risk import ConstraintMargins
from agc_runtime_assurance.state_machine import AssuranceMode, RuntimeAssuranceStateMachine
from agc_runtime_assurance.supervisor import RuntimeAssuranceSupervisor
from agc_runtime_assurance.validity import ActionValidityCertificate


def _observations(neighbor_age: float = 0.05):
    common = dict(
        monotonic_time=1.0, position=np.zeros(3), velocity=np.zeros(3),
        neighbor_age=neighbor_age, local_risk=0.1, interaction_risk=0.2,
        confidence=0.9, communication_delay=0.01, packet_loss=0.0,
        compute_budget=1.0,
    )
    return [
        AgentObservation(agent_id="u1", agent_kind="uav", **common),
        AgentObservation(agent_id="g1", agent_kind="ugv", **common),
    ]


def _supervisor(tmp_path):
    predicted = np.arange(1.0, 22.0)
    certificate = ActionValidityCertificate.fit(predicted, predicted - 0.1, alpha=0.1)
    audit = HashChainAuditLog(tmp_path / "decisions.jsonl")
    supervisor = RuntimeAssuranceSupervisor(
        validity_certificate=certificate,
        safety_filter=AffineSafetyFilter(np.array([-1.0]), np.array([1.0])),
        state_machine=RuntimeAssuranceStateMachine(recovery_hold_steps=2),
        audit_log=audit,
        max_neighbor_age=0.2,
        backup_validity=0.1,
    )
    return supervisor, audit


def _invariant(*, radius: float = 0.2):
    return LinearFeedbackInvariantBoxVerifier(
        A=np.array([[1.0]]), B=np.array([[1.0]]), C=np.array([[1.0]]),
        K=np.array([[-0.5]]), equilibrium_state=np.array([0.0]),
        equilibrium_input=np.array([0.0]), invariant_radius=np.array([radius]),
        disturbance_radius=np.array([0.05]), state_lower=np.array([-1.0]),
        state_upper=np.array([1.0]), input_lower=np.array([-1.0]),
        input_upper=np.array([1.0]),
    )


def _latency(*, observation_age_bound: float = 0.2, source: str = "d"):
    return HandoverLatencyCertificate(
        observation_age_bound=observation_age_bound, communication_bound=0.01,
        computation_bound=0.02, actuation_bound=0.02,
        dispatch_jitter_bound=0.005, guard_bound=0.05,
        source_fingerprint=source * 64,
    )


def _recoverability(
    coupling_margin: float = 1.0, *, radius: float = 0.2,
    latency_certificate=None,
):
    margin_gate = MarginErosionRecoverabilityGate(
        np.array([0.5, 0.5, 1.0, 0.0]), np.array([0.1, 0.1, 0.2, 0.0]),
        backup_settle_time=0.1,
    )
    backup_invariant = _invariant(radius=radius).verify()
    gate = VerifiedBackupRecoverabilityGate(
        backup_invariant=backup_invariant, margin_gate=margin_gate
    )
    return gate.evaluate(
        ConstraintMargins(1.0, 1.0, coupling_margin, 10.0),
        latency_certificate=latency_certificate or _latency(),
    )


def _fallback_plan(*, radius: float = 0.2):
    tube = LinearBoxFallbackTubeVerifier(
        A=np.array([[1.0]]), B=np.array([[1.0]]), C=np.array([[1.0]]),
        K=np.array([[-0.5]]), state_lower=np.array([-1.0]),
        state_upper=np.array([1.0]), input_lower=np.array([-1.0]),
        input_upper=np.array([1.0]), disturbance_radius=np.array([0.05]),
        estimation_error_radius=np.array([0.0]),
        recovery_lower=np.array([-radius]), recovery_upper=np.array([radius]),
    )
    return InvariantBoundFallbackTubeVerifier(
        tube_verifier=tube, invariant_verifier=_invariant(radius=radius)
    ).verify(
        state_estimate=np.array([0.0]), fallback_inputs=np.array([[0.0]]),
        nominal_first_input=np.array([0.0]),
    )


def test_safe_action_is_issued_with_calibrated_expiration(tmp_path):
    supervisor, audit = _supervisor(tmp_path)
    latency = _latency()
    result = supervisor.decide(
        observations=_observations(), nominal_action=np.array([0.2]),
        fallback_plan=_fallback_plan(), A=np.array([[1.0]]), b=np.array([0.5]),
        issued_at=5.0, predicted_horizon=1.0, latency_certificate=latency,
        recoverability=_recoverability(latency_certificate=latency),
    )
    assert result.transition.current == AssuranceMode.NOMINAL
    assert result.envelope.source == "nominal"
    assert 5.0 < result.envelope.valid_until < 6.0
    assert audit.verify()


def test_unsafe_nominal_is_filtered_before_expiration(tmp_path):
    supervisor, _ = _supervisor(tmp_path)
    latency = _latency()
    result = supervisor.decide(
        observations=_observations(), nominal_action=np.array([0.9]),
        fallback_plan=_fallback_plan(), A=np.array([[1.0]]), b=np.array([0.5]),
        issued_at=5.0, predicted_horizon=1.0, latency_certificate=latency,
        recoverability=_recoverability(latency_certificate=latency),
    )
    assert result.transition.current == AssuranceMode.FILTERED
    assert result.envelope.source == "safety_filter"
    assert np.allclose(result.envelope.action, [0.5], atol=1e-5)


def test_stale_observation_and_zero_horizon_fail_closed(tmp_path):
    supervisor, audit = _supervisor(tmp_path)
    latency = _latency(observation_age_bound=0.2)
    stale = supervisor.decide(
        observations=_observations(neighbor_age=0.3), nominal_action=np.array([0.2]),
        fallback_plan=_fallback_plan(), A=np.array([[1.0]]), b=np.array([0.5]),
        issued_at=5.0, predicted_horizon=1.0, latency_certificate=latency,
        recoverability=_recoverability(latency_certificate=latency),
    )
    assert stale.transition.current == AssuranceMode.BACKUP
    assert stale.envelope.source == "verified_backup"

    zero = supervisor.decide(
        observations=_observations(), nominal_action=np.array([0.2]),
        fallback_plan=_fallback_plan(), A=np.array([[1.0]]), b=np.array([0.5]),
        issued_at=6.0, predicted_horizon=0.05, latency_certificate=latency,
        recoverability=_recoverability(latency_certificate=latency),
    )
    assert zero.transition.current == AssuranceMode.BACKUP
    assert zero.transition.reason == "certificate_invalid"
    assert audit.verify()
    rows = [json.loads(line) for line in audit.path.read_text().splitlines()]
    assert rows[-1]["payload"]["source"] == "verified_backup"


def test_mechanical_recoverability_gate_preempts_statistical_deadline(tmp_path):
    supervisor, audit = _supervisor(tmp_path)
    latency = _latency()
    result = supervisor.decide(
        observations=_observations(), nominal_action=np.array([0.2]),
        fallback_plan=_fallback_plan(), A=np.array([[1.0]]), b=np.array([0.5]),
        issued_at=7.0, predicted_horizon=5.0, latency_certificate=latency,
        recoverability=_recoverability(
            coupling_margin=0.39, latency_certificate=latency
        ),
    )
    assert result.transition.current == AssuranceMode.BACKUP
    assert result.transition.reason == "recoverability_gate_triggered"
    row = json.loads(audit.path.read_text().splitlines()[-1])
    assert row["payload"]["recoverability_critical_constraint"] == 2
    assert len(row["payload"]["backup_invariant_fingerprint"]) == 64


def test_supervisor_refuses_mismatched_fallback_and_recoverability_fingerprints(tmp_path):
    supervisor, _ = _supervisor(tmp_path)
    latency = _latency()
    with pytest.raises(ValueError, match="fingerprints differ"):
        supervisor.decide(
            observations=_observations(), nominal_action=np.array([0.2]),
            fallback_plan=_fallback_plan(radius=0.3),
            A=np.array([[1.0]]), b=np.array([0.5]), issued_at=8.0,
            predicted_horizon=1.0, latency_certificate=latency,
            recoverability=_recoverability(radius=0.2, latency_certificate=latency),
        )


def test_supervisor_refuses_inconsistent_latency_evidence(tmp_path):
    supervisor, _ = _supervisor(tmp_path)
    validity_latency = _latency(source="d")
    recovery_latency = _latency(source="e")
    with pytest.raises(ValueError, match="latency fingerprints differ"):
        supervisor.decide(
            observations=_observations(), nominal_action=np.array([0.2]),
            fallback_plan=_fallback_plan(), A=np.array([[1.0]]), b=np.array([0.5]),
            issued_at=9.0, predicted_horizon=1.0,
            latency_certificate=validity_latency,
            recoverability=_recoverability(latency_certificate=recovery_latency),
        )
