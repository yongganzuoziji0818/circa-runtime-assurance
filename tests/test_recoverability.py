import numpy as np

from agc_runtime_assurance.backup_invariant import LinearFeedbackInvariantBoxVerifier
from agc_runtime_assurance.latency import HandoverLatencyCertificate
from agc_runtime_assurance.recoverability import (
    MarginErosionRecoverabilityGate,
    VerifiedBackupRecoverabilityGate,
)
from agc_runtime_assurance.risk import ConstraintMargins


def test_gate_accounts_for_handover_and_backup_settling():
    gate = MarginErosionRecoverabilityGate(
        np.array([1.0, 0.5, 2.0, 0.0]), np.array([0.1, 0.1, 0.2, 0.0]),
        backup_settle_time=0.2,
    )
    result = gate.evaluate(
        ConstraintMargins(0.5, 0.4, 1.0, 10.0), handover_delay=0.1
    )
    assert result.recoverable
    assert np.allclose(result.required_margins, [0.4, 0.25, 0.8, 0.0])


def test_gate_fails_before_coupling_margin_is_consumed():
    gate = MarginErosionRecoverabilityGate(
        np.array([1.0, 0.5, 2.0, 0.0]), np.array([0.1, 0.1, 0.2, 0.0]),
        backup_settle_time=0.2,
    )
    result = gate.evaluate(
        ConstraintMargins(0.5, 0.4, 0.79, 10.0), handover_delay=0.1
    )
    assert not result.recoverable
    assert result.critical_constraint == 2


def test_longer_handover_can_only_make_gate_more_conservative():
    gate = MarginErosionRecoverabilityGate(
        np.ones(4), np.zeros(4), backup_settle_time=0.0
    )
    margins = ConstraintMargins(0.3, 0.3, 0.3, 0.3)
    assert gate.evaluate(margins, handover_delay=0.2).recoverable
    assert not gate.evaluate(margins, handover_delay=0.31).recoverable


def _backup_invariant(*, disturbance=0.05):
    return LinearFeedbackInvariantBoxVerifier(
        A=np.array([[1.0]]), B=np.array([[1.0]]), C=np.array([[1.0]]),
        K=np.array([[-0.5]]), equilibrium_state=np.array([0.0]),
        equilibrium_input=np.array([0.0]), invariant_radius=np.array([0.2]),
        disturbance_radius=np.array([disturbance]), state_lower=np.array([-1.0]),
        state_upper=np.array([1.0]), input_lower=np.array([-1.0]),
        input_upper=np.array([1.0]),
    ).verify()


def _latency():
    return HandoverLatencyCertificate(
        observation_age_bound=0.1, communication_bound=0.0,
        computation_bound=0.0, actuation_bound=0.0,
        dispatch_jitter_bound=0.0, guard_bound=0.0,
        source_fingerprint="c" * 64,
    )


def test_composed_gate_binds_recoverability_to_invariant_fingerprint():
    margin_gate = MarginErosionRecoverabilityGate(
        np.zeros(4), np.zeros(4), backup_settle_time=0.0
    )
    result = VerifiedBackupRecoverabilityGate(
        backup_invariant=_backup_invariant(), margin_gate=margin_gate
    ).evaluate(
        ConstraintMargins(1.0, 1.0, 1.0, 1.0), latency_certificate=_latency()
    )
    assert result.recoverable
    assert result.reason == "verified_backup_recoverable"
    assert len(result.backup_invariant_fingerprint) == 64


def test_composed_gate_fails_closed_for_unverified_backup_invariant():
    margin_gate = MarginErosionRecoverabilityGate(
        np.zeros(4), np.zeros(4), backup_settle_time=0.0
    )
    result = VerifiedBackupRecoverabilityGate(
        backup_invariant=_backup_invariant(disturbance=0.11), margin_gate=margin_gate
    ).evaluate(
        ConstraintMargins(1.0, 1.0, 1.0, 1.0), latency_certificate=_latency()
    )
    assert not result.recoverable
    assert result.reason.startswith("backup_invariant_not_verified")
