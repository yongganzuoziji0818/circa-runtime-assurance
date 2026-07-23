import numpy as np

from agc_runtime_assurance.backup_invariant import LinearFeedbackInvariantBoxVerifier
from agc_runtime_assurance.fallback_mpc import (
    InvariantBoundFallbackTubeVerifier,
    LinearBoxFallbackTubeVerifier,
)


def _invariant(*, A=1.0, disturbance=0.0):
    return LinearFeedbackInvariantBoxVerifier(
        A=np.array([[A]]), B=np.array([[1.0]]), C=np.array([[1.0]]),
        K=np.array([[-0.5]]), equilibrium_state=np.array([0.0]),
        equilibrium_input=np.array([0.0]), invariant_radius=np.array([0.2]),
        disturbance_radius=np.array([disturbance]), state_lower=np.array([-2.0]),
        state_upper=np.array([2.0]), input_lower=np.array([-1.0]),
        input_upper=np.array([1.0]),
    )


def _tube(*, A=1.0, recovery_radius=0.2, disturbance=0.0):
    return LinearBoxFallbackTubeVerifier(
        A=np.array([[A]]), B=np.array([[1.0]]), C=np.array([[1.0]]),
        K=np.array([[-0.5]]), state_lower=np.array([-2.0]),
        state_upper=np.array([2.0]), input_lower=np.array([-1.0]),
        input_upper=np.array([1.0]), disturbance_radius=np.array([disturbance]),
        estimation_error_radius=np.array([0.0]),
        recovery_lower=np.array([-recovery_radius]),
        recovery_upper=np.array([recovery_radius]),
    )


def test_fallback_tube_is_bound_to_matching_verified_invariant_box():
    result = InvariantBoundFallbackTubeVerifier(
        tube_verifier=_tube(), invariant_verifier=_invariant()
    ).verify(
        state_estimate=np.array([1.0]),
        fallback_inputs=np.array([[-0.5], [-0.25], [-0.125]]),
        nominal_first_input=np.array([-0.5]),
    )
    assert result.feasible
    assert result.reason == "invariant_bound_fallback_tube_feasible"
    assert np.allclose(result.action, [-0.5])
    assert len(result.backup_invariant_fingerprint) == 64


def test_recovery_box_mismatch_is_rejected_before_tube_claim():
    result = InvariantBoundFallbackTubeVerifier(
        tube_verifier=_tube(recovery_radius=0.3), invariant_verifier=_invariant()
    ).verify(
        state_estimate=np.array([0.0]), fallback_inputs=np.array([[0.0]]),
        nominal_first_input=np.array([0.0]),
    )
    assert not result.feasible
    assert result.reason == "backup_invariant_contract_mismatch:recovery_lower"
    assert result.tube_result is None


def test_dynamics_mismatch_is_rejected_before_tube_claim():
    result = InvariantBoundFallbackTubeVerifier(
        tube_verifier=_tube(A=0.9), invariant_verifier=_invariant(A=1.0)
    ).verify(
        state_estimate=np.array([0.0]), fallback_inputs=np.array([[0.0]]),
        nominal_first_input=np.array([0.0]),
    )
    assert not result.feasible
    assert result.reason == "backup_invariant_contract_mismatch:A"


def test_unverified_invariant_cannot_certify_a_feasible_terminal_tube():
    result = InvariantBoundFallbackTubeVerifier(
        tube_verifier=_tube(disturbance=0.11),
        invariant_verifier=_invariant(disturbance=0.11),
    ).verify(
        state_estimate=np.array([0.0]), fallback_inputs=np.array([[0.0]]),
        nominal_first_input=np.array([0.0]),
    )
    assert not result.feasible
    assert result.reason.startswith("backup_invariant_not_verified")
    assert result.tube_result is None
