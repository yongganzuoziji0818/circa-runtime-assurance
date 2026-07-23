import numpy as np

from agc_runtime_assurance.backup_invariant import LinearFeedbackInvariantBoxVerifier


def _verifier(*, A=1.0, disturbance=0.05, input_limit=1.0, equilibrium_state=0.0):
    return LinearFeedbackInvariantBoxVerifier(
        A=np.array([[A]]), B=np.array([[1.0]]), C=np.array([[1.0]]),
        K=np.array([[-0.5]]), equilibrium_state=np.array([equilibrium_state]),
        equilibrium_input=np.array([0.0]), invariant_radius=np.array([0.2]),
        disturbance_radius=np.array([disturbance]), state_lower=np.array([-1.0]),
        state_upper=np.array([1.0]), input_lower=np.array([-input_limit]),
        input_upper=np.array([input_limit]),
    )


def test_scalar_backup_box_is_robust_positive_invariant():
    result = _verifier().verify()
    assert result.verified
    assert result.reason == "linear_box_backup_invariant_verified"
    assert np.allclose(result.next_radius, [0.15])
    assert np.allclose(result.invariant_slack, [0.05])
    assert len(result.fingerprint) == 64


def test_disturbance_can_destroy_robust_positive_invariance():
    result = _verifier(disturbance=0.11).verify()
    assert not result.verified
    assert result.reason == "robust_positive_invariance_failed"


def test_backup_feedback_must_respect_input_constraints_on_entire_box():
    result = _verifier(input_limit=0.05).verify()
    assert not result.verified
    assert result.reason == "backup_input_constraints_failed"


def test_declared_center_must_be_an_equilibrium():
    result = _verifier(A=0.5, equilibrium_state=0.2).verify()
    assert not result.verified
    assert result.reason == "equilibrium_condition_failed"
