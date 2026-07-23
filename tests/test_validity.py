import numpy as np
import pytest

from agc_runtime_assurance.contracts import ExpiredActionError
from agc_runtime_assurance.validity import ActionValidityCertificate, first_violation_time


def test_first_violation_time_uses_any_team_constraint():
    times = np.array([0.0, 0.1, 0.2, 0.3])
    margins = np.array([[1.0, 1.0], [0.8, 0.2], [0.5, -0.1], [-0.2, -0.3]])
    assert first_violation_time(times, margins) == pytest.approx(0.2)


def test_validity_horizon_subtracts_calibration_and_pipeline_debits():
    predicted = np.arange(1.0, 22.0)
    realized = predicted - 0.5
    certificate = ActionValidityCertificate.fit(predicted, realized, alpha=0.1)
    duration = certificate.certified_duration(
        3.0,
        observation_age=0.2,
        compute_delay=0.1,
        communication_delay=0.1,
        actuation_delay=0.1,
        guard_time=0.1,
    )
    assert duration == pytest.approx(1.9)


def test_zero_horizon_is_fail_closed_and_expiration_is_enforced():
    certificate = ActionValidityCertificate.fit(
        np.arange(1.0, 22.0), np.arange(1.0, 22.0) - 0.5, alpha=0.1
    )
    envelope = certificate.issue(
        np.zeros(2), issued_at=10.0, predicted_horizon=0.4,
        observation_age=0.2, compute_delay=0.1, communication_delay=0.1,
        actuation_delay=0.1, guard_time=0.1,
    )
    assert envelope.valid_until == 10.0
    assert envelope.constraint_state == "reject_zero_horizon"
    with pytest.raises(ExpiredActionError):
        envelope.checked_action(10.0001)


def test_calibration_fingerprint_is_order_invariant():
    predicted = np.arange(1.0, 22.0)
    realized = predicted - np.linspace(0.1, 0.8, predicted.size)
    a = ActionValidityCertificate.fit(predicted, realized, alpha=0.1)
    b = ActionValidityCertificate.fit(predicted[::-1], realized[::-1], alpha=0.1)
    assert a == b
    assert a.fingerprint() == b.fingerprint()


def test_training_conditional_certificate_is_no_less_conservative():
    predicted = np.arange(1.0, 202.0)
    realized = predicted - np.linspace(0.0, 1.0, predicted.size)
    marginal = ActionValidityCertificate.fit(predicted, realized, alpha=0.1)
    conditional = ActionValidityCertificate.fit_training_conditional(
        predicted, realized, alpha=0.1, delta=0.05
    )
    assert conditional.optimism_correction >= marginal.optimism_correction
    assert conditional.calibration_delta == 0.05


def test_training_conditional_certificate_refuses_when_sample_is_too_small():
    predicted = np.arange(1.0, 6.0)
    certificate = ActionValidityCertificate.fit_training_conditional(
        predicted, predicted - 0.1, alpha=0.05, delta=0.05
    )
    assert np.isinf(certificate.optimism_correction)
    assert certificate.certified_duration(
        10.0, observation_age=0.0, compute_delay=0.0,
        communication_delay=0.0, actuation_delay=0.0,
    ) == 0.0
