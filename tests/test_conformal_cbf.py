import numpy as np
import pytest

from agc_runtime_assurance.conformal_cbf import (
    MultiAgentConformalCBF,
    conformal_cbf_interval_loss,
)
from agc_runtime_assurance.filtering import AffineSafetyFilter, FilterStatus


def test_interval_loss_is_worst_signed_gap_over_agents_and_time():
    predicted = np.array([[1.0, 0.0], [0.2, 0.4]])
    truth = np.array([[0.5, 0.2], [0.4, 0.4]])
    loss = conformal_cbf_interval_loss(predicted, truth, conformal_value=-0.1)
    assert loss == pytest.approx(np.arctan(0.4) / np.pi)


def test_lower_lambda_makes_conformal_cbf_constraint_stricter():
    controller = MultiAgentConformalCBF(
        safety_filter=AffineSafetyFilter(np.array([-1.0]), np.array([1.0])),
        target_loss=0.0, learning_rate=0.1, initial_value=-0.2,
    )
    step = controller.filter_action(
        nominal_action=np.array([0.0]), control_coefficients=np.array([[1.0]]),
        predicted_offsets=np.array([-0.5]), fallback_action=np.array([1.0]),
    )
    assert step.filter_result.status == FilterStatus.FILTERED
    assert step.filter_result.action[0] == pytest.approx(0.7, abs=1e-4)
    assert np.all(step.affine_A @ step.filter_result.action <= step.affine_b + 1e-7)


def test_positive_worst_gap_decreases_lambda_after_delayed_feedback():
    controller = MultiAgentConformalCBF(
        safety_filter=AffineSafetyFilter(np.array([-1.0]), np.array([1.0])),
        target_loss=-0.05, learning_rate=0.5, initial_value=0.0,
    )
    loss = controller.observe_interval(
        predicted_constraint_terms=np.array([0.5]),
        true_constraint_terms=np.array([0.0]),
    )
    assert loss > 0.0
    assert controller.variable.value < 0.0


def test_negative_target_loss_is_valid_for_signed_multiagent_loss():
    controller = MultiAgentConformalCBF(
        safety_filter=AffineSafetyFilter(np.array([-1.0]), np.array([1.0])),
        target_loss=-0.1, learning_rate=0.2, initial_value=0.0,
    )
    assert controller.variable.target_loss == -0.1
