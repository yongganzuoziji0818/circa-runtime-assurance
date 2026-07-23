import numpy as np
import pytest

from agc_runtime_assurance.baselines import (
    ACoFiUpdateKernel,
    ConformalDecisionVariable,
    acofi_target,
    fixed_ttl_envelope,
    point_horizon_envelope,
)
from agc_runtime_assurance.contracts import ExpiredActionError


def test_fixed_ttl_has_no_hidden_calibration():
    envelope = fixed_ttl_envelope(np.array([1.0]), issued_at=2.0, ttl=0.5)
    assert envelope.valid_until == 2.5
    assert envelope.constraint_state == "fixed_ttl_no_statistical_certificate"


def test_point_horizon_subtracts_same_runtime_debits_as_tavh():
    envelope = point_horizon_envelope(
        np.array([1.0]), issued_at=2.0, predicted_horizon=1.0,
        observation_age=0.2, compute_delay=0.1, communication_delay=0.1,
        actuation_delay=0.1, guard_time=0.1,
    )
    assert envelope.valid_until == pytest.approx(2.4)


def test_zero_point_horizon_cannot_execute_even_at_issue_time():
    envelope = point_horizon_envelope(
        np.array([1.0]), issued_at=2.0, predicted_horizon=0.1,
        observation_age=0.2, compute_delay=0.0, communication_delay=0.0,
        actuation_delay=0.0,
    )
    with pytest.raises(ExpiredActionError):
        envelope.checked_action(2.0)


def test_acofi_target_and_one_sided_score_match_paper_equations():
    target = acofi_target(2.0, 1.0, gamma=0.9)
    assert target == pytest.approx(1.1)
    assert ACoFiUpdateKernel.one_sided_score(1.4, target) == pytest.approx(0.3)
    assert ACoFiUpdateKernel.one_sided_score(0.8, target) == 0.0


def test_acofi_bad_delayed_target_forces_infinite_small_sample_quantile():
    kernel = ACoFiUpdateKernel(target_alpha=0.1, learning_rate=0.05)
    assert kernel.observe(predicted_q=1.0, realized_target=0.5)
    assert np.isinf(kernel.quantile)
    assert not kernel.allow_task_action(
        predicted_task_q=100.0, local_margin=1.0, gamma=0.9, safety_threshold=0.1
    )


def test_acofi_switching_inequality_uses_all_threshold_terms():
    kernel = ACoFiUpdateKernel(target_alpha=0.5, learning_rate=0.1)
    kernel.quantile = 0.2
    assert kernel.allow_task_action(
        predicted_task_q=0.65, local_margin=0.5, gamma=0.5, safety_threshold=0.4
    )
    assert not kernel.allow_task_action(
        predicted_task_q=0.64, local_margin=0.5, gamma=0.5, safety_threshold=0.4
    )


def test_cdt_loss_above_target_makes_variable_more_conservative():
    variable = ConformalDecisionVariable(target_loss=0.1, learning_rate=0.5, value=0.0)
    assert variable.update(0.5) == pytest.approx(-0.2)
    assert variable.average_loss == pytest.approx(0.5)
    assert variable.update(0.0) == pytest.approx(-0.15)
    assert variable.average_loss == pytest.approx(0.25)


@pytest.mark.parametrize("bad_loss", [-0.1, 1.1, np.inf])
def test_cdt_rejects_invalid_loss(bad_loss):
    variable = ConformalDecisionVariable(target_loss=0.1, learning_rate=0.5, value=0.0)
    with pytest.raises(ValueError):
        variable.update(bad_loss)
