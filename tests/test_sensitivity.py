import numpy as np
import pytest

from agc_runtime_assurance.sensitivity import (
    first_passage_sensitivity_bound,
    gronwall_state_deviation,
)


def test_zero_state_lipschitz_recovers_integrator_deviation_bound():
    deviation = gronwall_state_deviation(
        state_dynamics_lipschitz=0.0, action_dynamics_lipschitz=1.0,
        horizon=1.0, initial_state_error=0.05, action_error=0.1,
    )
    assert deviation == pytest.approx(0.15)


def test_gronwall_bound_handles_exponential_state_amplification():
    deviation = gronwall_state_deviation(
        state_dynamics_lipschitz=1.0, action_dynamics_lipschitz=2.0,
        horizon=0.5, initial_state_error=0.1, action_error=0.2,
    )
    expected = np.exp(0.5) * 0.1 + 2.0 * (np.exp(0.5) - 1.0) * 0.2
    assert deviation == pytest.approx(expected)


def test_team_bound_uses_worst_constraint_time_debit():
    bound = first_passage_sensitivity_bound(
        state_dynamics_lipschitz=0.0, action_dynamics_lipschitz=1.0,
        horizon=1.0, initial_state_error=0.0, action_error=0.1,
        constraint_lipschitz=np.array([1.0, 2.0]),
        transversality_kappa=np.array([1.0, 0.5]),
    )
    assert bound.valid
    assert np.allclose(bound.per_constraint_time_debits, [0.1, 0.4])
    assert bound.worst_time_debit == pytest.approx(0.4)


def test_grazing_crossing_fails_closed_with_infinite_debit():
    bound = first_passage_sensitivity_bound(
        state_dynamics_lipschitz=0.0, action_dynamics_lipschitz=1.0,
        horizon=1.0, initial_state_error=0.0, action_error=0.1,
        constraint_lipschitz=np.array([1.0]),
        transversality_kappa=np.array([0.0]),
    )
    assert not bound.valid
    assert np.isinf(bound.worst_time_debit)
    assert bound.reason == "grazing_or_unverified_transversality"


def test_integrator_first_passage_change_is_bounded_locally():
    # xdot=u, h=x, x0=1, reference u=-1 has T*=1 and kappa=1.
    # More negative u=-1.1 yields T*=1/1.1; the bound debits 0.1 s.
    bound = first_passage_sensitivity_bound(
        state_dynamics_lipschitz=0.0, action_dynamics_lipschitz=1.0,
        horizon=1.0, initial_state_error=0.0, action_error=0.1,
        constraint_lipschitz=np.array([1.0]),
        transversality_kappa=np.array([1.0]),
    )
    actual_earlier_shift = 1.0 - 1.0 / 1.1
    assert actual_earlier_shift <= bound.worst_time_debit
