import pytest

from agc_runtime_assurance.horizon_solver import (
    solve_censored_horizon,
    solve_crossing_horizon,
)


def common() -> dict:
    return {
        "reference_horizon": 1.0,
        "state_dynamics_lipschitz": 0.0,
        "action_dynamics_lipschitz": 1.0,
        "initial_state_error": 0.0,
        "action_error": 0.1,
        "constraint_lipschitz": 1.0,
    }


def test_crossing_solver_recovers_integrator_fixed_point() -> None:
    result = solve_crossing_horizon(
        **common(),
        transversality_kappa=1.0,
        pre_tube_minimum_margin=0.2,
        crossing_tube_width=0.2,
    )

    assert result.valid
    assert result.horizon == pytest.approx(1.0 / 1.1, abs=1e-9)
    assert result.time_debit == pytest.approx(1.0 - 1.0 / 1.1, abs=1e-9)


def test_crossing_solver_fails_when_initial_error_exhausts_horizon() -> None:
    params = common()
    params["initial_state_error"] = 1.1
    params["action_error"] = 0.0
    result = solve_crossing_horizon(
        **params,
        transversality_kappa=1.0,
        pre_tube_minimum_margin=2.0,
        crossing_tube_width=2.0,
    )

    assert not result.valid
    assert result.horizon == 0.0
    assert result.reason == "initial_uncertainty_exhausts_reference_horizon"


def test_crossing_solver_preserves_tube_gate() -> None:
    result = solve_crossing_horizon(
        **common(),
        transversality_kappa=1.0,
        pre_tube_minimum_margin=0.2,
        crossing_tube_width=0.05,
    )

    assert not result.valid
    assert result.horizon == 0.0
    assert result.reason == "transversality_tube_too_narrow"


def test_censored_solver_keeps_full_horizon_when_margin_is_robust() -> None:
    result = solve_censored_horizon(
        **common(),
        censored_minimum_margin=0.2,
    )

    assert result.valid
    assert result.horizon == 1.0


def test_censored_solver_returns_strictly_feasible_side_of_margin_root() -> None:
    result = solve_censored_horizon(
        **common(),
        censored_minimum_margin=0.05,
    )

    assert result.valid
    assert result.horizon == pytest.approx(0.5, abs=1e-9)
    assert result.barrier_deviation < 0.05
