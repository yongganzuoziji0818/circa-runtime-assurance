"""Monotone fixed-point solvers for self-consistent transported horizons."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable

from .sensitivity import gronwall_state_deviation


@dataclass(frozen=True)
class SelfConsistentHorizon:
    valid: bool
    horizon: float
    trajectory_state_deviation: float
    barrier_deviation: float
    time_debit: float
    iterations: int
    reason: str


def _validate_common(
    *,
    reference_horizon: float,
    state_dynamics_lipschitz: float,
    action_dynamics_lipschitz: float,
    initial_state_error: float,
    action_error: float,
    constraint_lipschitz: float,
    tolerance: float,
    max_iterations: int,
) -> None:
    values = (
        reference_horizon,
        state_dynamics_lipschitz,
        action_dynamics_lipschitz,
        initial_state_error,
        action_error,
        constraint_lipschitz,
        tolerance,
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("fixed-point inputs must be finite")
    if reference_horizon <= 0.0:
        raise ValueError("reference_horizon must be positive")
    if any(value < 0.0 for value in values[1:6]):
        raise ValueError("Lipschitz constants and errors must be non-negative")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")


def _deviation_function(
    *,
    state_dynamics_lipschitz: float,
    action_dynamics_lipschitz: float,
    initial_state_error: float,
    action_error: float,
) -> Callable[[float], float]:
    def deviation(horizon: float) -> float:
        return gronwall_state_deviation(
            state_dynamics_lipschitz=state_dynamics_lipschitz,
            action_dynamics_lipschitz=action_dynamics_lipschitz,
            horizon=horizon,
            initial_state_error=initial_state_error,
            action_error=action_error,
        )

    return deviation


def solve_crossing_horizon(
    *,
    reference_horizon: float,
    state_dynamics_lipschitz: float,
    action_dynamics_lipschitz: float,
    initial_state_error: float,
    action_error: float,
    constraint_lipschitz: float,
    transversality_kappa: float,
    pre_tube_minimum_margin: float,
    crossing_tube_width: float,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> SelfConsistentHorizon:
    """Solve H + L_h D(H) / kappa <= reference_horizon from below."""

    _validate_common(
        reference_horizon=reference_horizon,
        state_dynamics_lipschitz=state_dynamics_lipschitz,
        action_dynamics_lipschitz=action_dynamics_lipschitz,
        initial_state_error=initial_state_error,
        action_error=action_error,
        constraint_lipschitz=constraint_lipschitz,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    if (
        not isfinite(transversality_kappa)
        or transversality_kappa <= 0.0
        or not isfinite(pre_tube_minimum_margin)
        or pre_tube_minimum_margin <= 0.0
        or not isfinite(crossing_tube_width)
        or crossing_tube_width <= 0.0
    ):
        raise ValueError("crossing proof constants must be positive and finite")

    deviation = _deviation_function(
        state_dynamics_lipschitz=state_dynamics_lipschitz,
        action_dynamics_lipschitz=action_dynamics_lipschitz,
        initial_state_error=initial_state_error,
        action_error=action_error,
    )
    coefficient = constraint_lipschitz / transversality_kappa

    def residual(horizon: float) -> float:
        return horizon + coefficient * deviation(horizon) - reference_horizon

    if residual(0.0) > 0.0:
        initial_deviation = deviation(0.0)
        return SelfConsistentHorizon(
            False,
            0.0,
            initial_deviation,
            constraint_lipschitz * initial_deviation,
            float("inf"),
            0,
            "initial_uncertainty_exhausts_reference_horizon",
        )

    if residual(reference_horizon) <= 0.0:
        horizon = reference_horizon
        iterations = 0
    else:
        lo, hi = 0.0, reference_horizon
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            mid = (lo + hi) / 2.0
            if residual(mid) <= 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo <= tolerance:
                break
        horizon = lo

    state_deviation = deviation(horizon)
    barrier_deviation = constraint_lipschitz * state_deviation
    time_debit = reference_horizon - horizon
    if barrier_deviation > pre_tube_minimum_margin:
        return SelfConsistentHorizon(
            False,
            0.0,
            state_deviation,
            barrier_deviation,
            time_debit,
            iterations,
            "pre_crossing_tube_margin_not_robust",
        )
    if time_debit > crossing_tube_width:
        return SelfConsistentHorizon(
            False,
            0.0,
            state_deviation,
            barrier_deviation,
            time_debit,
            iterations,
            "transversality_tube_too_narrow",
        )
    return SelfConsistentHorizon(
        True,
        horizon,
        state_deviation,
        barrier_deviation,
        time_debit,
        iterations,
        "self_consistent_crossing_horizon",
    )


def solve_censored_horizon(
    *,
    reference_horizon: float,
    state_dynamics_lipschitz: float,
    action_dynamics_lipschitz: float,
    initial_state_error: float,
    action_error: float,
    constraint_lipschitz: float,
    censored_minimum_margin: float,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> SelfConsistentHorizon:
    """Find the largest returned lower bound with L_h D(H) < margin."""

    _validate_common(
        reference_horizon=reference_horizon,
        state_dynamics_lipschitz=state_dynamics_lipschitz,
        action_dynamics_lipschitz=action_dynamics_lipschitz,
        initial_state_error=initial_state_error,
        action_error=action_error,
        constraint_lipschitz=constraint_lipschitz,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    if not isfinite(censored_minimum_margin) or censored_minimum_margin <= 0.0:
        raise ValueError("censored_minimum_margin must be positive and finite")

    deviation = _deviation_function(
        state_dynamics_lipschitz=state_dynamics_lipschitz,
        action_dynamics_lipschitz=action_dynamics_lipschitz,
        initial_state_error=initial_state_error,
        action_error=action_error,
    )

    def residual(horizon: float) -> float:
        return (
            constraint_lipschitz * deviation(horizon)
            - censored_minimum_margin
        )

    if residual(0.0) >= 0.0:
        initial_deviation = deviation(0.0)
        return SelfConsistentHorizon(
            False,
            0.0,
            initial_deviation,
            constraint_lipschitz * initial_deviation,
            float("inf"),
            0,
            "initial_uncertainty_exhausts_censored_margin",
        )

    if residual(reference_horizon) < 0.0:
        horizon = reference_horizon
        iterations = 0
    else:
        lo, hi = 0.0, reference_horizon
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            mid = (lo + hi) / 2.0
            if residual(mid) < 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo <= tolerance:
                break
        horizon = lo

    state_deviation = deviation(horizon)
    barrier_deviation = constraint_lipschitz * state_deviation
    return SelfConsistentHorizon(
        True,
        horizon,
        state_deviation,
        barrier_deviation,
        reference_horizon - horizon,
        iterations,
        "self_consistent_censored_horizon",
    )
