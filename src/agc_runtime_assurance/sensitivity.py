"""Conservative constants for the candidate first-passage sensitivity theorem."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class FirstPassageSensitivityBound:
    trajectory_state_deviation: float
    per_constraint_time_debits: np.ndarray
    worst_time_debit: float
    horizon: float
    valid: bool
    reason: str


def gronwall_state_deviation(
    *,
    state_dynamics_lipschitz: float,
    action_dynamics_lipschitz: float,
    horizon: float,
    initial_state_error: float,
    action_error: float,
) -> float:
    """Bound trajectory separation for constant/piecewise-uniform action error."""

    values = np.asarray([
        state_dynamics_lipschitz, action_dynamics_lipschitz, horizon,
        initial_state_error, action_error,
    ], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Lipschitz constants, horizon, and errors must be finite and non-negative")
    lx, lu, time, state_error, control_error = map(float, values)
    if lx == 0.0:
        return state_error + lu * time * control_error
    growth = math.exp(lx * time)
    return growth * state_error + lu * (growth - 1.0) / lx * control_error


def first_passage_sensitivity_bound(
    *,
    state_dynamics_lipschitz: float,
    action_dynamics_lipschitz: float,
    horizon: float,
    initial_state_error: float,
    action_error: float,
    constraint_lipschitz: np.ndarray,
    transversality_kappa: np.ndarray,
) -> FirstPassageSensitivityBound:
    """Convert a joint trajectory error bound into a first-passage time debit.

    A positive kappa is a proof obligation: each potentially first-active team
    constraint must move toward its boundary at a uniform nonzero rate in the
    declared crossing tube.  Zero/negative kappa fails closed rather than
    manufacturing a finite deadline.
    """

    constraint = np.asarray(constraint_lipschitz, dtype=float).reshape(-1)
    kappa = np.asarray(transversality_kappa, dtype=float).reshape(-1)
    if constraint.size == 0 or constraint.shape != kappa.shape:
        raise ValueError("constraint Lipschitz and transversality arrays must align")
    if not np.all(np.isfinite(constraint)) or not np.all(np.isfinite(kappa)):
        raise ValueError("constraint constants must be finite")
    if np.any(constraint < 0.0):
        raise ValueError("constraint Lipschitz constants must be non-negative")
    deviation = gronwall_state_deviation(
        state_dynamics_lipschitz=state_dynamics_lipschitz,
        action_dynamics_lipschitz=action_dynamics_lipschitz,
        horizon=horizon,
        initial_state_error=initial_state_error,
        action_error=action_error,
    )
    if np.any(kappa <= 0.0):
        return FirstPassageSensitivityBound(
            deviation, np.full_like(kappa, np.inf), math.inf, float(horizon),
            False, "grazing_or_unverified_transversality",
        )
    debits = constraint * deviation / kappa
    return FirstPassageSensitivityBound(
        deviation, debits, float(np.max(debits)), float(horizon),
        True, "uniform_transversality_bound",
    )
