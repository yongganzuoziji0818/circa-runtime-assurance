"""Prior-art betting intervals for a signed paired risk difference.

This module adapts the published MOPE and hedged-capital constructions to the
P4 estimand.  The statistical procedures are prior art, not P4 novelties.

MOPE reference (MIT, pinned separately in the Route-B manifest):
Karampatziakis, Mineiro, and Ramdas (ICML 2021), ``Off-Policy Confidence
Sequences``.  The small quadratic solver and wealth update below are adapted
from the authors' public ``opebet.py`` implementation.

Hedged-CI reference (MIT, pinned separately in the Route-B manifest):
Waudby-Smith and Ramdas (JRSSB 2024), ``Estimating means of bounded random
variables by betting``.  The fixed-time predictable bets and capital inversion
follow the authors' public ``confseq`` implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


class PairedRiskBettingError(ValueError):
    """Raised when a betting interval is statistically unevaluable."""


@dataclass(frozen=True)
class PairedBettingInterval:
    procedure: str
    sample_size: int
    delta_lower: float
    delta_upper: float
    transformed_lower: float
    transformed_upper: float
    family_alpha: float
    rho: float
    numerical_resolution: float


def _one_dimensional(name: str, values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise PairedRiskBettingError(f"{name} must be one-dimensional with at least two entries")
    if not np.all(np.isfinite(array)):
        raise PairedRiskBettingError(f"{name} contains a non-finite value")
    return array


def _paired_inputs(
    weights: Sequence[float] | np.ndarray,
    differences: Sequence[float] | np.ndarray,
    *,
    rho: float,
    family_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not math.isfinite(rho) or not 0.0 < rho < 1.0:
        raise PairedRiskBettingError("rho must lie strictly between zero and one")
    if not math.isfinite(family_alpha) or not 0.0 < family_alpha < 1.0:
        raise PairedRiskBettingError("family_alpha must lie in (0, 1)")
    w = _one_dimensional("weights", weights)
    d = _one_dimensional("differences", differences)
    if w.shape != d.shape:
        raise PairedRiskBettingError("weights and differences must be exactly paired")
    if np.any(w < 0.0) or np.any(w > 1.0 / rho + 1e-12):
        raise PairedRiskBettingError("weights violate the defensive-mixture interval")
    if not np.all((d == -1.0) | (d == 0.0) | (d == 1.0)):
        raise PairedRiskBettingError("differences must lie in {-1, 0, 1}")
    return w, d


def signed_to_off_policy_reward(differences: Sequence[float] | np.ndarray) -> np.ndarray:
    """Map D in {-1,0,1} to the bounded OPE reward r=(D+1)/2."""

    d = _one_dimensional("differences", differences)
    if not np.all((d == -1.0) | (d == 0.0) | (d == 1.0)):
        raise PairedRiskBettingError("differences must lie in {-1, 0, 1}")
    return (d + 1.0) / 2.0


def signed_weighted_to_bounded_mean(
    weights: Sequence[float] | np.ndarray,
    differences: Sequence[float] | np.ndarray,
    *,
    rho: float,
) -> np.ndarray:
    """Map Z=wD in [-1/rho,1/rho] to U=(rho Z+1)/2 in [0,1]."""

    w, d = _paired_inputs(weights, differences, rho=rho, family_alpha=0.5)
    bounded = (rho * w * d + 1.0) / 2.0
    if np.any(bounded < -1e-12) or np.any(bounded > 1.0 + 1e-12):
        raise PairedRiskBettingError("affine bounded-mean mapping left [0, 1]")
    return np.clip(bounded, 0.0, 1.0)


def _update_stats(stats: dict[str, float], w: float, r: float) -> None:
    stats["n"] += 1.0
    stats["z"] += w * r
    stats["y"] += w * w * r * r
    stats["x"] += w * w
    stats["w"] += w
    stats["v"] += w * w * r


def _fill_quadratic(stats: dict[str, float], value: float) -> tuple[np.ndarray, np.ndarray]:
    scale = 8.0 * math.log(2.0) - 4.0
    matrix = np.empty((2, 2), dtype=float)
    vector = np.empty(2, dtype=float)
    matrix[0, 0] = scale * (stats["x"] + stats["n"] - 2.0 * stats["w"])
    matrix[0, 1] = scale * (
        stats["v"] - stats["z"] + value * (stats["n"] - stats["w"])
    )
    matrix[1, 0] = matrix[0, 1]
    matrix[1, 1] = scale * (
        stats["y"] - 2.0 * value * stats["z"] + value * value * stats["n"]
    )
    vector[0] = stats["w"] - stats["n"]
    vector[1] = stats["z"] - value * stats["n"]
    return matrix, vector


@dataclass
class _MopeSolverState:
    candidates: np.ndarray
    unconstrained: np.ndarray


def _new_mope_solver_state(wmax: float) -> _MopeSolverState:
    candidates = np.zeros((6, 2), dtype=float)
    candidates[3] = [0.5, 0.0]
    candidates[4] = [0.0, 0.5]
    candidates[5] = [-0.5 / (wmax - 1.0), 0.0]
    return _MopeSolverState(candidates=candidates, unconstrained=np.zeros(2, dtype=float))


def _solve_mope_quadratic(
    matrix: np.ndarray,
    vector: np.ndarray,
    wmax: float,
    state: _MopeSolverState,
) -> np.ndarray:
    """Pure-NumPy transcription of the pinned MOPE two-dimensional solver."""

    constraints = np.array(
        [[0.0, -1.0, wmax - 1.0], [1.0, -1.0, -1.0]], dtype=float
    )
    offsets = np.array([0.0, -0.5, -0.5], dtype=float)
    candidates = state.candidates

    a = matrix.copy()
    determinant = a[0, 0] * a[1, 1] - a[1, 0] * a[0, 1]
    if a[0, 0] <= 0.0 or determinant <= 0.0:
        fixed = candidates[3:]
        objective = 0.5 * np.sum((fixed @ a) * fixed, axis=1) - fixed @ vector
        return fixed[int(np.argmin(objective))]

    a[0, 0] = math.sqrt(a[0, 0])
    a[1, 0] /= a[0, 0]
    a[1, 1] -= a[1, 0] * a[1, 0]
    if a[1, 1] <= 0.0:
        state.unconstrained[:] = 0.0
        return state.unconstrained
    a[1, 1] = math.sqrt(a[1, 1])
    a[0, 1] = 0.0

    first = vector[0] / a[0, 0]
    unconstrained = state.unconstrained
    unconstrained[1] = (vector[1] - a[1, 0] * first) / (a[1, 1] * a[1, 1])
    unconstrained[0] = (first - a[1, 0] * unconstrained[1]) / a[0, 0]
    slack = unconstrained @ constraints - offsets
    if np.all(slack >= 0.0):
        return unconstrained

    inverse_times_constraint = np.zeros_like(constraints)
    for index in range(3):
        first = constraints[0, index] / a[0, 0]
        inverse_times_constraint[1, index] = (
            constraints[1, index] - a[1, 0] * first
        ) / (a[1, 1] * a[1, 1])
        inverse_times_constraint[0, index] = (
            first - a[1, 0] * inverse_times_constraint[1, index]
        ) / a[0, 0]
        denominator = float(inverse_times_constraint[:, index] @ constraints[:, index])
        if abs(denominator) > 1e-15:
            multiplier = -slack[index] / denominator
            candidates[index] = unconstrained + multiplier * inverse_times_constraint[:, index]

    if candidates[0, 0] < -0.5 / (wmax - 1.0) or candidates[0, 0] > 0.5:
        candidates[0] = 0.0
    if candidates[1, 0] < 0.0 or candidates[1, 1] < 0.0:
        candidates[1] = 0.0
    if candidates[2, 0] > 0.0 or candidates[2, 1] < 0.0:
        candidates[2] = 0.0

    transformed = candidates @ a
    objective = 0.5 * np.sum(transformed * transformed, axis=1) - candidates @ vector
    return candidates[int(np.argmin(objective))]


def _first_accepted(grid: np.ndarray, log_wealth: np.ndarray, threshold: float) -> float:
    indices = np.flatnonzero(log_wealth < threshold)
    return 0.0 if indices.size == 0 else float(grid[int(indices[0])])


def mope_paired_interval(
    weights: Sequence[float] | np.ndarray,
    differences: Sequence[float] | np.ndarray,
    *,
    rho: float,
    family_alpha: float,
    grid_size: int = 501,
) -> PairedBettingInterval:
    """Return the pinned MOPE grid interval mapped to Delta=2v-1."""

    w, d = _paired_inputs(weights, differences, rho=rho, family_alpha=family_alpha)
    if not isinstance(grid_size, int) or grid_size < 101:
        raise PairedRiskBettingError("grid_size must be an integer of at least 101")
    reward = (d + 1.0) / 2.0
    values = np.linspace(0.0, 1.0, grid_size)
    lower_value = 0.0
    upper_complement = 0.0
    positive_stats = {name: 0.0 for name in ("n", "z", "y", "x", "w", "v")}
    negative_stats = {name: 0.0 for name in ("n", "z", "y", "x", "w", "v")}
    positive_bet = np.zeros(2, dtype=float)
    negative_bet = np.zeros(2, dtype=float)
    log_positive = np.full(grid_size, math.log(0.5), dtype=float)
    log_negative = np.full(grid_size, math.log(0.5), dtype=float)
    threshold = math.log(1.0 / family_alpha)
    wmax = 1.0 / rho
    solver_state = _new_mope_solver_state(wmax)

    for wi, ri in zip(w, reward):
        positive_increment = positive_bet[0] * (wi - 1.0) + positive_bet[1] * (wi * ri - values)
        negative_increment = negative_bet[0] * (wi - 1.0) + negative_bet[1] * (
            wi * (1.0 - ri) - values
        )
        if np.any(positive_increment <= -1.0) or np.any(negative_increment <= -1.0):
            raise PairedRiskBettingError("MOPE bet violated nonnegative-wealth constraints")
        log_positive += np.log1p(positive_increment)
        log_negative += np.log1p(negative_increment)
        new_lower = _first_accepted(values, log_positive, threshold)
        new_upper_complement = _first_accepted(values, log_negative, threshold)
        lower_value = min(1.0 - upper_complement, max(lower_value, new_lower))
        upper_complement = min(1.0 - lower_value, max(upper_complement, new_upper_complement))
        _update_stats(positive_stats, float(wi), float(ri))
        _update_stats(negative_stats, float(wi), float(1.0 - ri))
        positive_bet = _solve_mope_quadratic(
            *_fill_quadratic(positive_stats, lower_value), wmax, solver_state
        )
        negative_bet = _solve_mope_quadratic(
            *_fill_quadratic(negative_stats, upper_complement), wmax, solver_state
        )

    upper_value = 1.0 - upper_complement
    resolution = 1.0 / (grid_size - 1)
    return PairedBettingInterval(
        procedure="mope_off_policy_cs",
        sample_size=int(w.size),
        delta_lower=max(-1.0, 2.0 * lower_value - 1.0),
        delta_upper=min(1.0, 2.0 * upper_value - 1.0),
        transformed_lower=lower_value,
        transformed_upper=upper_value,
        family_alpha=float(family_alpha),
        rho=float(rho),
        numerical_resolution=2.0 * resolution,
    )


def _fixed_time_predmix_bets(x: np.ndarray, alpha: float) -> np.ndarray:
    n = int(x.size)
    time_index = np.arange(1, n + 1, dtype=float)
    regularized_mean = np.minimum((0.5 + np.cumsum(x)) / (time_index + 1.0), 1.0)
    previous_mean = np.concatenate(([0.5], regularized_mean[:-1]))
    regularized_variance = (
        0.25 + np.cumsum(np.square(x - regularized_mean))
    ) / (time_index + 1.0)
    previous_variance = np.concatenate(([0.25], regularized_variance[:-1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        bets = np.sqrt(2.0 * math.log(1.0 / alpha) / (n * previous_variance))
    bets[~np.isfinite(bets)] = 1.0
    return np.clip(bets, -1.0, 1.0)


def _hedged_log_capital(
    x: np.ndarray,
    candidate: float,
    positive_bets: np.ndarray,
    negative_bets: np.ndarray,
    theta: float,
) -> float:
    positive_terms = 1.0 + positive_bets * (x - candidate)
    negative_terms = 1.0 - negative_bets * (x - candidate)
    if np.any(positive_terms < 0.0) or np.any(negative_terms < 0.0):
        raise PairedRiskBettingError("hedged bet produced negative capital")
    with np.errstate(divide="ignore"):
        log_positive = math.log(theta) + float(np.sum(np.log(positive_terms)))
        log_negative = math.log(1.0 - theta) + float(np.sum(np.log(negative_terms)))
    return max(log_positive, log_negative)


def _conservative_root(
    accepted,
    center: float,
    *,
    lower: bool,
    iterations: int,
) -> float:
    if lower:
        if accepted(0.0):
            return 0.0
        rejected_side, accepted_side = 0.0, center
        for _ in range(iterations):
            midpoint = (rejected_side + accepted_side) / 2.0
            if accepted(midpoint):
                accepted_side = midpoint
            else:
                rejected_side = midpoint
        return rejected_side
    if accepted(1.0):
        return 1.0
    accepted_side, rejected_side = center, 1.0
    for _ in range(iterations):
        midpoint = (accepted_side + rejected_side) / 2.0
        if accepted(midpoint):
            accepted_side = midpoint
        else:
            rejected_side = midpoint
    return rejected_side


def hedged_bounded_mean_interval(
    weights: Sequence[float] | np.ndarray,
    differences: Sequence[float] | np.ndarray,
    *,
    rho: float,
    family_alpha: float,
    root_iterations: int = 48,
) -> PairedBettingInterval:
    """Return a fixed-time Hedged-CI for U=(rho*w*D+1)/2, mapped to Delta."""

    w, d = _paired_inputs(weights, differences, rho=rho, family_alpha=family_alpha)
    if not isinstance(root_iterations, int) or root_iterations < 24:
        raise PairedRiskBettingError("root_iterations must be an integer of at least 24")
    bounded = (rho * w * d + 1.0) / 2.0
    theta = 0.5
    positive_bets = _fixed_time_predmix_bets(bounded, family_alpha * theta)
    negative_bets = _fixed_time_predmix_bets(bounded, family_alpha * (1.0 - theta))
    threshold = math.log(1.0 / family_alpha)

    def accepted(candidate: float) -> bool:
        return _hedged_log_capital(
            bounded, candidate, positive_bets, negative_bets, theta
        ) <= threshold

    center = float(np.mean(bounded))
    if not accepted(center):
        search = np.linspace(0.0, 1.0, 1001)
        accepted_values = [float(value) for value in search if accepted(float(value))]
        if not accepted_values:
            raise PairedRiskBettingError("hedged confidence set is numerically empty")
        center = accepted_values[len(accepted_values) // 2]
    lower_mean = _conservative_root(
        accepted, center, lower=True, iterations=root_iterations
    )
    upper_mean = _conservative_root(
        accepted, center, lower=False, iterations=root_iterations
    )
    scale = 2.0 / rho
    return PairedBettingInterval(
        procedure="hedged_bounded_mean_ci",
        sample_size=int(w.size),
        delta_lower=max(-1.0, (2.0 * lower_mean - 1.0) / rho),
        delta_upper=min(1.0, (2.0 * upper_mean - 1.0) / rho),
        transformed_lower=lower_mean,
        transformed_upper=upper_mean,
        family_alpha=float(family_alpha),
        rho=float(rho),
        numerical_resolution=scale * 2.0 ** (-root_iterations),
    )
