"""Bounded-loss CIRCA primitives for the V10 non-empirical extension.

The module generalizes the binary missing counterfactual from ``[0, 1]`` to a
registered bounded loss ``[0, B]``.  It contains no simulator, seed, experiment,
or scientific-output entry point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


class BoundedLossCircaError(ValueError):
    """Raised when a bounded-loss evidence object violates its contract."""


@dataclass(frozen=True)
class IntervalWitnessGroup:
    """Population-mass summary for one registered interval witness.

    ``lower_error_mass`` bounds the joint mass for which the asserted lower
    endpoint may be too high. ``upper_error_mass`` analogously bounds the joint
    mass for which the asserted upper endpoint may be too low.
    """

    name: str
    mass: float
    lower: float
    upper: float
    lower_error_mass: float = 0.0
    upper_error_mass: float = 0.0


@dataclass(frozen=True)
class RobustBoundedLossInterval:
    lower: float
    upper: float
    width: float
    general_lower: float
    general_upper: float
    general_width: float
    intervention_mass: float
    unresolved_mass: float
    tightening: float


def _indicator(name: str, values: Iterable[int]) -> np.ndarray:
    result = np.asarray(values, dtype=np.int8)
    if result.size == 0 or not np.all((result == 0) | (result == 1)):
        raise BoundedLossCircaError(f"{name} must be a nonempty binary array")
    return result


def _loss(name: str, values: Iterable[float], upper_bound: float) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise BoundedLossCircaError(f"{name} must be finite and nonempty")
    if np.any(result < 0.0) or np.any(result > upper_bound):
        raise BoundedLossCircaError(f"{name} must lie in [0, upper_bound]")
    return result


def _upper_bound(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise BoundedLossCircaError("upper_bound must be finite and positive")
    return result


def manski_bounded_loss_bounds(
    intervention: Iterable[int],
    active_loss: Iterable[float],
    *,
    upper_bound: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return trajectory-level general bounds for missing ``Z(0)``."""

    bound = _upper_bound(upper_bound)
    r = _indicator("intervention", intervention)
    z1 = _loss("active_loss", active_loss, bound)
    if r.shape != z1.shape:
        raise BoundedLossCircaError("intervention and active_loss shapes differ")
    lower = (1 - r) * z1
    upper = lower + r * bound
    return lower.astype(float), upper.astype(float)


def interval_witness_bounds(
    intervention: Iterable[int],
    active_loss: Iterable[float],
    witnessed: Iterable[int],
    witness_lower: Iterable[float],
    witness_upper: Iterable[float],
    *,
    upper_bound: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply registered interval witnesses to intervened trajectories.

    Non-witnessed entries must carry ``NaN`` endpoints. This prevents a caller
    from smuggling an unregistered point estimate into the unresolved group.
    """

    bound = _upper_bound(upper_bound)
    r = _indicator("intervention", intervention)
    z1 = _loss("active_loss", active_loss, bound)
    accepted = _indicator("witnessed", witnessed).astype(bool)
    lower_witness = np.asarray(witness_lower, dtype=float)
    upper_witness = np.asarray(witness_upper, dtype=float)
    if not (
        r.shape
        == z1.shape
        == accepted.shape
        == lower_witness.shape
        == upper_witness.shape
    ):
        raise BoundedLossCircaError("bounded-loss evidence shapes differ")
    if np.any(accepted & (r == 0)):
        raise BoundedLossCircaError(
            "non-intervened trajectories use observed consistency, not a witness"
        )
    unresolved = ~accepted
    if np.any(np.isfinite(lower_witness[unresolved])) or np.any(
        np.isfinite(upper_witness[unresolved])
    ):
        raise BoundedLossCircaError(
            "unresolved trajectories require NaN witness endpoints"
        )
    if np.any(~np.isfinite(lower_witness[accepted])) or np.any(
        ~np.isfinite(upper_witness[accepted])
    ):
        raise BoundedLossCircaError("accepted witness endpoints must be finite")
    if (
        np.any(lower_witness[accepted] < 0.0)
        or np.any(upper_witness[accepted] > bound)
        or np.any(lower_witness[accepted] > upper_witness[accepted])
    ):
        raise BoundedLossCircaError("invalid registered witness interval")

    lower, upper = manski_bounded_loss_bounds(
        r, z1, upper_bound=bound
    )
    lower[accepted] = lower_witness[accepted]
    upper[accepted] = upper_witness[accepted]
    return lower, upper


def bounded_loss_identification_interval(
    lower_z0: Iterable[float],
    upper_z0: Iterable[float],
    active_loss: Iterable[float],
    *,
    upper_bound: float,
) -> tuple[float, float]:
    """Average the bounded counterfactual-loss difference endpoints."""

    bound = _upper_bound(upper_bound)
    lower = _loss("lower_z0", lower_z0, bound)
    upper = _loss("upper_z0", upper_z0, bound)
    z1 = _loss("active_loss", active_loss, bound)
    if lower.shape != upper.shape or lower.shape != z1.shape:
        raise BoundedLossCircaError("identification arrays must be shape matched")
    if np.any(lower > upper):
        raise BoundedLossCircaError("lower_z0 exceeds upper_z0")
    return float(np.mean(lower - z1)), float(np.mean(upper - z1))


def robust_interval_from_groups(
    general_lower: float,
    general_upper: float,
    *,
    upper_bound: float,
    intervention_mass: float,
    groups: Sequence[IntervalWitnessGroup],
    tolerance: float = 1e-12,
) -> RobustBoundedLossInterval:
    """Inflate registered interval witnesses by directional joint error masses.

    The population partition contains the supplied witness groups plus an
    unresolved intervened group. Group intervals are constant within each
    registered group. The returned endpoints are sharp for this declared
    uncertainty class.
    """

    bound = _upper_bound(upper_bound)
    gl = float(general_lower)
    gu = float(general_upper)
    p = float(intervention_mass)
    tol = float(tolerance)
    if (
        not all(math.isfinite(value) for value in (gl, gu, p, tol))
        or gl > gu
        or p < 0.0
        or p > 1.0
        or tol < 0.0
    ):
        raise BoundedLossCircaError("invalid general interval arguments")
    expected_general_width = bound * p
    if abs((gu - gl) - expected_general_width) > tol:
        raise BoundedLossCircaError(
            "general interval width must equal upper_bound times intervention_mass"
        )

    total_group_mass = 0.0
    lower_increment = 0.0
    upper_decrement = 0.0
    declared_width = 0.0
    names: set[str] = set()
    for group in groups:
        if not group.name or group.name in names:
            raise BoundedLossCircaError("witness group names must be unique")
        names.add(group.name)
        values = (
            group.mass,
            group.lower,
            group.upper,
            group.lower_error_mass,
            group.upper_error_mass,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise BoundedLossCircaError("witness group values must be finite")
        q = float(group.mass)
        lo = float(group.lower)
        hi = float(group.upper)
        err_lo = float(group.lower_error_mass)
        err_hi = float(group.upper_error_mass)
        if (
            q < 0.0
            or lo < 0.0
            or hi > bound
            or lo > hi
            or err_lo < 0.0
            or err_hi < 0.0
            or err_lo > q
            or err_hi > q
        ):
            raise BoundedLossCircaError("invalid witness group")
        total_group_mass += q
        lower_increment += (q - err_lo) * lo
        upper_decrement += (q - err_hi) * (bound - hi)
        declared_width += (
            q * (hi - lo)
            + err_lo * lo
            + err_hi * (bound - hi)
        )

    if total_group_mass > p + tol:
        raise BoundedLossCircaError(
            "witness group mass exceeds intervention mass"
        )
    unresolved = max(0.0, p - total_group_mass)
    lower = max(gl, gl + lower_increment)
    upper = min(gu, gu - upper_decrement)
    width = upper - lower
    formula_width = unresolved * bound + declared_width
    if width < -tol or abs(width - formula_width) > tol:
        raise BoundedLossCircaError("robust width identity failed")
    width = max(0.0, width)
    general_width = gu - gl
    return RobustBoundedLossInterval(
        lower=lower,
        upper=upper,
        width=width,
        general_lower=gl,
        general_upper=gu,
        general_width=general_width,
        intervention_mass=p,
        unresolved_mass=unresolved,
        tightening=general_width - width,
    )


__all__ = [
    "BoundedLossCircaError",
    "IntervalWitnessGroup",
    "RobustBoundedLossInterval",
    "bounded_loss_identification_interval",
    "interval_witness_bounds",
    "manski_bounded_loss_bounds",
    "robust_interval_from_groups",
]
