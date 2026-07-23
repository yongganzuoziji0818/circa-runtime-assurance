"""CIRCA identification and confidence primitives.

This module contains no experiment launch logic.  It implements the frozen
binary first-violation contract and fail-closed evidence checks used by the
exact-truth G0 fixture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


OBSERVED_CONSISTENCY = 1
INEVITABLE_VIOLATION = 2
NOMINAL_SAFETY = 3
NO_WITNESS = 0

VALID = "VALID"
INVALID_WITNESS = "INVALID_WITNESS"
INVALID_PROVENANCE = "INVALID_PROVENANCE"
OUTCOME_CENSORED_INVALIDLY = "OUTCOME_CENSORED_INVALIDLY"
INTERFERENCE_UNMODELED = "INTERFERENCE_UNMODELED"


class CircaError(ValueError):
    """Raised when a requested numerical certificate violates the contract."""


@dataclass(frozen=True)
class EvidenceCheck:
    status: str
    numeric_certificate_allowed: bool


def _binary(name: str, values: Iterable[int]) -> np.ndarray:
    result = np.asarray(values, dtype=np.int8)
    if not np.all((result == 0) | (result == 1)):
        raise CircaError(f"{name} must be binary")
    return result


def manski_bounds(r: Iterable[int], y1: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return trajectory-level lower/upper bounds for the censored Y0."""
    r_arr = _binary("R", r)
    y_arr = _binary("Y1", y1)
    if r_arr.shape != y_arr.shape:
        raise CircaError("R and Y1 shapes differ")
    lower = (1 - r_arr) * y_arr
    upper = lower + r_arr
    return lower.astype(float), upper.astype(float)


def structured_bounds(
    r: Iterable[int], y1: Iterable[int], witness: Iterable[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return CIRCA trajectory bounds after validating witness semantics."""
    r_arr = _binary("R", r)
    y_arr = _binary("Y1", y1)
    w_arr = np.asarray(witness, dtype=np.int8)
    if r_arr.shape != y_arr.shape or r_arr.shape != w_arr.shape:
        raise CircaError("R, Y1, and witness shapes differ")
    allowed = np.isin(w_arr, [NO_WITNESS, OBSERVED_CONSISTENCY, INEVITABLE_VIOLATION, NOMINAL_SAFETY])
    if not np.all(allowed):
        raise CircaError("unknown witness code")
    if np.any((r_arr == 0) & (w_arr != OBSERVED_CONSISTENCY)):
        raise CircaError("non-intervened trajectories require observed consistency")
    if np.any((r_arr == 1) & (w_arr == OBSERVED_CONSISTENCY)):
        raise CircaError("observed-consistency witness is invalid under intervention")

    lower, upper = manski_bounds(r_arr, y_arr)
    inevitable = (r_arr == 1) & (w_arr == INEVITABLE_VIOLATION)
    nominal_safe = (r_arr == 1) & (w_arr == NOMINAL_SAFETY)
    lower[inevitable] = 1.0
    upper[inevitable] = 1.0
    lower[nominal_safe] = 0.0
    upper[nominal_safe] = 0.0
    if np.any(lower > upper):
        raise CircaError("contradictory witness bounds")
    return lower, upper


def identification_interval(
    lower_y0: Iterable[float], upper_y0: Iterable[float], y1: Iterable[int]
) -> tuple[float, float]:
    lower = np.asarray(lower_y0, dtype=float)
    upper = np.asarray(upper_y0, dtype=float)
    y_arr = _binary("Y1", y1).astype(float)
    if lower.shape != upper.shape or lower.shape != y_arr.shape or lower.size == 0:
        raise CircaError("identification arrays must be nonempty and shape matched")
    if np.any(lower > upper) or np.any(lower < 0) or np.any(upper > 1):
        raise CircaError("invalid Y0 bounds")
    return float(np.mean(lower - y_arr)), float(np.mean(upper - y_arr))


def hoeffding_radius(n: int, alpha: float, endpoint_count: int = 8, value_range: float = 2.0) -> float:
    """One-sided Bonferroni Hoeffding radius for a bounded endpoint mean."""
    if n <= 0 or not (0 < alpha < 1) or endpoint_count <= 0 or value_range <= 0:
        raise CircaError("invalid confidence arguments")
    return value_range * math.sqrt(math.log(endpoint_count / alpha) / (2.0 * n))


def simultaneous_confidence_interval(
    identification_lower: float,
    identification_upper: float,
    n: int,
    alpha: float,
    endpoint_count: int = 8,
    value_range: float = 2.0,
) -> tuple[float, float]:
    if identification_lower > identification_upper:
        raise CircaError("lower endpoint exceeds upper endpoint")
    radius = hoeffding_radius(n, alpha, endpoint_count, value_range)
    return max(-1.0, identification_lower - radius), min(1.0, identification_upper + radius)


def verify_evidence_contract(
    *,
    interference_registered: bool = True,
    outcome_complete: bool = True,
    witness_hash_matches: bool = True,
    policy_hash_matches: bool = True,
    constraint_hash_matches: bool = True,
    horizon_matches: bool = True,
    witnesses_contradictory: bool = False,
) -> EvidenceCheck:
    """Validate evidence in the frozen fail-closed priority order."""
    if not interference_registered:
        return EvidenceCheck(INTERFERENCE_UNMODELED, False)
    if not outcome_complete:
        return EvidenceCheck(OUTCOME_CENSORED_INVALIDLY, False)
    if not witness_hash_matches or witnesses_contradictory:
        return EvidenceCheck(INVALID_WITNESS, False)
    if not policy_hash_matches or not constraint_hash_matches or not horizon_matches:
        return EvidenceCheck(INVALID_PROVENANCE, False)
    return EvidenceCheck(VALID, True)


def _betacf(a: float, b: float, x: float) -> float:
    max_iter, eps, fpmin = 300, 3.0e-14, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise CircaError("incomplete beta continued fraction did not converge")


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _beta_quantile(probability: float, a: float, b: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _regularized_beta(mid, a, b) < probability:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def clopper_pearson_lower(successes: int, trials: int, confidence: float = 0.95) -> float:
    if not (0 <= successes <= trials) or trials <= 0 or not (0 < confidence < 1):
        raise CircaError("invalid binomial confidence arguments")
    if successes == 0:
        return 0.0
    return _beta_quantile(1.0 - confidence, successes, trials - successes + 1)


def clopper_pearson_upper(successes: int, trials: int, confidence: float = 0.95) -> float:
    if not (0 <= successes <= trials) or trials <= 0 or not (0 < confidence < 1):
        raise CircaError("invalid binomial confidence arguments")
    if successes == trials:
        return 1.0
    return _beta_quantile(confidence, successes + 1, trials - successes)
