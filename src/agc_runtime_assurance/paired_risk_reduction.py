"""Fail-closed statistics for paired nominal-law risk-reduction estimates.

This module contains engineering-level implementations of classical importance-
sampling and concentration identities.  It does not run a benchmark and its unit
tests are not scientific efficacy evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


class PairedRiskReductionError(ValueError):
    """Raised when a paired risk certificate is not statistically evaluable."""


@dataclass(frozen=True)
class FamilyRiskReductionCertificate:
    family: str
    sample_size: int
    estimate: float
    sample_variance: float
    empirical_bernstein_lower: float
    hoeffding_lower: float
    baseline_only_failures: int
    full_only_failures: int
    shared_failures: int
    shared_safe: int
    generic_weight_ess: float
    maximum_weight: float
    rho: float
    family_alpha: float


@dataclass(frozen=True)
class SimultaneousRiskReductionCertificate:
    registered_families: tuple[str, ...]
    family_certificates: tuple[FamilyRiskReductionCertificate, ...]
    empirical_bernstein_lower_min: float
    hoeffding_lower_min: float
    confidence: float
    minimum_relevant_reduction: float
    certified: bool


def _one_dimensional(name: str, values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise PairedRiskReductionError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise PairedRiskReductionError(f"{name} contains a non-finite value")
    return array


def _binary(name: str, values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = _one_dimensional(name, values)
    if not np.all((array == 0.0) | (array == 1.0)):
        raise PairedRiskReductionError(f"{name} must contain only binary indicators")
    return array.astype(np.int8)


def importance_weights_from_log_densities(
    log_nominal: Sequence[float] | np.ndarray,
    log_proposal: Sequence[float] | np.ndarray,
    *,
    rho: float,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Return exact likelihood ratios and enforce the defensive-mixture bound."""

    if not 0.0 < rho <= 1.0 or not math.isfinite(rho):
        raise PairedRiskReductionError("rho must lie in (0, 1]")
    nominal = _one_dimensional("log_nominal", log_nominal)
    proposal = _one_dimensional("log_proposal", log_proposal)
    if nominal.shape != proposal.shape or nominal.size == 0:
        raise PairedRiskReductionError("log-density arrays must be non-empty and paired")
    log_weights = nominal - proposal
    maximum_log_weight = -math.log(rho)
    if np.any(log_weights > maximum_log_weight + tolerance):
        raise PairedRiskReductionError("likelihood ratio exceeds the defensive-mixture bound")
    weights = np.exp(log_weights)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise PairedRiskReductionError("invalid likelihood ratio")
    return weights


def family_risk_reduction_certificate(
    family: str,
    weights: Sequence[float] | np.ndarray,
    baseline_failures: Sequence[float] | np.ndarray,
    full_failures: Sequence[float] | np.ndarray,
    *,
    rho: float,
    family_alpha: float,
) -> FamilyRiskReductionCertificate:
    """Compute a fixed-sample one-sided paired risk-reduction certificate.

    The empirical-Bernstein expression is the signed affine specialization of
    Maurer-Pontil's bounded-variable result used by Dietrich et al. (2026).
    """

    if not isinstance(family, str) or not family:
        raise PairedRiskReductionError("family must be a non-empty string")
    if not 0.0 < rho <= 1.0 or not math.isfinite(rho):
        raise PairedRiskReductionError("rho must lie in (0, 1]")
    if not 0.0 < family_alpha < 1.0 or not math.isfinite(family_alpha):
        raise PairedRiskReductionError("family_alpha must lie in (0, 1)")

    w = _one_dimensional("weights", weights)
    baseline = _binary("baseline_failures", baseline_failures)
    full = _binary("full_failures", full_failures)
    if w.shape != baseline.shape or w.shape != full.shape or w.size < 2:
        raise PairedRiskReductionError("at least two exactly paired evaluation paths are required")
    if np.any(w < 0.0) or np.any(w > 1.0 / rho + 1e-12):
        raise PairedRiskReductionError("weights violate the defensive-mixture interval")

    differences = baseline.astype(float) - full.astype(float)
    weighted = w * differences
    n = int(w.size)
    estimate = float(np.mean(weighted))
    variance = float(np.var(weighted, ddof=1))
    log_term = math.log(2.0 / family_alpha)
    eb_radius = math.sqrt(2.0 * variance * log_term / n) + 14.0 * log_term / (3.0 * rho * (n - 1))
    hoeffding_radius = math.sqrt(2.0 * math.log(1.0 / family_alpha) / (n * rho * rho))
    eb_lower = max(-1.0, estimate - eb_radius)
    hoeffding_lower = max(-1.0, estimate - hoeffding_radius)

    sum_weights = float(np.sum(w))
    sum_squared = float(np.dot(w, w))
    ess = 0.0 if sum_squared == 0.0 else sum_weights * sum_weights / sum_squared
    return FamilyRiskReductionCertificate(
        family=family,
        sample_size=n,
        estimate=estimate,
        sample_variance=variance,
        empirical_bernstein_lower=eb_lower,
        hoeffding_lower=hoeffding_lower,
        baseline_only_failures=int(np.sum((baseline == 1) & (full == 0))),
        full_only_failures=int(np.sum((baseline == 0) & (full == 1))),
        shared_failures=int(np.sum((baseline == 1) & (full == 1))),
        shared_safe=int(np.sum((baseline == 0) & (full == 0))),
        generic_weight_ess=ess,
        maximum_weight=float(np.max(w)),
        rho=float(rho),
        family_alpha=float(family_alpha),
    )


def simultaneous_risk_reduction_certificate(
    registered_families: Sequence[str],
    family_samples: Mapping[str, tuple[Sequence[float], Sequence[float], Sequence[float]]],
    *,
    rho: float,
    alpha: float,
    minimum_relevant_reduction: float,
) -> SimultaneousRiskReductionCertificate:
    """Compute an equal-allocation simultaneous certificate for all families."""

    families = tuple(registered_families)
    if not families or any(not isinstance(value, str) or not value for value in families):
        raise PairedRiskReductionError("registered_families must be non-empty strings")
    if len(set(families)) != len(families):
        raise PairedRiskReductionError("registered_families contains duplicates")
    if set(family_samples) != set(families):
        raise PairedRiskReductionError("family samples do not exactly match the registered family set")
    if not 0.0 < alpha < 1.0 or not math.isfinite(alpha):
        raise PairedRiskReductionError("alpha must lie in (0, 1)")
    if not -1.0 <= minimum_relevant_reduction <= 1.0 or not math.isfinite(minimum_relevant_reduction):
        raise PairedRiskReductionError("minimum_relevant_reduction must lie in [-1, 1]")

    family_alpha = alpha / len(families)
    certificates = []
    for family in families:
        weights, baseline, full = family_samples[family]
        certificates.append(
            family_risk_reduction_certificate(
                family,
                weights,
                baseline,
                full,
                rho=rho,
                family_alpha=family_alpha,
            )
        )
    eb_min = min(item.empirical_bernstein_lower for item in certificates)
    hoeffding_min = min(item.hoeffding_lower for item in certificates)
    return SimultaneousRiskReductionCertificate(
        registered_families=families,
        family_certificates=tuple(certificates),
        empirical_bernstein_lower_min=eb_min,
        hoeffding_lower_min=hoeffding_min,
        confidence=1.0 - alpha,
        minimum_relevant_reduction=float(minimum_relevant_reduction),
        certified=bool(eb_min > minimum_relevant_reduction),
    )
