"""Known-probability proposal-efficiency primitives for P4 Route B.

The finite-support benchmark isolates proposal allocation from controller
efficacy.  Outcome categories are latent until a paired dual-method simulation;
only the hazard stratum is available to a screening-fitted proposal.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np


CATEGORY_ORDER = (
    "shared_safe",
    "baseline_only_failure",
    "full_only_failure",
    "shared_failure",
)
LEARNED_TARGETS = (
    "baseline_failure",
    "union_failure",
    "positive_disagreement",
    "bidirectional_disagreement",
)


class ProposalEfficiencyError(ValueError):
    """Raised when a proposal-efficiency object is not evaluable."""


@dataclass(frozen=True)
class KnownProbabilityFamily:
    family: str
    nominal_stratum_probabilities: np.ndarray
    category_probabilities: np.ndarray
    true_risk_reduction: float
    true_disagreement_probability: float


@dataclass(frozen=True)
class FittedProposal:
    target: str
    target_scores: np.ndarray
    proposal_stratum_probabilities: np.ndarray
    defensive_stratum_probabilities: np.ndarray
    importance_weights: np.ndarray
    fingerprint: str


def _probability_vector(values: Sequence[float], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ProposalEfficiencyError(f"{label} must be a finite one-dimensional vector")
    if np.any(array <= 0.0) or not math.isclose(float(array.sum()), 1.0, abs_tol=1e-12):
        raise ProposalEfficiencyError(f"{label} must contain positive probabilities summing to one")
    return array


def family_from_mapping(raw: Mapping[str, object]) -> KnownProbabilityFamily:
    family = raw.get("family")
    if not isinstance(family, str) or not family:
        raise ProposalEfficiencyError("family must be a non-empty string")
    nominal = _probability_vector(
        raw.get("nominal_stratum_probabilities", []), "nominal stratum probabilities"
    )
    plus = np.asarray(raw.get("baseline_only_probabilities", []), dtype=float)
    minus = np.asarray(raw.get("full_only_probabilities", []), dtype=float)
    shared = np.asarray(raw.get("shared_failure_probabilities", []), dtype=float)
    if any(array.shape != nominal.shape for array in (plus, minus, shared)):
        raise ProposalEfficiencyError("conditional category arrays must match the stratum vector")
    if any(not np.all(np.isfinite(array)) for array in (plus, minus, shared)):
        raise ProposalEfficiencyError("conditional category arrays contain non-finite values")
    if any(np.any((array < 0.0) | (array > 1.0)) for array in (plus, minus, shared)):
        raise ProposalEfficiencyError("conditional category probabilities must lie in [0, 1]")
    safe = 1.0 - plus - minus - shared
    if np.any(safe < -1e-12):
        raise ProposalEfficiencyError("conditional category probabilities exceed one")
    safe = np.maximum(safe, 0.0)
    categories = np.column_stack((safe, plus, minus, shared))
    if not np.allclose(categories.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise ProposalEfficiencyError("conditional category rows must sum to one")
    delta = float(nominal @ (plus - minus))
    disagreement = float(nominal @ (plus + minus))
    return KnownProbabilityFamily(
        family=family,
        nominal_stratum_probabilities=nominal,
        category_probabilities=categories,
        true_risk_reduction=delta,
        true_disagreement_probability=disagreement,
    )


def target_indicator(categories: Sequence[int] | np.ndarray, target: str) -> np.ndarray:
    values = np.asarray(categories, dtype=int)
    if values.ndim != 1 or np.any((values < 0) | (values >= len(CATEGORY_ORDER))):
        raise ProposalEfficiencyError("categories must be a one-dimensional valid index vector")
    if target == "baseline_failure":
        return ((values == 1) | (values == 3)).astype(float)
    if target == "union_failure":
        return (values != 0).astype(float)
    if target == "positive_disagreement":
        return (values == 1).astype(float)
    if target == "bidirectional_disagreement":
        return ((values == 1) | (values == 2)).astype(float)
    raise ProposalEfficiencyError(f"unknown proposal target: {target}")


def sample_paths(
    family: KnownProbabilityFamily,
    stratum_probabilities: Sequence[float] | np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    proposal = _probability_vector(stratum_probabilities, "sampling stratum probabilities")
    if proposal.shape != family.nominal_stratum_probabilities.shape:
        raise ProposalEfficiencyError("sampling distribution has the wrong number of strata")
    if not isinstance(sample_size, int) or sample_size < 1:
        raise ProposalEfficiencyError("sample_size must be a positive integer")
    strata = rng.choice(proposal.size, size=sample_size, p=proposal)
    uniforms = rng.random(sample_size)
    cumulative = np.cumsum(family.category_probabilities[strata], axis=1)
    categories = np.sum(uniforms[:, None] > cumulative, axis=1).astype(np.int8)
    return strata.astype(np.int16), categories


def fit_defensive_proposal(
    family: KnownProbabilityFamily,
    screening_strata: Sequence[int] | np.ndarray,
    screening_categories: Sequence[int] | np.ndarray,
    *,
    target: str,
    rho: float,
    beta_prior_half_count: float,
    score_floor: float,
) -> FittedProposal:
    if target not in LEARNED_TARGETS:
        raise ProposalEfficiencyError(f"unsupported learned target: {target}")
    if not 0.0 < rho < 1.0 or not math.isfinite(rho):
        raise ProposalEfficiencyError("rho must lie in (0, 1) for a learned proposal")
    if beta_prior_half_count <= 0.0 or not math.isfinite(beta_prior_half_count):
        raise ProposalEfficiencyError("beta prior half-count must be positive")
    if score_floor <= 0.0 or not math.isfinite(score_floor):
        raise ProposalEfficiencyError("score_floor must be positive")
    strata = np.asarray(screening_strata, dtype=int)
    categories = np.asarray(screening_categories, dtype=int)
    if strata.ndim != 1 or categories.ndim != 1 or strata.shape != categories.shape:
        raise ProposalEfficiencyError("screening strata and categories must be paired vectors")
    if strata.size < 2 or np.any((strata < 0) | (strata >= family.nominal_stratum_probabilities.size)):
        raise ProposalEfficiencyError("screening strata are empty or invalid")
    labels = target_indicator(categories, target)
    counts = np.bincount(strata, minlength=family.nominal_stratum_probabilities.size).astype(float)
    hits = np.bincount(
        strata, weights=labels, minlength=family.nominal_stratum_probabilities.size
    ).astype(float)
    scores = (hits + beta_prior_half_count) / (counts + 2.0 * beta_prior_half_count)
    tilted = family.nominal_stratum_probabilities * np.maximum(scores, score_floor)
    tilted /= float(tilted.sum())
    defensive = rho * family.nominal_stratum_probabilities + (1.0 - rho) * tilted
    weights = family.nominal_stratum_probabilities / defensive
    if np.any(weights > 1.0 / rho + 1e-12):
        raise ProposalEfficiencyError("defensive proposal violates its weight bound")
    payload = np.concatenate((scores, tilted, defensive, weights)).astype("<f8").tobytes()
    fingerprint = hashlib.sha256(target.encode("utf-8") + payload).hexdigest()
    return FittedProposal(
        target=target,
        target_scores=scores,
        proposal_stratum_probabilities=tilted,
        defensive_stratum_probabilities=defensive,
        importance_weights=weights,
        fingerprint=fingerprint,
    )


def signed_differences(categories: Sequence[int] | np.ndarray) -> np.ndarray:
    values = np.asarray(categories, dtype=int)
    target_indicator(values, "bidirectional_disagreement")
    return (values == 1).astype(float) - (values == 2).astype(float)


def equal_call_counts(total_path_budget: int, screen_fraction: float, learned: bool) -> tuple[int, int, int]:
    if not isinstance(total_path_budget, int) or total_path_budget < 10:
        raise ProposalEfficiencyError("total_path_budget must be an integer of at least ten")
    if not 0.0 < screen_fraction < 0.5 or not math.isfinite(screen_fraction):
        raise ProposalEfficiencyError("screen_fraction must lie in (0, 0.5)")
    screening = int(round(total_path_budget * screen_fraction)) if learned else 0
    evaluation = total_path_budget - screening
    if evaluation < 2:
        raise ProposalEfficiencyError("equal-call design leaves fewer than two evaluation paths")
    simulator_calls = 2 * (screening + evaluation)
    return screening, evaluation, simulator_calls


def oracle_second_moment_plan(
    family: KnownProbabilityFamily,
    *,
    target: str,
    rho: float,
    score_floor: float,
) -> dict[str, float]:
    """Deterministic design audit using true conditional scores, never run evidence."""

    categories = family.category_probabilities
    if target == "baseline_failure":
        scores = categories[:, 1] + categories[:, 3]
    elif target == "union_failure":
        scores = 1.0 - categories[:, 0]
    elif target == "positive_disagreement":
        scores = categories[:, 1]
    elif target == "bidirectional_disagreement":
        scores = categories[:, 1] + categories[:, 2]
    else:
        raise ProposalEfficiencyError(f"unknown oracle target: {target}")
    nominal = family.nominal_stratum_probabilities
    tilted = nominal * np.maximum(scores, score_floor)
    tilted /= float(tilted.sum())
    defensive = rho * nominal + (1.0 - rho) * tilted
    disagreement = categories[:, 1] + categories[:, 2]
    second_moment = float(np.sum(nominal * nominal / defensive * disagreement))
    variance = second_moment - family.true_risk_reduction**2
    return {
        "variance": variance,
        "maximum_weight": float(np.max(nominal / defensive)),
    }
