"""A-priori policy-seed power planning for the P4 primary paired endpoint.

The SESOI is fixed independently of observed pilot effects.  Development data
may inform only the seed-difference variance, which is inflated upward.  This
module never computes post-hoc/observed power.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import string
from typing import Any

import numpy as np
from scipy.stats import norm, t


class PowerPlanningError(RuntimeError):
    pass


@dataclass(frozen=True)
class SeedPowerPoint:
    policy_seed_count: int
    estimated_power: float
    power_mc_interval: tuple[float, float]
    estimated_type_i_error: float
    type_i_mc_interval: tuple[float, float]
    type_i_calibrated: bool


@dataclass(frozen=True)
class SeedPowerPlan:
    sesoi_absolute_reduction: float
    observed_seed_difference_sd: float
    variance_inflation_factor: float
    conservative_seed_difference_sd: float
    alpha: float
    target_power: float
    monte_carlo_replicates: int
    selected_policy_seed_count: int | None
    max_affordable_policy_seeds: int
    resource_feasible: bool
    curve: tuple[SeedPowerPoint, ...]
    planning_method: str
    fingerprint: str


def plan_seed_power_from_manifest(path: str | Path) -> SeedPowerPlan:
    """Load a frozen planning input; never infer authority or variance."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("stage") != "formal_power_planning":
        raise PowerPlanningError("stage must be formal_power_planning")
    if manifest.get("development_variance_available") is not True:
        raise PowerPlanningError("authorized development variance is not available")
    if manifest.get("formal_experiment_authorized") is not False:
        raise PowerPlanningError("power planning must not grant formal experiment authority")
    if manifest.get("sealed_data_used") is not False:
        raise PowerPlanningError("power planning must not use sealed data")
    fields = (
        "sesoi_absolute_reduction", "effect_basis", "sesoi_source_hash",
        "observed_seed_difference_sd", "variance_inflation_factor",
        "variance_basis", "variance_source_hash", "alpha", "target_power",
        "monte_carlo_replicates", "monte_carlo_seed", "minimum_policy_seeds",
        "maximum_policy_seeds", "max_affordable_policy_seeds",
    )
    missing = [field for field in fields if field not in manifest]
    if missing:
        raise PowerPlanningError(f"power manifest missing fields: {missing}")
    return simulate_seed_power_plan(**{field: manifest[field] for field in fields})


def simulate_seed_power_plan(
    *,
    sesoi_absolute_reduction: float,
    effect_basis: str,
    sesoi_source_hash: str,
    observed_seed_difference_sd: float,
    variance_inflation_factor: float,
    variance_basis: str,
    variance_source_hash: str,
    alpha: float = 0.05,
    target_power: float = 0.90,
    monte_carlo_replicates: int = 5000,
    monte_carlo_seed: int = 4172026,
    minimum_policy_seeds: int = 3,
    maximum_policy_seeds: int = 100,
    max_affordable_policy_seeds: int = 40,
) -> SeedPowerPlan:
    """Simulate two-sided paired-t planning power under the frozen SESOI.

    The paired t test is the parametric primary planning model; the final
    analysis must additionally report the pre-registered paired bootstrap CI
    and permutation sensitivity.  Both alternative and null are simulated so
    an anticonservative implementation cannot yield a sample-size decision.
    """
    if effect_basis != "pre_registered_sesoi":
        raise PowerPlanningError("effect basis must be pre_registered_sesoi, never pilot effect")
    if variance_basis != "development_sd_upward_adjusted":
        raise PowerPlanningError("variance basis must be development_sd_upward_adjusted")
    _require_digest(sesoi_source_hash, "sesoi_source_hash")
    _require_digest(variance_source_hash, "variance_source_hash")
    numeric = np.asarray(
        [sesoi_absolute_reduction, observed_seed_difference_sd,
         variance_inflation_factor, alpha, target_power], dtype=float,
    )
    if not np.all(np.isfinite(numeric)):
        raise PowerPlanningError("power inputs must be finite")
    if not 0.0 < sesoi_absolute_reduction < 1.0:
        raise PowerPlanningError("SESOI must lie strictly between 0 and 1")
    if observed_seed_difference_sd <= 0.0:
        raise PowerPlanningError("observed seed-difference SD must be positive")
    if variance_inflation_factor < 1.0:
        raise PowerPlanningError("variance inflation factor must be at least one")
    if not 0.0 < alpha < 1.0 or not 0.0 < target_power < 1.0:
        raise PowerPlanningError("alpha and target power must lie strictly between 0 and 1")
    if not isinstance(monte_carlo_replicates, int) or monte_carlo_replicates < 5000:
        raise PowerPlanningError("at least 5000 Monte Carlo replicates are required")
    integer_values = (
        monte_carlo_seed, minimum_policy_seeds, maximum_policy_seeds,
        max_affordable_policy_seeds,
    )
    if any(not isinstance(value, int) for value in integer_values):
        raise PowerPlanningError("seed and policy-seed limits must be integers")
    if minimum_policy_seeds < 3 or maximum_policy_seeds < minimum_policy_seeds:
        raise PowerPlanningError("policy-seed search range is invalid")
    if max_affordable_policy_seeds < 1:
        raise PowerPlanningError("max affordable policy seeds must be positive")

    conservative_sd = observed_seed_difference_sd * variance_inflation_factor
    generator = np.random.default_rng(monte_carlo_seed)
    points: list[SeedPowerPoint] = []
    selected: int | None = None
    z = float(norm.ppf(1.0 - alpha / 2.0))
    for count in range(minimum_policy_seeds, maximum_policy_seeds + 1):
        alternative = generator.normal(
            -sesoi_absolute_reduction,
            conservative_sd,
            size=(monte_carlo_replicates, count),
        )
        null = generator.normal(
            0.0,
            conservative_sd,
            size=(monte_carlo_replicates, count),
        )
        critical = float(t.ppf(1.0 - alpha / 2.0, df=count - 1))
        alternative_reject = _paired_t_rejections(alternative, critical)
        null_reject = _paired_t_rejections(null, critical)
        power = float(np.mean(alternative_reject))
        type_i = float(np.mean(null_reject))
        power_interval = _wilson_interval(
            int(np.sum(alternative_reject)), monte_carlo_replicates, z,
        )
        type_i_interval = _wilson_interval(
            int(np.sum(null_reject)), monte_carlo_replicates, z,
        )
        type_i_tolerance = max(
            0.01,
            2.0 * math.sqrt(alpha * (1.0 - alpha) / monte_carlo_replicates),
        )
        calibrated = abs(type_i - alpha) <= type_i_tolerance
        points.append(SeedPowerPoint(
            count, power, power_interval, type_i, type_i_interval, calibrated,
        ))
        if selected is None and power_interval[0] >= target_power and calibrated:
            selected = count

    payload = {
        "sesoi_absolute_reduction": sesoi_absolute_reduction,
        "effect_basis": effect_basis,
        "sesoi_source_hash": sesoi_source_hash.lower(),
        "observed_seed_difference_sd": observed_seed_difference_sd,
        "variance_inflation_factor": variance_inflation_factor,
        "variance_basis": variance_basis,
        "variance_source_hash": variance_source_hash.lower(),
        "alpha": alpha,
        "target_power": target_power,
        "monte_carlo_replicates": monte_carlo_replicates,
        "monte_carlo_seed": monte_carlo_seed,
        "minimum_policy_seeds": minimum_policy_seeds,
        "maximum_policy_seeds": maximum_policy_seeds,
        "max_affordable_policy_seeds": max_affordable_policy_seeds,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return SeedPowerPlan(
        float(sesoi_absolute_reduction), float(observed_seed_difference_sd),
        float(variance_inflation_factor), float(conservative_sd), float(alpha),
        float(target_power), monte_carlo_replicates, selected,
        max_affordable_policy_seeds,
        selected is not None and selected <= max_affordable_policy_seeds,
        tuple(points),
        "a_priori_two_sided_paired_t_monte_carlo_under_sesoi_"
        "with_null_type_i_check_and_bootstrap_permutation_sensitivity_required",
        fingerprint,
    )


def _paired_t_rejections(samples: np.ndarray, critical: float) -> np.ndarray:
    means = np.mean(samples, axis=1)
    standard_errors = np.std(samples, axis=1, ddof=1) / math.sqrt(samples.shape[1])
    statistics = np.divide(
        means, standard_errors,
        out=np.full_like(means, np.inf), where=standard_errors > 0.0,
    )
    return np.abs(statistics) > critical


def _wilson_interval(successes: int, trials: int, z: float) -> tuple[float, float]:
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _require_digest(value: Any, label: str) -> None:
    if (
        not isinstance(value, str) or len(value) != 64
        or any(character not in string.hexdigits for character in value)
    ):
        raise PowerPlanningError(f"{label} must be a 64-character hexadecimal digest")
