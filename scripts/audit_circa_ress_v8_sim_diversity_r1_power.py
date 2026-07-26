"""Deterministic, outcome-blind power and precision audit for V8.

This planning script uses analytic normal approximations and exact zero-event
Clopper-Pearson bounds. It creates no scientific seeds, simulated observations,
or scientific output.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist


def required_paired_n(
    *,
    discordance: float,
    null_difference: float,
    alternative_difference: float,
    alpha: float,
    power: float,
) -> int:
    if not (
        0 <= abs(null_difference) < alternative_difference <= discordance <= 1
    ):
        raise ValueError("paired-difference and discordance ordering is invalid")
    z_alpha = NormalDist().inv_cdf(1 - alpha)
    z_power = NormalDist().inv_cdf(power)
    variance_null = discordance - null_difference**2
    variance_alt = discordance - alternative_difference**2
    numerator = (
        z_alpha * math.sqrt(variance_null)
        + z_power * math.sqrt(variance_alt)
    ) ** 2
    return math.ceil(
        numerator / (alternative_difference - null_difference) ** 2
    )


def zero_event_upper_bound(n: int, alpha: float = 0.05) -> float:
    return 1 - alpha ** (1 / n)


def build_audit() -> dict:
    discordance_grid = (0.20, 0.30, 0.40, 0.50)
    pooled = {
        f"{value:.2f}": required_paired_n(
            discordance=value,
            null_difference=0.10,
            alternative_difference=0.20,
            alpha=0.05,
            power=0.80,
        )
        for value in discordance_grid
    }
    family = {
        f"{value:.2f}": required_paired_n(
            discordance=value,
            null_difference=0.00,
            alternative_difference=0.20,
            alpha=0.0125,
            power=0.80,
        )
        for value in discordance_grid
    }
    planned_pooled_n = 512
    planned_family_n = 128
    validation_n = 64
    refusal_n = 128
    validation_upper = zero_event_upper_bound(validation_n)
    refusal_upper = zero_event_upper_bound(refusal_n)
    passed = (
        planned_pooled_n >= max(pooled.values())
        and planned_family_n >= max(family.values())
        and validation_upper <= 0.05
        and refusal_upper <= 0.05
    )
    return {
        "schema_version": "1.0",
        "audit_id": "circa-ress-v8-sim-diversity-r1-power-audit",
        "status": "PASS_OUTCOME_BLIND_POWER_AND_PRECISION"
        if passed
        else "FAIL_OUTCOME_BLIND_POWER_AND_PRECISION",
        "method": "analytic_paired_binary_normal_approximation_plus_exact_zero_event_clopper_pearson",
        "random_number_generator_used": False,
        "scientific_seed_generated": False,
        "scientific_observation_generated": False,
        "primary_pooled_test": {
            "positive_families": 4,
            "planned_independent_evaluation_units": planned_pooled_n,
            "one_sided_alpha": 0.05,
            "target_power": 0.80,
            "null_paired_risk_difference": 0.10,
            "alternative_paired_risk_difference": 0.20,
            "required_n_sensitivity_by_total_discordance": pooled,
            "pass": planned_pooled_n >= max(pooled.values()),
        },
        "family_directional_gate": {
            "planned_independent_evaluation_units_per_family": planned_family_n,
            "one_sided_bonferroni_alpha": 0.0125,
            "target_power": 0.80,
            "null_paired_risk_difference": 0.00,
            "alternative_paired_risk_difference": 0.20,
            "required_n_sensitivity_by_total_discordance": family,
            "pass": planned_family_n >= max(family.values()),
            "claim_limit": "directional_positive_gate_only_not_family_specific_lcb_above_0.10",
        },
        "validation_precision": {
            "independent_validation_units_per_family": validation_n,
            "zero_error_one_sided_95_percent_upper_bound": validation_upper,
            "required_upper_bound": 0.05,
            "pass": validation_upper <= 0.05,
        },
        "refusal_precision": {
            "independent_falsification_units": refusal_n,
            "zero_false_certification_one_sided_95_percent_upper_bound": refusal_upper,
            "required_upper_bound": 0.05,
            "pass": refusal_upper <= 0.05,
        },
        "interpretation": [
            "The independent unit is family-candidate-future-seed, not a method rollout or control step.",
            "The pooled positive-family test is the sole confirmatory efficacy endpoint.",
            "Family gates prevent favorable pooling but do not support separate family LCB-above-0.10 claims.",
            "Power is prospective and does not use V6, V7, GZ0, GZ1, HIL, or future V8 outcomes.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {args.output}")
    audit = build_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary audit exists: {temporary}")
    temporary.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    return 0 if audit["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
