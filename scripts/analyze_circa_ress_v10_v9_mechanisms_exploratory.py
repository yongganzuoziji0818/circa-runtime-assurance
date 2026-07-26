"""Result-visible descriptive mechanism decomposition of immutable V9 evidence.

This script is post-result and exploratory. It performs no simulator call,
generates no seed, changes no V9 endpoint, and makes no confirmatory test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from agc_runtime_assurance.circa_ress_v9_schema import array_schema


ROUTE_ID = "CIRCA-RESS-V10-THEORY-MECHANISM-AND-ASSURANCE-CASE-R1"
TRACE_SHA256 = "4d24ead5a673b902963d274715807d4a8c8e586afc8d999d7fa09087d469a6a0"
FAMILIES = ("SDF1", "SDF2", "SDF3", "SDF4", "SDF5")
SPLITS = ("validation", "evaluation")
DRIVERS = (
    "command_persistent_unbounded_v3",
    "planar_speed_projected_v4",
)
METHODS = (
    "shadow_no_override",
    "registered_one_step_cbf",
    "robust_backup_filter_v7_stale_point",
    "timestamp_aligned_set_backup_v8",
)
SHADOW = 0
PRIMARY = 3
FROZEN_MINIMUM_REDUCTION = 0.10
FROZEN_MAXIMUM_PRIMARY_RISK = 0.40


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_headers(arrays: dict[str, np.ndarray]) -> None:
    expected = array_schema()
    if set(arrays) != set(expected):
        raise RuntimeError("V9 trace member set differs from frozen schema")
    for name, (dtype, shape) in expected.items():
        value = arrays[name]
        if value.dtype != dtype or value.shape != shape or value.dtype.hasobject:
            raise RuntimeError(f"V9 trace header mismatch: {name}")


def selected_rows(
    arrays: dict[str, np.ndarray],
    *,
    split: int,
    family: int,
    method: int,
    driver: int | None = None,
) -> np.ndarray:
    mask = (
        (arrays["split_index"] == split)
        & (arrays["family_index"] == family)
        & (arrays["method_index"] == method)
    )
    if driver is not None:
        mask &= arrays["driver_index"] == driver
    return mask


def grouped_boolean(
    arrays: dict[str, np.ndarray],
    mask: np.ndarray,
    field: str,
    *,
    reducer: str = "any",
) -> tuple[np.ndarray, np.ndarray]:
    pair_ids = np.unique(arrays["pair_id"][mask])
    values = np.asarray(arrays[field], dtype=bool)
    if reducer == "any":
        grouped = [np.any(values[mask & (arrays["pair_id"] == pair)]) for pair in pair_ids]
    elif reducer == "all":
        grouped = [np.all(values[mask & (arrays["pair_id"] == pair)]) for pair in pair_ids]
    else:
        raise ValueError("unknown reducer")
    return pair_ids, np.asarray(grouped, dtype=bool)


def paired_outcomes(
    arrays: dict[str, np.ndarray],
    family: int,
    field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = SPLITS.index("evaluation")
    shadow_ids, shadow = grouped_boolean(
        arrays,
        selected_rows(arrays, split=split, family=family, method=SHADOW),
        field,
    )
    primary_ids, primary = grouped_boolean(
        arrays,
        selected_rows(arrays, split=split, family=family, method=PRIMARY),
        field,
    )
    if not np.array_equal(shadow_ids, primary_ids):
        raise RuntimeError("paired identities differ")
    return shadow_ids, shadow, primary


def method_summary(
    arrays: dict[str, np.ndarray],
    family: int,
    method: int,
) -> dict:
    mask = selected_rows(
        arrays,
        split=SPLITS.index("evaluation"),
        family=family,
        method=method,
    )
    ids, operational = grouped_boolean(
        arrays, mask, "operational_first_violation"
    )
    _, hard = grouped_boolean(arrays, mask, "hard_first_violation")
    _, intervention = grouped_boolean(arrays, mask, "applied_intervention")
    _, typed_any = grouped_boolean(arrays, mask, "typed_refusal")
    _, typed_all = grouped_boolean(
        arrays, mask, "typed_refusal", reducer="all"
    )
    by_driver = {}
    for driver, driver_name in enumerate(DRIVERS):
        driver_mask = selected_rows(
            arrays,
            split=SPLITS.index("evaluation"),
            family=family,
            method=method,
            driver=driver,
        )
        by_driver[driver_name] = {
            "rollout_rows": int(np.sum(driver_mask)),
            "operational_first_violation_rate": float(
                np.mean(arrays["operational_first_violation"][driver_mask])
            ),
            "hard_first_violation_rate": float(
                np.mean(arrays["hard_first_violation"][driver_mask])
            ),
            "applied_intervention_rate": float(
                np.mean(arrays["applied_intervention"][driver_mask])
            ),
            "typed_refusal_rate": float(
                np.mean(arrays["typed_refusal"][driver_mask])
            ),
        }
    return {
        "independent_units": int(ids.size),
        "operational_first_violation_rate": float(np.mean(operational)),
        "hard_first_violation_rate": float(np.mean(hard)),
        "applied_intervention_rate": float(np.mean(intervention)),
        "typed_refusal_any_driver_unit_rate": float(np.mean(typed_any)),
        "typed_refusal_all_driver_unit_rate": float(np.mean(typed_all)),
        "by_driver": by_driver,
    }


def validation_witness_summary(
    arrays: dict[str, np.ndarray],
    family: int,
) -> dict:
    mask = selected_rows(
        arrays,
        split=SPLITS.index("validation"),
        family=family,
        method=PRIMARY,
    )
    if not np.any(mask):
        raise RuntimeError(
            f"missing frozen validation rows for family={FAMILIES[family]}"
        )
    certificate = (
        (arrays["refusal_code"] == 0)
        & (arrays["backup_tube_feasible"] == 1)
        & (arrays["terminal_reachability"] == 1)
    )
    transition_error = np.any(
        certificate[mask] & (arrays["operational_margin_m"][mask] < 0.0),
        axis=1,
    )
    ids = arrays["pair_id"][mask]
    unique_ids = np.unique(ids)
    unit_errors = np.asarray(
        [np.any(transition_error[ids == pair]) for pair in unique_ids],
        dtype=bool,
    )
    return {
        "rollout_rows": int(np.sum(mask)),
        "independent_units": int(unique_ids.size),
        "certificate_step_exposure_rate": float(np.mean(certificate[mask])),
        "negative_margin_certificate_steps": int(
            np.sum(
                certificate[mask]
                & (arrays["operational_margin_m"][mask] < 0.0)
            )
        ),
        "witness_error_units": int(np.sum(unit_errors)),
        "witness_error_unit_rate": float(np.mean(unit_errors)),
    }


def family_summary(arrays: dict[str, np.ndarray], family: int) -> dict:
    pair_ids, shadow, primary = paired_outcomes(
        arrays, family, "operational_first_violation"
    )
    _, shadow_hard, primary_hard = paired_outcomes(
        arrays, family, "hard_first_violation"
    )
    reduction = shadow.astype(float) - primary.astype(float)
    validation = validation_witness_summary(arrays, family)
    primary_risk = float(np.mean(primary))
    opportunity = float(np.mean(shadow))
    effect = float(np.mean(reduction))
    tags = []
    if opportunity == 0.0:
        tags.append("OPPORTUNITY_ZERO_IN_FROZEN_SHADOW")
    if effect < FROZEN_MINIMUM_REDUCTION:
        tags.append("REDUCTION_BELOW_ORIGINAL_FROZEN_MINIMUM")
    if primary_risk > FROZEN_MAXIMUM_PRIMARY_RISK:
        tags.append("RESIDUAL_RISK_ABOVE_ORIGINAL_FROZEN_MAXIMUM")
    if validation["witness_error_units"] > 0:
        tags.append("VALIDATION_WITNESS_ERRORS_OBSERVED")
    methods = {
        method_name: method_summary(arrays, family, method)
        for method, method_name in enumerate(METHODS)
    }
    return {
        "family": FAMILIES[family],
        "independent_evaluation_units": int(pair_ids.size),
        "shadow_opportunity_rate": opportunity,
        "primary_residual_operational_risk": primary_risk,
        "paired_operational_risk_reduction": effect,
        "hard_risk_change_primary_minus_shadow": float(
            np.mean(primary_hard.astype(float) - shadow_hard.astype(float))
        ),
        "validation_witness": validation,
        "methods": methods,
        "descriptive_boundary_tags": tags,
    }


def analyze(trace_path: Path) -> dict:
    observed_hash = sha256(trace_path)
    if observed_hash != TRACE_SHA256:
        raise RuntimeError("immutable V9 trace hash mismatch")
    with np.load(trace_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    validate_headers(arrays)
    families = [family_summary(arrays, family) for family in range(len(FAMILIES))]
    return {
        "schema_version": "1.0",
        "analysis_id": "circa-ress-v10-v9-mechanisms-exploratory-r1",
        "route_id": ROUTE_ID,
        "status": "PASS_RESULT_VISIBLE_EXPLORATORY_DESCRIPTIVE_ANALYSIS",
        "classification": (
            "POST_RESULT_EXPLORATORY_NOT_CONFIRMATORY_NOT_A_V9_RETRY"
        ),
        "input_trace_sha256": observed_hash,
        "all_registered_families_reported": True,
        "all_registered_methods_reported": True,
        "original_frozen_thresholds_used_as_descriptive_references_only": {
            "minimum_paired_reduction": FROZEN_MINIMUM_REDUCTION,
            "maximum_primary_operational_risk": FROZEN_MAXIMUM_PRIMARY_RISK,
        },
        "new_hypothesis_tests": False,
        "new_p_values": False,
        "candidate_or_family_selection": False,
        "causal_mechanism_claim": False,
        "v9_route_disposition_changed": False,
        "families": families,
        "mechanical_trace_checks": {
            "all_rollouts_complete": bool(
                np.all(arrays["completed_steps"] == 80)
            ),
            "all_steps_complete": bool(np.all(arrays["completed_step_mask"])),
            "rollout_rows": int(arrays["completed_steps"].size),
            "unique_pair_ids": int(np.unique(arrays["pair_id"]).size),
        },
        "zero_science_record": {
            "scientific_seed_generated": False,
            "simulator_or_scientific_runner_invoked": False,
            "v9_scientific_attempt_consumed": False,
            "v9_result_or_frozen_analysis_modified": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError("exploratory output target already exists")
    payload = analyze(arguments.trace.resolve())
    arguments.output.parent.mkdir(parents=True, exist_ok=False)
    temporary = arguments.output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
