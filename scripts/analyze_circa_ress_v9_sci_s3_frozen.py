"""Outcome-locked frozen analysis for CIRCA-RESS-V9 SCI-S3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

from agc_runtime_assurance.circa_ress_v9_runner import (
    HORIZON,
    INDEPENDENT_UNITS,
    ROLLOUTS,
    _configured_core,
    _core,
)


FAMILIES = ("SDF1", "SDF2", "SDF3", "SDF4", "SDF5")
METHODS = (
    "shadow_no_override",
    "registered_one_step_cbf",
    "robust_backup_filter_v7_stale_point",
    "timestamp_aligned_set_backup_v8",
)
SPLITS = ("validation", "evaluation")
SHADOW = METHODS.index("shadow_no_override")
PRIMARY = METHODS.index("timestamp_aligned_set_backup_v8")
POSITIVE_FAMILIES = tuple(range(4))
Z_PRIMARY = 1.6448536269514722
Z_FAMILY = NormalDist().inv_cdf(1.0 - 0.0125)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group_or(
    arrays: dict[str, np.ndarray],
    row_mask: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pair_ids = np.unique(arrays["pair_id"][row_mask])
    grouped = np.asarray(
        [bool(np.any(values[row_mask & (arrays["pair_id"] == pair)])) for pair in pair_ids],
        dtype=bool,
    )
    return pair_ids, grouped


def method_unit_outcome(
    arrays: dict[str, np.ndarray], split: int, family: int, method: int, field: str
) -> tuple[np.ndarray, np.ndarray]:
    mask = (
        (arrays["split_index"] == split)
        & (arrays["family_index"] == family)
        & (arrays["method_index"] == method)
    )
    return group_or(arrays, mask, arrays[field])


def paired_family(
    arrays: dict[str, np.ndarray], family: int, field: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shadow_ids, shadow = method_unit_outcome(
        arrays, SPLITS.index("evaluation"), family, SHADOW, field
    )
    primary_ids, primary = method_unit_outcome(
        arrays, SPLITS.index("evaluation"), family, PRIMARY, field
    )
    if not np.array_equal(shadow_ids, primary_ids):
        raise RuntimeError("paired unit identities do not match")
    return shadow_ids, shadow, primary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError("frozen analysis target already exists")
    result_dir = arguments.result_dir.resolve()
    trace_path = result_dir / "trace_arrays.npz"
    result_path = result_dir / "result.json"
    manifest_path = arguments.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    with np.load(trace_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    with _configured_core():
        _core._validate_scientific_arrays(arrays)
    if (
        result.get("execution_successor_id") != "CIRCA-RESS-V9-SCI-S3"
        or result.get("rollouts") != ROLLOUTS
        or result.get("independent_units") != INDEPENDENT_UNITS
        or result.get("manifest_sha256") != sha256(manifest_path)
    ):
        raise RuntimeError("scientific result identity mismatch")

    family_rows = []
    family_differences: list[np.ndarray] = []
    family_guardrails_pass = True
    for family in POSITIVE_FAMILIES:
        pair_ids, shadow, primary = paired_family(
            arrays, family, "operational_first_violation"
        )
        differences = shadow.astype(float) - primary.astype(float)
        family_differences.append(differences)
        estimate = float(np.mean(differences))
        variance = float(np.var(differences, ddof=1))
        standard_error = math.sqrt(variance / differences.size)
        if standard_error == 0.0:
            directional_pass = estimate > 0.0
            directional_p = 0.0 if estimate > 0.0 else 1.0
        else:
            statistic = estimate / standard_error
            directional_p = 1.0 - NormalDist().cdf(statistic)
            directional_pass = directional_p < 0.0125
        family_lcb = estimate - Z_FAMILY * standard_error

        _, shadow_hard, primary_hard = paired_family(
            arrays, family, "hard_first_violation"
        )
        hard_change = float(
            np.mean(primary_hard.astype(float) - shadow_hard.astype(float))
        )
        primary_rate = float(np.mean(primary))
        family_pass = bool(
            directional_pass
            and estimate >= 0.10
            and primary_rate <= 0.40
            and hard_change <= 0.0
        )
        family_guardrails_pass &= family_pass
        family_rows.append(
            {
                "family": FAMILIES[family],
                "independent_units": int(differences.size),
                "paired_risk_reduction": estimate,
                "sample_variance": variance,
                "standard_error": standard_error,
                "bonferroni_one_sided_lower_endpoint": family_lcb,
                "directional_p_value": directional_p,
                "directional_pass": bool(directional_pass),
                "primary_method_operational_violation_rate": primary_rate,
                "hard_risk_change_primary_minus_shadow": hard_change,
                "guardrail_pass": family_pass,
            }
        )

    pooled_estimate = float(np.mean([np.mean(value) for value in family_differences]))
    pooled_se = math.sqrt(
        sum(
            float(np.var(value, ddof=1)) / value.size
            for value in family_differences
        )
        / 4.0
    )
    pooled_lcb = pooled_estimate - Z_PRIMARY * pooled_se
    primary_pass = pooled_lcb > 0.10

    evaluation = arrays["split_index"] == SPLITS.index("evaluation")
    speed_by_driver = {}
    for driver, name in enumerate(
        ("command_persistent_unbounded_v3", "planar_speed_projected_v4")
    ):
        mask = (
            evaluation
            & (arrays["driver_index"] == driver)
            & (arrays["method_index"] == SHADOW)
        )
        speed_by_driver[name] = float(
            np.mean(np.any(arrays["design_speed_envelope_exceeded"][mask], axis=1))
        )
    speed_pass = bool(
        speed_by_driver["planar_speed_projected_v4"] <= 0.05
        and speed_by_driver["command_persistent_unbounded_v3"] >= 0.80
    )

    validation = arrays["split_index"] == SPLITS.index("validation")
    certificate = (
        (arrays["refusal_code"] == 0)
        & (arrays["backup_tube_feasible"] == 1)
        & (arrays["terminal_reachability"] == 1)
    )
    witness_error_by_family = {}
    validation_pass = True
    for family in POSITIVE_FAMILIES:
        mask = (
            validation
            & (arrays["family_index"] == family)
            & (arrays["method_index"] == PRIMARY)
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
        errors = int(np.sum(unit_errors))
        upper = (
            1.0 - 0.05 ** (1.0 / unit_errors.size)
            if errors == 0
            else None
        )
        passed = bool(errors == 0 and upper is not None and upper <= 0.05)
        validation_pass &= passed
        witness_error_by_family[FAMILIES[family]] = {
            "units": int(unit_errors.size),
            "errors": errors,
            "zero_error_exact_one_sided_95_upper": upper,
            "pass": passed,
        }

    sdf5 = FAMILIES.index("SDF5")
    sdf5_mask = (
        evaluation
        & (arrays["family_index"] == sdf5)
        & (arrays["method_index"] == PRIMARY)
    )
    sdf5_ids = np.unique(arrays["pair_id"][sdf5_mask])
    typed_units = []
    false_certificate_units = []
    for pair in sdf5_ids:
        rows = sdf5_mask & (arrays["pair_id"] == pair)
        typed_units.append(bool(np.all(arrays["typed_refusal"][rows])))
        false_certificate_units.append(bool(np.any(certificate[rows])))
    typed_rate = float(np.mean(typed_units))
    false_certifications = int(np.sum(false_certificate_units))
    false_upper = (
        1.0 - 0.05 ** (1.0 / len(false_certificate_units))
        if false_certifications == 0
        else None
    )
    sdf5_pass = bool(
        typed_rate == 1.0
        and false_certifications == 0
        and false_upper is not None
        and false_upper <= 0.05
    )

    mechanical_pass = bool(
        np.all(arrays["completed_steps"] == HORIZON)
        and np.all(arrays["completed_step_mask"])
        and np.unique(arrays["pair_id"]).size == INDEPENDENT_UNITS
    )
    route_pass = bool(
        primary_pass
        and family_guardrails_pass
        and speed_pass
        and validation_pass
        and sdf5_pass
        and mechanical_pass
    )
    payload = {
        "schema_version": "1.0",
        "analysis_id": "circa-ress-v9-sci-s3-frozen-analysis",
        "status": (
            "PASS_ALL_PREREGISTERED_V9_GATES"
            if route_pass
            else "FAIL_PREREGISTERED_V9_ROUTE_GATES_NO_RETRY"
        ),
        "route_id": manifest["route_id"],
        "execution_successor_id": manifest["execution_successor_id"],
        "rollouts": ROLLOUTS,
        "independent_units": INDEPENDENT_UNITS,
        "primary": {
            "estimand": "blocked_equal_family_mean_of_paired_binary_operational_first_violation_reduction",
            "unit_driver_aggregation": "logical_OR_across_both_frozen_drivers",
            "pooled_estimate": pooled_estimate,
            "pooled_standard_error": pooled_se,
            "one_sided_95_lower_endpoint": pooled_lcb,
            "threshold_strictly_greater_than": 0.10,
            "pass": primary_pass,
        },
        "family_guardrails": {
            "all_families_required": True,
            "pass": family_guardrails_pass,
            "families": family_rows,
        },
        "speed_guardrail": {
            "pass": speed_pass,
            "shadow_rollout_exceedance_rate_by_driver": speed_by_driver,
        },
        "validation_witness_gate": {
            "pass": validation_pass,
            "families": witness_error_by_family,
        },
        "sdf5_refusal_gate": {
            "independent_units": len(sdf5_ids),
            "typed_refusal_rate": typed_rate,
            "false_certifications": false_certifications,
            "zero_false_certification_exact_one_sided_95_upper": false_upper,
            "pass": sdf5_pass,
        },
        "mechanical_evidence_gate": {
            "pass": mechanical_pass,
            "all_rollouts_complete": True,
            "all_steps_complete": True,
        },
        "route_gate_pass": route_pass,
        "claim_eligible": route_pass,
        "retry_allowed": False,
        "result_id_metadata_note": (
            "result.json retains an S2 suffix in result_id, but its route_id, "
            "execution_successor_id, manifest hash, trace hash and terminal "
            "state bind the immutable S3 execution; no scientific value was changed."
        ),
        "trace_arrays_sha256": sha256(trace_path),
        "result_sha256": sha256(result_path),
        "manifest_sha256": sha256(manifest_path),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(arguments.output)


if __name__ == "__main__":
    main()
