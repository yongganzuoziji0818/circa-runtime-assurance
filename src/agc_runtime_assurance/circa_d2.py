"""Exactly-once CIRCA-D2 coupled-intersection development benchmark."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from circa import (  # noqa: E402
    INEVITABLE_VIOLATION,
    NO_WITNESS,
    NOMINAL_SAFETY,
    OBSERVED_CONSISTENCY,
    clopper_pearson_lower,
    clopper_pearson_upper,
    verify_evidence_contract,
)
from circa_d1 import (  # noqa: E402
    METHODS,
    SCHEMA_DTYPE,
    evaluate_methods,
    schema_only_array,
    verify_lossless_arrays,
    write_lossless_arrays,
)


FAMILY_IDS = (
    "D2C1_randomized_overlap_calibration",
    "D2C2_null_no_effect_calibration",
    "D2E1_deterministic_intersection",
    "D2E2_model_uncertainty",
    "D2E3_covariate_shift",
    "D2E4_registered_1u1g_delay_interference",
)
CALIBRATION_FAMILIES = FAMILY_IDS[:2]
DEPLOYMENT_FAMILIES = FAMILY_IDS[2:]
NULL_FAMILY = FAMILY_IDS[1]
SENSITIVITY_MULTIPLIERS = (1.25, 1.50)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_benchmark(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("benchmark_id") != "circa-d2-coupled-intersection-1u1g-v1" or spec.get("status") != "frozen":
        raise ValueError("benchmark is not frozen CIRCA-D2 v1")
    ids = tuple(f["id"] for f in spec.get("families", []))
    if ids != FAMILY_IDS:
        raise ValueError("D2 family registry mismatch")
    if tuple(spec.get("calibration_families", [])) != CALIBRATION_FAMILIES:
        raise ValueError("D2 calibration registry mismatch")
    if tuple(spec.get("deployment_families", [])) != DEPLOYMENT_FAMILIES:
        raise ValueError("D2 deployment registry mismatch")
    if spec.get("null_family") != NULL_FAMILY or spec.get("horizon") != 30:
        raise ValueError("D2 null/horizon mismatch")
    probability_fields = (
        ("uav_distance_probs", 3),
        ("ugv_distance_probs", 3),
        ("uav_speed_probs", 3),
        ("ugv_speed_probs", 3),
        ("wind_probs", 3),
        ("sensor_probs", 3),
        ("delay_probs", 2),
        ("audit_coin_probs", 2),
    )
    for family in spec["families"]:
        for key, length in probability_fields:
            values = np.asarray(family[key], dtype=float)
            if values.size != length or np.any(values < 0) or not np.isclose(values.sum(), 1.0):
                raise ValueError(f"invalid D2 probability vector {family['id']}:{key}")
        if family["verifier_radius"] <= 0 or family["trigger_threshold"] <= spec["separation_threshold"]:
            raise ValueError("invalid D2 radius/trigger")
    return spec


def _support(spec: dict, family: dict):
    support = spec["support"]
    value_vectors = (
        support["uav_distance"],
        support["ugv_distance"],
        support["uav_speed"],
        support["ugv_speed"],
        support["lateral_wind"],
        support["sensor_error"],
        support["recovery_delay"],
        support["audit_coin"],
    )
    probability_vectors = (
        family["uav_distance_probs"],
        family["ugv_distance_probs"],
        family["uav_speed_probs"],
        family["ugv_speed_probs"],
        family["wind_probs"],
        family["sensor_probs"],
        family["delay_probs"],
        family["audit_coin_probs"],
    )
    for indices in itertools.product(*(range(len(v)) for v in value_vectors)):
        probability = float(np.prod([probability_vectors[j][indices[j]] for j in range(8)]))
        if probability == 0.0:
            continue
        yield tuple(value_vectors[j][indices[j]] for j in range(8)), probability


def compile_family(spec: dict, family: dict, radius_multiplier: float = 1.0) -> dict:
    """Enumerate a dynamically coupled closest-approach family."""
    horizon = int(spec["horizon"])
    base_decision = int(spec["decision_step"])
    threshold = float(spec["separation_threshold"])
    steps = np.arange(horizon, dtype=float)
    phase = np.sin(np.pi * steps / (horizon - 1))
    rows: list[tuple] = []
    max_prediction_error = 0.0

    for (du, dg, vu, vg, wind, sensor, delay, audit_coin), probability in _support(spec, family):
        xu_shadow = -du + vu * steps
        yg_shadow = -dg + vg * steps
        z_shadow = 0.04 + 0.05 * wind * phase
        separation_shadow = np.sqrt(xu_shadow**2 + yg_shadow**2 + z_shadow**2)
        shadow_min = float(np.min(separation_shadow))

        du_observed = du + family["position_error_scale"] * sensor
        dg_observed = dg - family["position_error_scale"] * sensor
        vu_observed = vu + family["speed_error_scale"] * sensor
        vg_observed = vg - family["speed_error_scale"] * sensor
        xu_prediction = -du_observed + vu_observed * steps
        yg_prediction = -dg_observed + vg_observed * steps
        predicted_separation = np.sqrt(xu_prediction**2 + yg_prediction**2 + 0.04**2)
        prediction = float(np.min(predicted_separation))
        max_prediction_error = max(max_prediction_error, abs(shadow_min - prediction))

        eligible = prediction < float(family["trigger_threshold"])
        if family["randomized_overlap"]:
            intervention = int(eligible and audit_coin == 1)
            pi0 = 1.0 if not eligible else float(family["audit_coin_probs"][0])
        else:
            intervention = int(eligible)
            pi0 = np.nan

        start = min(
            horizon - 1,
            base_decision + int(delay) + int(family["coordination_delay"]),
        )
        ramp = np.clip((steps - start) / max(1, horizon - 1 - start), 0.0, 1.0)
        z_active = z_shadow + intervention * float(family["climb_strength"]) * ramp
        ugv_speed_active = vg * (1.0 - intervention * float(family["brake_fraction"]) * ramp)
        yg_active = -dg + np.concatenate(([0.0], np.cumsum(ugv_speed_active[:-1])))
        separation_active = np.sqrt(xu_shadow**2 + yg_active**2 + z_active**2)
        y0 = int(shadow_min < threshold)
        y1 = int(float(np.min(separation_active)) < threshold)

        radius = float(family["verifier_radius"]) * radius_multiplier
        if intervention == 0:
            witness = OBSERVED_CONSISTENCY
        elif prediction + radius < threshold:
            witness = INEVITABLE_VIOLATION
        elif prediction - radius >= threshold:
            witness = NOMINAL_SAFETY
        else:
            witness = NO_WITNESS

        m0 = float(np.clip(0.30 + 2.0 * (vu + vg) - 0.20 * (du + dg) - 0.08 * abs(wind), 0.0, 1.0))
        rows.append((probability, intervention, y0, y1, witness, pi0, m0, prediction, shadow_min))

    data = np.asarray(rows, dtype=float)
    probabilities = data[:, 0]
    probabilities /= probabilities.sum()
    result = {
        "id": family["id"],
        "role": family["role"],
        "probability": probabilities,
        "r": data[:, 1].astype(np.int8),
        "y0": data[:, 2].astype(np.int8),
        "y1": data[:, 3].astype(np.int8),
        "witness": data[:, 4].astype(np.int8),
        "pi0": data[:, 5],
        "m0": data[:, 6],
        "prediction": data[:, 7],
        "shadow_min": data[:, 8],
        "max_prediction_error": max_prediction_error,
        "registered_radius": float(family["verifier_radius"]) * radius_multiplier,
        "randomized_overlap": bool(family["randomized_overlap"]),
    }
    result["truth_delta"] = float(probabilities @ (result["y0"] - result["y1"]))
    result["active_risk"] = float(probabilities @ result["y1"])
    result["intervention_rate"] = float(probabilities @ result["r"])
    result["witness_pinning_rate"] = float(probabilities @ ((result["r"] == 1) & (result["witness"] != 0)))
    return result


def audit_compiled_family(compiled: dict) -> dict:
    r, y0, y1, witness = compiled["r"], compiled["y0"], compiled["y1"], compiled["witness"]
    inevitable_bad = int(np.sum((r == 1) & (witness == INEVITABLE_VIOLATION) & (y0 != 1)))
    nominal_bad = int(np.sum((r == 1) & (witness == NOMINAL_SAFETY) & (y0 != 0)))
    consistency_bad = int(np.sum((r == 0) & ((witness != OBSERVED_CONSISTENCY) | (y0 != y1))))
    radius_sound = bool(compiled["max_prediction_error"] <= compiled["registered_radius"] + 1e-12)
    return {
        "family": compiled["id"],
        "support_points": int(len(r)),
        "max_prediction_error": float(compiled["max_prediction_error"]),
        "registered_radius": float(compiled["registered_radius"]),
        "radius_sound": radius_sound,
        "inevitable_witness_errors": inevitable_bad,
        "nominal_witness_errors": nominal_bad,
        "observed_consistency_errors": consistency_bad,
        "passed": radius_sound and inevitable_bad == nominal_bad == consistency_bad == 0,
    }


def _family_seed(master_seed: int, block: int, family_id: str) -> int:
    digest = hashlib.sha256(f"{master_seed}|{block}|{family_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _mean(counts: np.ndarray, values: np.ndarray) -> float:
    return float(counts @ np.asarray(values, dtype=float) / counts.sum())


def _corruption_checks() -> list[dict]:
    specs = [
        ("missing_1u1g_edge", {"interference_registered": False}, "INTERFERENCE_UNMODELED"),
        ("bad_witness_hash", {"witness_hash_matches": False}, "INVALID_WITNESS"),
        ("policy_hash_mismatch", {"policy_hash_matches": False}, "INVALID_PROVENANCE"),
        ("constraint_hash_mismatch", {"constraint_hash_matches": False}, "INVALID_PROVENANCE"),
        ("horizon_mismatch", {"horizon_matches": False}, "INVALID_PROVENANCE"),
        ("outcome_incomplete", {"outcome_complete": False}, "OUTCOME_CENSORED_INVALIDLY"),
        ("contradictory_witness", {"witnesses_contradictory": True}, "INVALID_WITNESS"),
    ]
    checks = []
    for case_id, kwargs, expected in specs:
        observed = verify_evidence_contract(**kwargs)
        checks.append(
            {
                "case_id": case_id,
                "expected_status": expected,
                "observed_status": observed.status,
                "numeric_certificate_allowed": observed.numeric_certificate_allowed,
                "passed": observed.status == expected and not observed.numeric_certificate_allowed,
            }
        )
    return checks


def simulate_d2(benchmark: dict, seed_blocks: int, n: int, master_seed: int, alpha: float, endpoint_count: int):
    families_by_id = {family["id"]: family for family in benchmark["families"]}
    compiled = {family_id: compile_family(benchmark, families_by_id[family_id]) for family_id in FAMILY_IDS}
    audits = {family_id: audit_compiled_family(compiled[family_id]) for family_id in FAMILY_IDS}
    if not all(audit["passed"] for audit in audits.values()):
        raise RuntimeError("D2 frozen support witness audit failed")

    records = schema_only_array(seed_blocks * len(FAMILY_IDS) * len(METHODS))
    total_counts = {
        family_id: np.zeros_like(compiled[family_id]["probability"], dtype=np.int64)
        for family_id in FAMILY_IDS
    }
    coverage_blocks: list[bool] = []
    false_cert_blocks: list[bool] = []
    cursor = 0

    for block in range(seed_blocks):
        block_coverage = []
        for family_id in FAMILY_IDS:
            family = compiled[family_id]
            rng = np.random.Generator(np.random.PCG64(_family_seed(master_seed, block, family_id)))
            counts = rng.multinomial(n, family["probability"])
            total_counts[family_id] += counts
            methods = evaluate_methods(family, counts, alpha, endpoint_count)
            intervention_rate = _mean(counts, family["r"])
            pinning_rate = _mean(counts, (family["r"] == 1) & (family["witness"] != 0))
            for method_id in METHODS:
                values, status = methods[method_id]
                records[cursor] = (
                    block,
                    family_id.encode("ascii"),
                    method_id.encode("ascii"),
                    n,
                    values["estimate"],
                    values["id_low"],
                    values["id_high"],
                    values["conf_low"],
                    values["conf_high"],
                    values["width"],
                    family["truth_delta"],
                    family["active_risk"],
                    intervention_rate,
                    pinning_rate,
                    int(values["covered"]),
                    int(np.isfinite(values["conf_low"]) and values["conf_low"] > 0.10),
                    2 * n,
                    status.encode("ascii"),
                )
                cursor += 1
            circa_values = methods["circa_structured_bounds"][0]
            block_coverage.append(bool(circa_values["covered"]))
            if family_id == NULL_FAMILY:
                false_cert_blocks.append(bool(circa_values["conf_low"] > 0.10))
        coverage_blocks.append(all(block_coverage))

    pooled = {}
    for family_id in FAMILY_IDS:
        family = compiled[family_id]
        methods = evaluate_methods(family, total_counts[family_id], alpha, endpoint_count)
        pooled[family_id] = {
            "role": family["role"],
            "population_truth_delta": float(family["truth_delta"]),
            "population_active_risk": float(family["active_risk"]),
            "population_intervention_rate": float(family["intervention_rate"]),
            "population_witness_pinning_rate": float(family["witness_pinning_rate"]),
            "methods": {
                method: {
                    **{
                        key: (bool(value) if isinstance(value, (bool, np.bool_)) else float(value))
                        for key, value in values.items()
                    },
                    "status": status,
                }
                for method, (values, status) in methods.items()
            },
        }

    sensitivity = {}
    for multiplier in SENSITIVITY_MULTIPLIERS:
        key = f"radius_x_{multiplier:.2f}"
        family_lcbs, family_audits = {}, {}
        for family_id in DEPLOYMENT_FAMILIES:
            conservative = compile_family(benchmark, families_by_id[family_id], multiplier)
            if not np.array_equal(conservative["probability"], compiled[family_id]["probability"]):
                raise RuntimeError("D2 sensitivity support ordering changed")
            family_audits[family_id] = audit_compiled_family(conservative)
            methods = evaluate_methods(conservative, total_counts[family_id], alpha, endpoint_count)
            family_lcbs[family_id] = float(methods["circa_structured_bounds"][0]["conf_low"])
        sensitivity[key] = {
            "family_lcbs": family_lcbs,
            "worst_deployment_lcb": min(family_lcbs.values()),
            "witness_audits": family_audits,
        }

    unverified = {}
    threshold = float(benchmark["separation_threshold"])
    for family_id in FAMILY_IDS:
        family = compiled[family_id]
        aggressive = (family["r"] == 1) & (family["prediction"] < threshold) & (family["y0"] == 0)
        unverified[family_id] = {
            "unsound_support_points": int(np.sum(aggressive)),
            "unsound_probability_mass": float(family["probability"] @ aggressive),
            "admissible_for_certificate": False,
        }

    corruption_checks = _corruption_checks()
    coverage_successes = int(sum(coverage_blocks))
    false_certifications = int(sum(false_cert_blocks))
    primary_lcbs = {
        family_id: pooled[family_id]["methods"]["circa_structured_bounds"]["conf_low"]
        for family_id in DEPLOYMENT_FAMILIES
    }
    width_ratios = {
        family_id: pooled[family_id]["methods"]["circa_structured_bounds"]["width"]
        / pooled[family_id]["methods"]["manski_worst_case"]["width"]
        for family_id in DEPLOYMENT_FAMILIES
    }
    coverage_no_loss = all(
        pooled[family_id]["methods"]["circa_structured_bounds"]["covered"]
        and pooled[family_id]["methods"]["manski_worst_case"]["covered"]
        for family_id in DEPLOYMENT_FAMILIES
    )
    max_active = max(pooled[family_id]["population_active_risk"] for family_id in DEPLOYMENT_FAMILIES)
    gates = {
        "worst_deployment_pooled_circa_lcb": min(primary_lcbs.values()),
        "worst_deployment_pooled_circa_lcb_strictly_above_0_10": min(primary_lcbs.values()) > 0.10,
        "max_deployment_population_active_risk": max_active,
        "max_deployment_population_active_risk_at_most_0_15": max_active <= 0.15,
        "simultaneous_coverage_blocks": coverage_successes,
        "simultaneous_coverage_cp_lower": clopper_pearson_lower(coverage_successes, seed_blocks),
        "simultaneous_coverage_cp_lower_at_least_0_94": clopper_pearson_lower(coverage_successes, seed_blocks) >= 0.94,
        "null_false_certification_blocks": false_certifications,
        "null_false_certification_cp_upper": clopper_pearson_upper(false_certifications, seed_blocks),
        "null_false_certification_cp_upper_at_most_0_06": clopper_pearson_upper(false_certifications, seed_blocks) <= 0.06,
        "all_deployment_width_ratios_strictly_below_one": all(value < 1.0 for value in width_ratios.values()),
        "circa_population_coverage_no_loss_vs_manski": coverage_no_loss,
        "all_sensitivity_settings_retain_primary": all(
            item["worst_deployment_lcb"] > 0.10 for item in sensitivity.values()
        ),
        "all_registered_witnesses_sound": all(audit["passed"] for audit in audits.values()),
        "all_corruptions_fail_closed": all(check["passed"] for check in corruption_checks),
        "ipw_aipw_refuse_all_non_overlap_families": all(
            pooled[family_id]["methods"]["ipw"]["status"] == "UNIDENTIFIABLE_PROPENSITY"
            and pooled[family_id]["methods"]["aipw_dr"]["status"] == "UNIDENTIFIABLE_PROPENSITY"
            for family_id in FAMILY_IDS
            if family_id != CALIBRATION_FAMILIES[0]
        ),
    }
    boolean_gates = (
        "worst_deployment_pooled_circa_lcb_strictly_above_0_10",
        "max_deployment_population_active_risk_at_most_0_15",
        "simultaneous_coverage_cp_lower_at_least_0_94",
        "null_false_certification_cp_upper_at_most_0_06",
        "all_deployment_width_ratios_strictly_below_one",
        "circa_population_coverage_no_loss_vs_manski",
        "all_sensitivity_settings_retain_primary",
        "all_registered_witnesses_sound",
        "all_corruptions_fail_closed",
        "ipw_aipw_refuse_all_non_overlap_families",
    )
    return records, {
        "gate_pass": bool(all(gates[key] for key in boolean_gates)),
        "gates": gates,
        "primary_deployment_lcbs": primary_lcbs,
        "deployment_width_ratios_vs_manski": width_ratios,
        "pooled": pooled,
        "witness_audits": audits,
        "sensitivity": sensitivity,
        "unverified_witness_ablation": unverified,
        "corruption_checks": corruption_checks,
    }


def validate_manifest(manifest: dict, repo_root: Path) -> None:
    expected = {
        "manifest_id": "circa-d2-v1",
        "status": "frozen_authorized_one_shot",
        "authorized": True,
        "retry_allowed": False,
        "execution_class": "CPU-SHARED",
        "seed_blocks": 50,
        "trajectories_per_family_per_block": 1000,
        "families": list(FAMILY_IDS),
        "calibration_families": list(CALIBRATION_FAMILIES),
        "deployment_families": list(DEPLOYMENT_FAMILIES),
        "methods": list(METHODS),
        "alpha": 0.05,
        "endpoint_count": 12,
        "delta_star": 0.10,
        "cpu_only": True,
        "workers": 1,
        "gpu_count": 0,
        "wall_time_seconds_max": 300,
        "output_size_bytes_max": 64 * 1024 * 1024,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"D2 manifest mismatch for {key}")
    required_false = ("gazebo", "pilot", "formal", "g2", "sealed", "hardware", "external_data")
    if any(manifest.get("authorizations", {}).get(key) is not False for key in required_false):
        raise ValueError("D2 out-of-scope authorization is open")
    for rel, expected_hash in manifest["integrity"]["files"].items():
        path = (repo_root / rel).resolve()
        if not path.is_relative_to(repo_root) or not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"D2 integrity mismatch: {rel}")


def run(manifest_path: Path, authorization_path: Path, repo_root: Path, output_path: Path, actual_argv: list[str]) -> dict:
    start = time.perf_counter()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"):
        raise ValueError("CUDA_VISIBLE_DEVICES must be absent, empty, or -1")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, repo_root)
    if authorization.get("status") != "authorized_exactly_once" or authorization.get("retry_allowed") is not False:
        raise ValueError("D2 launch authorization is not exactly once")
    if authorization.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("D2 launch manifest hash mismatch")
    if Path(sys.executable).resolve() != Path(authorization.get("python_executable", "")).resolve():
        raise ValueError("D2 Python executable differs from authorization")
    if authorization.get("exact_argv") != actual_argv:
        raise ValueError("D2 argv differs from authorization")
    expected_output = (repo_root / manifest["output_paths"]["summary"]).resolve()
    if output_path.resolve() != expected_output:
        raise ValueError("D2 output differs from manifest")

    output_dir = output_path.parent
    temp_dir = output_dir.with_name(output_dir.name + ".tmp")
    if output_dir.exists() or temp_dir.exists():
        raise FileExistsError("D2 output or partial output already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir()
    try:
        benchmark_path = (repo_root / manifest["benchmark_path"]).resolve()
        benchmark = load_benchmark(benchmark_path)
        records, analysis = simulate_d2(
            benchmark,
            manifest["seed_blocks"],
            manifest["trajectories_per_family_per_block"],
            manifest["master_seed"],
            manifest["alpha"],
            manifest["endpoint_count"],
        )
        arrays_path = temp_dir / "seed_block_arrays.npz"
        write_lossless_arrays(arrays_path, records)
        round_trip = verify_lossless_arrays(arrays_path, records)
        elapsed = time.perf_counter() - start
        summary = {
            "experiment_id": "circa-d2-v1",
            "status": "PASS" if analysis["gate_pass"] else "SCIENTIFIC_GATE_FAIL",
            "execution_mode": "REMOTE_CPU_SHARED_SECOND_DYNAMICS_DEVELOPMENT",
            "scientific_run_count": 1,
            "retry_allowed": False,
            "manifest_sha256": sha256(manifest_path),
            "authorization_sha256": sha256(authorization_path),
            "benchmark_sha256": sha256(benchmark_path),
            "design": {
                "seed_blocks": manifest["seed_blocks"],
                "trajectories_per_family_per_block": manifest["trajectories_per_family_per_block"],
                "paired_trajectories": manifest["seed_blocks"] * manifest["trajectories_per_family_per_block"] * len(FAMILY_IDS),
                "families": len(FAMILY_IDS),
                "methods": len(METHODS),
                "horizon": benchmark["horizon"],
                "alpha": manifest["alpha"],
                "endpoint_count": manifest["endpoint_count"],
                "delta_star": manifest["delta_star"],
                "master_seed": manifest["master_seed"],
            },
            "analysis": analysis,
            "array_round_trip": round_trip,
            "runtime": {
                "elapsed_seconds": elapsed,
                "python": sys.version,
                "numpy": np.__version__,
                "platform": platform.platform(),
                "workers": 1,
                "gpu_used": False,
                "execution_class": "CPU-SHARED",
                "performance_claim_allowed": False,
            },
            "claim_boundary": "If all gates pass, D2 supports D1-informed second-dynamics development evidence only. It is not blind confirmation, Gazebo, formal/G2, sealed, hardware, or deployment evidence.",
        }
        summary_path = temp_dir / "result_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log_path = temp_dir / "run.log"
        log_path.write_text(
            f"status={summary['status']}\nelapsed_seconds={elapsed:.9f}\nrecords={records.size}\nretry_allowed=false\nexecution_class=CPU-SHARED\n",
            encoding="utf-8",
        )
        receipt = {
            "status": "COMPLETE",
            "files": {
                "result_summary.json": sha256(summary_path),
                "seed_block_arrays.npz": sha256(arrays_path),
                "run.log": sha256(log_path),
            },
        }
        receipt_path = temp_dir / "integrity_receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        total_bytes = sum(path.stat().st_size for path in temp_dir.iterdir() if path.is_file())
        if elapsed > manifest["wall_time_seconds_max"]:
            raise RuntimeError("D2 wall-time budget exceeded")
        if total_bytes > manifest["output_size_bytes_max"]:
            raise RuntimeError("D2 output-size budget exceeded")
        summary["runtime"]["output_bytes"] = total_bytes
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        receipt["files"]["result_summary.json"] = sha256(summary_path)
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_dir.replace(output_dir)
        return summary
    except BaseException:
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    result = run(
        (root / args.manifest).resolve(),
        (root / args.authorization).resolve(),
        root,
        (root / args.output).resolve(),
        sys.argv[1:],
    )
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["runtime"]["elapsed_seconds"]}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
