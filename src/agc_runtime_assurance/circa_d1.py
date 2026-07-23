"""Exactly-once CIRCA-D1 stochastic 1U1G development benchmark.

The benchmark is independent of the constructed exact-truth G0 fixture.  It
samples a frozen finite-support dynamic hazard model while retaining exact
population truth for calibration auditing.
"""

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
    hoeffding_radius,
    manski_bounds,
    simultaneous_confidence_interval,
    structured_bounds,
    verify_evidence_contract,
)


FAMILY_IDS = (
    "D1F1_overlap_audit",
    "D1F2_deterministic_censoring",
    "D1F3_model_uncertainty",
    "D1F4_covariate_shift",
    "D1F5_registered_1u1g_interference",
    "D1F6_null_no_effect",
)
POSITIVE_FAMILIES = FAMILY_IDS[:5]
NULL_FAMILY = FAMILY_IDS[5]
METHODS = (
    "factual_only",
    "censor_as_safe",
    "complete_case",
    "manski_worst_case",
    "ipw",
    "aipw_dr",
    "oracle_full_counterfactual",
    "circa_structured_bounds",
)
SENSITIVITY_MULTIPLIERS = (1.25, 1.50)
SCHEMA_DTYPE = np.dtype(
    [
        ("seed_block", "<i4"),
        ("family", "S44"),
        ("method", "S32"),
        ("n", "<i4"),
        ("estimate", "<f8"),
        ("identification_lower", "<f8"),
        ("identification_upper", "<f8"),
        ("confidence_lower", "<f8"),
        ("confidence_upper", "<f8"),
        ("width", "<f8"),
        ("population_truth_delta", "<f8"),
        ("population_active_risk", "<f8"),
        ("intervention_rate", "<f8"),
        ("witness_pinning_rate", "<f8"),
        ("covered", "u1"),
        ("certified_reduction", "u1"),
        ("simulator_calls", "<i4"),
        ("status", "S48"),
    ],
    align=False,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def schema_only_array(record_count: int) -> np.ndarray:
    if record_count <= 0:
        raise ValueError("record_count must be positive")
    records = np.zeros(record_count, dtype=SCHEMA_DTYPE)
    for name in (
        "estimate",
        "identification_lower",
        "identification_upper",
        "confidence_lower",
        "confidence_upper",
        "width",
        "population_truth_delta",
        "population_active_risk",
        "intervention_rate",
        "witness_pinning_rate",
    ):
        records[name].fill(np.nan)
    return records


def write_lossless_arrays(path: Path, records: np.ndarray) -> None:
    np.savez(path, records=records)


def verify_lossless_arrays(path: Path, expected: np.ndarray | None = None) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"records"}:
            raise ValueError("unexpected D1 array members")
        records = archive["records"]
    if records.dtype != SCHEMA_DTYPE:
        raise ValueError("D1 dtype mismatch")
    if expected is not None:
        if records.shape != expected.shape or records.tobytes() != expected.tobytes():
            raise ValueError("D1 lossless round trip failed")
    return {"records": int(records.size), "dtype_itemsize": int(records.dtype.itemsize)}


def load_benchmark(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("benchmark_id") != "circa-d1-stochastic-1u1g-v1" or spec.get("status") != "frozen":
        raise ValueError("benchmark is not the frozen CIRCA-D1 v1 specification")
    ids = tuple(f["id"] for f in spec.get("families", []))
    if ids != FAMILY_IDS or tuple(spec.get("positive_families", [])) != POSITIVE_FAMILIES:
        raise ValueError("family registry mismatch")
    if spec.get("null_family") != NULL_FAMILY or spec.get("horizon") != 24:
        raise ValueError("benchmark horizon/null registry mismatch")
    for family in spec["families"]:
        for key, length in (
            ("hazard_probs", 5),
            ("gust_probs", 3),
            ("congestion_probs", 3),
            ("sensor_probs", 3),
            ("delay_probs", 2),
            ("audit_coin_probs", 2),
        ):
            values = np.asarray(family[key], dtype=float)
            if values.size != length or np.any(values < 0) or not np.isclose(values.sum(), 1.0):
                raise ValueError(f"invalid probability vector {family['id']}:{key}")
        if family["verifier_radius"] <= 0 or family["trigger_threshold"] >= 1:
            raise ValueError("invalid verifier/trigger configuration")
    return spec


def _support(spec: dict, family: dict):
    support = spec["support"]
    probability_vectors = (
        family["hazard_probs"],
        family["gust_probs"],
        family["congestion_probs"],
        family["sensor_probs"],
        family["delay_probs"],
        family["audit_coin_probs"],
    )
    value_vectors = (
        support["hazard"],
        support["gust"],
        support["congestion"],
        support["sensor_error"],
        support["delay"],
        support["audit_coin"],
    )
    for indices in itertools.product(*(range(len(v)) for v in value_vectors)):
        probability = float(np.prod([probability_vectors[j][indices[j]] for j in range(6)]))
        if probability == 0.0:
            continue
        yield tuple(value_vectors[j][indices[j]] for j in range(6)), probability


def compile_family(spec: dict, family: dict, radius_multiplier: float = 1.0) -> dict[str, np.ndarray | float | str]:
    """Enumerate one frozen family without random sampling."""
    horizon = int(spec["horizon"])
    base_decision = int(spec["decision_step"])
    threshold = float(spec["risk_threshold"])
    tau = np.linspace(0.0, 1.0, horizon)
    rows: list[tuple] = []
    max_prediction_error = 0.0

    for (hazard, gust, congestion, sensor_error, delay, audit_coin), probability in _support(spec, family):
        coupling = float(family["registered_coupling"])
        u_shadow = (
            0.16
            + 0.13 * hazard
            + (0.32 + 0.10 * hazard) * tau
            + 0.08 * gust * (0.25 + 0.75 * tau)
            + coupling * congestion * tau
        )
        g_shadow = 0.12 + 0.10 * congestion + 0.22 * tau + 0.06 * gust * tau + 0.025 * hazard * tau

        observed_hazard = float(np.clip(hazard + family["sensor_scale"] * sensor_error, 0.0, 4.0))
        u_prediction = (
            0.16
            + 0.13 * observed_hazard
            + (0.32 + 0.10 * observed_hazard) * tau
            + coupling * congestion * tau
        )
        g_prediction = 0.12 + 0.10 * congestion + 0.22 * tau + 0.025 * observed_hazard * tau
        shadow_max = float(max(np.max(u_shadow), np.max(g_shadow)))
        prediction = float(max(np.max(u_prediction), np.max(g_prediction)))
        max_prediction_error = max(max_prediction_error, abs(shadow_max - prediction))

        eligible = prediction >= float(family["trigger_threshold"])
        if family["randomized_overlap"]:
            intervention = int(eligible and audit_coin == 1)
            pi0 = 1.0 if not eligible else float(family["audit_coin_probs"][0])
        else:
            intervention = int(eligible)
            pi0 = np.nan

        decision_step = min(horizon - 1, base_decision + int(delay))
        ramp = np.clip((np.arange(horizon) - decision_step) / max(1, horizon - 1 - decision_step), 0.0, 1.0)
        u_active = u_shadow - intervention * float(family["shield_strength"]) * ramp
        g_active = g_shadow + intervention * float(family["transfer_penalty"]) * ramp
        y0 = int(shadow_max >= threshold)
        y1 = int(max(np.max(u_active), np.max(g_active)) >= threshold)

        radius = float(family["verifier_radius"]) * radius_multiplier
        if intervention == 0:
            witness = OBSERVED_CONSISTENCY
        elif prediction - radius >= threshold:
            witness = INEVITABLE_VIOLATION
        elif prediction + radius < threshold:
            witness = NOMINAL_SAFETY
        else:
            witness = NO_WITNESS

        # Misspecified outcome regression used only by the overlap AIPW baseline.
        m0 = float(np.clip(0.04 + 0.17 * hazard + 0.05 * congestion + 0.04 * max(gust, 0), 0.0, 1.0))
        rows.append((probability, intervention, y0, y1, witness, pi0, m0, prediction, shadow_max))

    data = np.asarray(rows, dtype=float)
    probabilities = data[:, 0]
    probabilities /= probabilities.sum()
    result: dict[str, np.ndarray | float | str] = {
        "id": family["id"],
        "probability": probabilities,
        "r": data[:, 1].astype(np.int8),
        "y0": data[:, 2].astype(np.int8),
        "y1": data[:, 3].astype(np.int8),
        "witness": data[:, 4].astype(np.int8),
        "pi0": data[:, 5],
        "m0": data[:, 6],
        "prediction": data[:, 7],
        "shadow_max": data[:, 8],
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
    r = compiled["r"]
    y0 = compiled["y0"]
    witness = compiled["witness"]
    inevitable_bad = int(np.sum((r == 1) & (witness == INEVITABLE_VIOLATION) & (y0 != 1)))
    nominal_bad = int(np.sum((r == 1) & (witness == NOMINAL_SAFETY) & (y0 != 0)))
    observed_bad = int(np.sum((r == 0) & ((witness != OBSERVED_CONSISTENCY) | (compiled["y0"] != compiled["y1"]))))
    radius_sound = bool(compiled["max_prediction_error"] <= compiled["registered_radius"] + 1e-12)
    return {
        "family": compiled["id"],
        "support_points": int(len(r)),
        "max_prediction_error": float(compiled["max_prediction_error"]),
        "registered_radius": float(compiled["registered_radius"]),
        "radius_sound": radius_sound,
        "inevitable_witness_errors": inevitable_bad,
        "nominal_witness_errors": nominal_bad,
        "observed_consistency_errors": observed_bad,
        "passed": radius_sound and inevitable_bad == nominal_bad == observed_bad == 0,
    }


def _family_seed(master_seed: int, block: int, family_id: str) -> int:
    digest = hashlib.sha256(f"{master_seed}|{block}|{family_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _mean(counts: np.ndarray, values: np.ndarray) -> float:
    return float(counts @ np.asarray(values, dtype=float) / counts.sum())


def _point(point: float, values: np.ndarray, n: int, alpha: float, endpoint_count: int, target: float) -> dict:
    value_range = float(np.ptp(values)) or 1.0
    radius = hoeffding_radius(n, alpha, endpoint_count=endpoint_count, value_range=value_range)
    low, high = max(-1.0, point - radius), min(1.0, point + radius)
    return {
        "estimate": point,
        "id_low": point,
        "id_high": point,
        "conf_low": low,
        "conf_high": high,
        "width": 0.0,
        "covered": low <= target <= high,
    }


def _bound(low: float, high: float, n: int, alpha: float, endpoint_count: int, target: float) -> dict:
    conf_low, conf_high = simultaneous_confidence_interval(
        low, high, n, alpha, endpoint_count=endpoint_count, value_range=2.0
    )
    return {
        "estimate": (low + high) / 2.0,
        "id_low": low,
        "id_high": high,
        "conf_low": conf_low,
        "conf_high": conf_high,
        "width": high - low,
        "covered": conf_low <= target <= conf_high,
    }


def _empty() -> dict:
    return {
        "estimate": np.nan,
        "id_low": np.nan,
        "id_high": np.nan,
        "conf_low": np.nan,
        "conf_high": np.nan,
        "width": np.nan,
        "covered": False,
    }


def evaluate_methods(compiled: dict, counts: np.ndarray, alpha: float, endpoint_count: int) -> dict[str, tuple[dict, str]]:
    n = int(counts.sum())
    r, y0, y1 = compiled["r"], compiled["y0"], compiled["y1"]
    truth = float(compiled["truth_delta"])
    lm, um = manski_bounds(r, y1)
    lc, uc = structured_bounds(r, y1, compiled["witness"])
    manski_low, manski_high = _mean(counts, lm - y1), _mean(counts, um - y1)
    circa_low, circa_high = _mean(counts, lc - y1), _mean(counts, uc - y1)

    result: dict[str, tuple[dict, str]] = {}
    factual = _point(_mean(counts, y1), y1, n, alpha, endpoint_count, float(compiled["active_risk"]))
    factual["covered"] = False
    result["factual_only"] = (factual, "WRONG_ESTIMAND_ACTIVE_RISK")
    censor_values = (1 - r) * y1 - y1
    result["censor_as_safe"] = (
        _point(_mean(counts, censor_values), censor_values, n, alpha, endpoint_count, truth),
        "NAIVE_BASELINE",
    )
    complete_values = np.zeros_like(y1, dtype=float)
    result["complete_case"] = (
        _point(0.0, complete_values, n, alpha, endpoint_count, truth),
        "NAIVE_BASELINE",
    )
    result["manski_worst_case"] = (
        _bound(manski_low, manski_high, n, alpha, endpoint_count, truth),
        "VALID_BOUND",
    )
    if compiled["randomized_overlap"]:
        ipw_values = (1 - r) * y1 / compiled["pi0"] - y1
        aipw_values = compiled["m0"] + (1 - r) * (y1 - compiled["m0"]) / compiled["pi0"] - y1
        result["ipw"] = (
            _point(_mean(counts, ipw_values), ipw_values, n, alpha, endpoint_count, truth),
            "VALID_RANDOMIZED_OVERLAP",
        )
        result["aipw_dr"] = (
            _point(_mean(counts, aipw_values), aipw_values, n, alpha, endpoint_count, truth),
            "VALID_RANDOMIZED_OVERLAP",
        )
    else:
        result["ipw"] = (_empty(), "UNIDENTIFIABLE_PROPENSITY")
        result["aipw_dr"] = (_empty(), "UNIDENTIFIABLE_PROPENSITY")
    oracle_values = y0 - y1
    result["oracle_full_counterfactual"] = (
        _point(_mean(counts, oracle_values), oracle_values, n, alpha, endpoint_count, truth),
        "EVALUATION_ONLY_ORACLE",
    )
    result["circa_structured_bounds"] = (
        _bound(circa_low, circa_high, n, alpha, endpoint_count, truth),
        "VALID_BOUND",
    )
    return result


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


def simulate_d1(benchmark: dict, seed_blocks: int, n: int, master_seed: int, alpha: float, endpoint_count: int) -> tuple[np.ndarray, dict]:
    families_by_id = {f["id"]: f for f in benchmark["families"]}
    compiled = {f: compile_family(benchmark, families_by_id[f]) for f in FAMILY_IDS}
    audits = {f: audit_compiled_family(compiled[f]) for f in FAMILY_IDS}
    if not all(a["passed"] for a in audits.values()):
        raise RuntimeError("frozen witness support audit failed")

    records = schema_only_array(seed_blocks * len(FAMILY_IDS) * len(METHODS))
    total_counts = {f: np.zeros_like(compiled[f]["probability"], dtype=np.int64) for f in FAMILY_IDS}
    coverage_blocks: list[bool] = []
    false_cert_blocks: list[bool] = []
    cursor = 0

    for block in range(seed_blocks):
        block_coverage: list[bool] = []
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

    pooled: dict[str, dict] = {}
    for family_id in FAMILY_IDS:
        family = compiled[family_id]
        methods = evaluate_methods(family, total_counts[family_id], alpha, endpoint_count)
        pooled[family_id] = {
            "population_truth_delta": float(family["truth_delta"]),
            "population_active_risk": float(family["active_risk"]),
            "population_intervention_rate": float(family["intervention_rate"]),
            "population_witness_pinning_rate": float(family["witness_pinning_rate"]),
            "methods": {
                m: {**{k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in values.items()}, "status": status}
                for m, (values, status) in methods.items()
            },
        }

    sensitivity: dict[str, dict] = {}
    for multiplier in SENSITIVITY_MULTIPLIERS:
        key = f"radius_x_{multiplier:.2f}"
        family_lcbs = {}
        family_audits = {}
        for family_id in POSITIVE_FAMILIES:
            conservative = compile_family(benchmark, families_by_id[family_id], multiplier)
            if not np.array_equal(conservative["probability"], compiled[family_id]["probability"]):
                raise RuntimeError("sensitivity support ordering changed")
            family_audits[family_id] = audit_compiled_family(conservative)
            methods = evaluate_methods(conservative, total_counts[family_id], alpha, endpoint_count)
            family_lcbs[family_id] = float(methods["circa_structured_bounds"][0]["conf_low"])
        sensitivity[key] = {
            "family_lcbs": family_lcbs,
            "worst_positive_lcb": min(family_lcbs.values()),
            "witness_audits": family_audits,
        }

    # An explicitly inadmissible diagnostic demonstrates why the radius cannot be dropped.
    unverified = {}
    for family_id in FAMILY_IDS:
        family = compiled[family_id]
        aggressive = (family["r"] == 1) & (family["prediction"] >= 1.0) & (family["y0"] == 0)
        unverified[family_id] = {
            "unsound_support_points": int(np.sum(aggressive)),
            "unsound_probability_mass": float(family["probability"] @ aggressive),
            "admissible_for_certificate": False,
        }

    corruption_checks = _corruption_checks()
    coverage_successes = int(sum(coverage_blocks))
    false_certifications = int(sum(false_cert_blocks))
    primary_lcbs = {
        f: pooled[f]["methods"]["circa_structured_bounds"]["conf_low"] for f in POSITIVE_FAMILIES
    }
    width_ratios = {
        f: pooled[f]["methods"]["circa_structured_bounds"]["width"]
        / pooled[f]["methods"]["manski_worst_case"]["width"]
        for f in POSITIVE_FAMILIES
    }
    coverage_no_loss = all(
        pooled[f]["methods"]["circa_structured_bounds"]["covered"]
        and pooled[f]["methods"]["manski_worst_case"]["covered"]
        for f in POSITIVE_FAMILIES
    )
    gates = {
        "worst_positive_pooled_circa_lcb": min(primary_lcbs.values()),
        "worst_positive_pooled_circa_lcb_strictly_above_0_10": min(primary_lcbs.values()) > 0.10,
        "max_positive_population_active_risk": max(pooled[f]["population_active_risk"] for f in POSITIVE_FAMILIES),
        "max_positive_population_active_risk_at_most_0_15": max(
            pooled[f]["population_active_risk"] for f in POSITIVE_FAMILIES
        ) <= 0.15,
        "simultaneous_coverage_blocks": coverage_successes,
        "simultaneous_coverage_cp_lower": clopper_pearson_lower(coverage_successes, seed_blocks),
        "simultaneous_coverage_cp_lower_at_least_0_94": clopper_pearson_lower(coverage_successes, seed_blocks) >= 0.94,
        "null_false_certification_blocks": false_certifications,
        "null_false_certification_cp_upper": clopper_pearson_upper(false_certifications, seed_blocks),
        "null_false_certification_cp_upper_at_most_0_06": clopper_pearson_upper(false_certifications, seed_blocks) <= 0.06,
        "all_positive_width_ratios_strictly_below_one": all(v < 1.0 for v in width_ratios.values()),
        "circa_population_coverage_no_loss_vs_manski": coverage_no_loss,
        "all_sensitivity_settings_retain_primary": all(v["worst_positive_lcb"] > 0.10 for v in sensitivity.values()),
        "all_registered_witnesses_sound": all(a["passed"] for a in audits.values()),
        "all_corruptions_fail_closed": all(c["passed"] for c in corruption_checks),
    }
    gate_pass = all(
        gates[key]
        for key in (
            "worst_positive_pooled_circa_lcb_strictly_above_0_10",
            "max_positive_population_active_risk_at_most_0_15",
            "simultaneous_coverage_cp_lower_at_least_0_94",
            "null_false_certification_cp_upper_at_most_0_06",
            "all_positive_width_ratios_strictly_below_one",
            "circa_population_coverage_no_loss_vs_manski",
            "all_sensitivity_settings_retain_primary",
            "all_registered_witnesses_sound",
            "all_corruptions_fail_closed",
        )
    )
    return records, {
        "gate_pass": bool(gate_pass),
        "gates": gates,
        "primary_family_lcbs": primary_lcbs,
        "width_ratios_vs_manski": width_ratios,
        "pooled": pooled,
        "witness_audits": audits,
        "sensitivity": sensitivity,
        "unverified_witness_ablation": unverified,
        "corruption_checks": corruption_checks,
    }


def validate_manifest(manifest: dict, repo_root: Path) -> None:
    expected = {
        "manifest_id": "circa-d1-v1",
        "status": "frozen_authorized_one_shot",
        "authorized": True,
        "retry_allowed": False,
        "seed_blocks": 50,
        "trajectories_per_family_per_block": 1000,
        "families": list(FAMILY_IDS),
        "methods": list(METHODS),
        "alpha": 0.05,
        "endpoint_count": 12,
        "delta_star": 0.10,
        "cpu_only": True,
        "workers": 1,
        "gpu_count": 0,
        "wall_time_seconds_max": 900,
        "output_size_bytes_max": 64 * 1024 * 1024,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"manifest mismatch for {key}")
    required_false = ("gazebo", "pilot", "formal", "g2", "sealed", "hardware", "upload", "external_data")
    if any(manifest.get("authorizations", {}).get(key) is not False for key in required_false):
        raise ValueError("out-of-scope authorization is open")
    for rel, expected_hash in manifest["integrity"]["files"].items():
        path = (repo_root / rel).resolve()
        if not path.is_relative_to(repo_root) or not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"integrity mismatch: {rel}")


def run(manifest_path: Path, authorization_path: Path, repo_root: Path, output_path: Path, actual_argv: list[str]) -> dict:
    start = time.perf_counter()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"):
        raise ValueError("CUDA_VISIBLE_DEVICES must be absent, empty, or -1")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, repo_root)
    if authorization.get("status") != "authorized_exactly_once" or authorization.get("retry_allowed") is not False:
        raise ValueError("launch authorization is not exactly once")
    if authorization.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("launch manifest hash mismatch")
    if Path(sys.executable).resolve() != Path(authorization.get("python_executable", "")).resolve():
        raise ValueError("Python executable differs from authorization")
    if authorization.get("exact_argv") != actual_argv:
        raise ValueError("argv differs from authorization")
    expected_output = (repo_root / manifest["output_paths"]["summary"]).resolve()
    if output_path.resolve() != expected_output:
        raise ValueError("output path differs from manifest")

    output_dir = output_path.parent
    temp_dir = output_dir.with_name(output_dir.name + ".tmp")
    if output_dir.exists() or temp_dir.exists():
        raise FileExistsError("CIRCA-D1 output or partial output already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir()

    try:
        benchmark_path = (repo_root / manifest["benchmark_path"]).resolve()
        benchmark = load_benchmark(benchmark_path)
        records, analysis = simulate_d1(
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
            "experiment_id": "circa-d1-v1",
            "status": "PASS" if analysis["gate_pass"] else "SCIENTIFIC_GATE_FAIL",
            "execution_mode": "CPU_ONLY_INDEPENDENT_STOCHASTIC_1U1G_DEVELOPMENT",
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
            },
            "claim_boundary": "Positive results, if gates pass, support only an independent stochastic development-benchmark CIRCA efficacy claim. They do not establish Gazebo, natural-system, formal/G2, sealed, pilot, hardware, or field efficacy.",
        }
        summary_path = temp_dir / "result_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log_path = temp_dir / "run.log"
        log_path.write_text(
            f"status={summary['status']}\nelapsed_seconds={elapsed:.9f}\nrecords={records.size}\nretry_allowed=false\n",
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
            raise RuntimeError("wall-time budget exceeded")
        if total_bytes > manifest["output_size_bytes_max"]:
            raise RuntimeError("output-size budget exceeded")
        summary["runtime"]["output_bytes"] = total_bytes
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        receipt["files"]["result_summary.json"] = sha256(summary_path)
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_dir.replace(output_dir)
        return summary
    except BaseException:
        # Partial evidence is intentionally retained; v1 may never be retried.
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
