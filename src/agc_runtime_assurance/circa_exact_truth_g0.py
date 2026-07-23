"""Exactly-once, CPU-only CIRCA exact-truth G0 runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from circa import (  # noqa: E402
    clopper_pearson_lower,
    clopper_pearson_upper,
    hoeffding_radius,
    simultaneous_confidence_interval,
    structured_bounds,
    manski_bounds,
    verify_evidence_contract,
)
from circa_fixtures import compile_atomic_families, load_fixture  # noqa: E402


FAMILY_IDS = (
    "F1_randomized_overlap_no_witness",
    "F2_deterministic_intervention_valid_witnesses",
    "F3_deterministic_no_support_no_witness",
    "F4_registered_1u1g_interference",
    "F5_unregistered_interference_corruption",
    "F6_provenance_and_outcome_corruptions",
)
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
SCHEMA_DTYPE = np.dtype(
    [
        ("replication", "<i4"),
        ("family", "S52"),
        ("method", "S32"),
        ("estimate", "<f8"),
        ("identification_lower", "<f8"),
        ("identification_upper", "<f8"),
        ("confidence_lower", "<f8"),
        ("confidence_upper", "<f8"),
        ("width", "<f8"),
        ("covered", "u1"),
        ("certified_reduction", "u1"),
        ("intervention_rate", "<f8"),
        ("witness_pinning_rate", "<f8"),
        ("truth_delta", "<f8"),
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
    result = np.zeros(record_count, dtype=SCHEMA_DTYPE)
    for name in (
        "estimate",
        "identification_lower",
        "identification_upper",
        "confidence_lower",
        "confidence_upper",
        "width",
        "intervention_rate",
        "witness_pinning_rate",
        "truth_delta",
    ):
        result[name].fill(np.nan)
    return result


def write_lossless_arrays(path: Path, records: np.ndarray) -> None:
    np.savez(path, records=records)


def verify_lossless_arrays(path: Path, expected: np.ndarray | None = None) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"records"}:
            raise ValueError("unexpected array archive members")
        records = archive["records"]
    if records.dtype != SCHEMA_DTYPE:
        raise ValueError("array dtype does not match frozen schema")
    if expected is not None:
        if records.shape != expected.shape or records.dtype != expected.dtype or records.tobytes() != expected.tobytes():
            raise ValueError("lossless array round trip failed")
    return {"records": int(records.size), "dtype_itemsize": int(records.dtype.itemsize)}


def _family_seed(master_seed: int, family_id: str) -> int:
    digest = hashlib.sha256(f"{master_seed}|{family_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _means(counts: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    return counts @ np.asarray(values, dtype=float) / float(n)


def _point_method(point: np.ndarray, truth: float, value_range: float, n: int, alpha: float) -> dict:
    radius = hoeffding_radius(n, alpha, endpoint_count=8, value_range=value_range)
    low = np.maximum(-1.0, point - radius)
    high = np.minimum(1.0, point + radius)
    return {
        "estimate": point,
        "id_low": point,
        "id_high": point,
        "conf_low": low,
        "conf_high": high,
        "width": np.zeros_like(point),
        "covered": (low <= truth) & (truth <= high),
    }


def _bound_method(low: np.ndarray, high: np.ndarray, truth: float, n: int, alpha: float) -> dict:
    conf_low = np.empty_like(low)
    conf_high = np.empty_like(high)
    for i in range(low.size):
        conf_low[i], conf_high[i] = simultaneous_confidence_interval(float(low[i]), float(high[i]), n, alpha)
    return {
        "estimate": (low + high) / 2.0,
        "id_low": low,
        "id_high": high,
        "conf_low": conf_low,
        "conf_high": conf_high,
        "width": high - low,
        "covered": (conf_low <= truth) & (truth <= conf_high),
    }


def _empty_method(replications: int) -> dict:
    nan = np.full(replications, np.nan)
    return {
        "estimate": nan.copy(),
        "id_low": nan.copy(),
        "id_high": nan.copy(),
        "conf_low": nan.copy(),
        "conf_high": nan.copy(),
        "width": nan.copy(),
        "covered": np.zeros(replications, dtype=bool),
    }


def simulate_records(fixture: dict, replications: int, n: int, master_seed: int, alpha: float) -> tuple[np.ndarray, dict]:
    compiled = compile_atomic_families(fixture)
    record_count = replications * len(FAMILY_IDS) * len(METHODS)
    records = schema_only_array(record_count)
    cursor = 0
    family_summary: dict[str, dict] = {}
    circa_coverage_by_family: dict[str, np.ndarray] = {}
    f1_false_cert = None

    for family_id in FAMILY_IDS:
        truth = float(next(f for f in fixture["families"] if f["id"] == family_id).get("truth", {}).get("delta", np.nan))
        if family_id in compiled:
            family = compiled[family_id]
            rng = np.random.Generator(np.random.PCG64(_family_seed(master_seed, family_id)))
            counts = rng.multinomial(n, family.probabilities, size=replications)
            intervention_rate = _means(counts, family.r, n)
            pinning = _means(counts, ((family.r == 1) & (family.witness != 0)).astype(float), n)

            lm, um = manski_bounds(family.r, family.y1)
            lc, uc = structured_bounds(family.r, family.y1, family.witness)
            manski_low = _means(counts, lm - family.y1, n)
            manski_high = _means(counts, um - family.y1, n)
            circa_low = _means(counts, lc - family.y1, n)
            circa_high = _means(counts, uc - family.y1, n)

            method_values: dict[str, tuple[dict, str]] = {}
            method_values["factual_only"] = (_empty_method(replications), "WRONG_ESTIMAND")
            censor_value = (1 - family.r) * family.y1 - family.y1
            method_values["censor_as_safe"] = (
                _point_method(_means(counts, censor_value, n), truth, float(np.ptp(censor_value)) or 1.0, n, alpha),
                "NAIVE_BASELINE",
            )
            method_values["complete_case"] = (
                _point_method(np.zeros(replications), truth, 1.0, n, alpha),
                "NAIVE_BASELINE",
            )
            method_values["manski_worst_case"] = (_bound_method(manski_low, manski_high, truth, n, alpha), "VALID_BOUND")
            if family_id == FAMILY_IDS[0]:
                ipw_values = (1 - family.r) * family.y1 / family.pi0 - family.y1
                aipw_values = family.m0 + (1 - family.r) * (family.y1 - family.m0) / family.pi0 - family.y1
                method_values["ipw"] = (
                    _point_method(_means(counts, ipw_values, n), truth, float(np.ptp(ipw_values)), n, alpha),
                    "VALID_RANDOMIZED_OVERLAP",
                )
                method_values["aipw_dr"] = (
                    _point_method(_means(counts, aipw_values, n), truth, float(np.ptp(aipw_values)), n, alpha),
                    "VALID_RANDOMIZED_OVERLAP",
                )
            else:
                method_values["ipw"] = (_empty_method(replications), "UNIDENTIFIABLE_PROPENSITY")
                method_values["aipw_dr"] = (_empty_method(replications), "UNIDENTIFIABLE_PROPENSITY")
            oracle_values = family.y0 - family.y1
            method_values["oracle_full_counterfactual"] = (
                _point_method(_means(counts, oracle_values, n), truth, float(np.ptp(oracle_values)) or 1.0, n, alpha),
                "EVALUATION_ONLY_ORACLE",
            )
            circa_status = "BOUND_TOO_WIDE" if family_id == FAMILY_IDS[2] else "VALID_BOUND"
            method_values["circa_structured_bounds"] = (_bound_method(circa_low, circa_high, truth, n, alpha), circa_status)

            for method in METHODS:
                values, status = method_values[method]
                sl = slice(cursor, cursor + replications)
                records["replication"][sl] = np.arange(replications, dtype=np.int32)
                records["family"][sl] = family_id.encode("ascii")
                records["method"][sl] = method.encode("ascii")
                records["estimate"][sl] = values["estimate"]
                records["identification_lower"][sl] = values["id_low"]
                records["identification_upper"][sl] = values["id_high"]
                records["confidence_lower"][sl] = values["conf_low"]
                records["confidence_upper"][sl] = values["conf_high"]
                records["width"][sl] = values["width"]
                records["covered"][sl] = values["covered"].astype(np.uint8)
                records["certified_reduction"][sl] = (values["conf_low"] > float(fixture["delta_star"])).astype(np.uint8)
                records["intervention_rate"][sl] = intervention_rate
                records["witness_pinning_rate"][sl] = pinning
                records["truth_delta"][sl] = truth
                records["status"][sl] = status.encode("ascii")
                cursor += replications

            circa_values = method_values["circa_structured_bounds"][0]
            circa_coverage_by_family[family_id] = circa_values["covered"]
            if family_id == FAMILY_IDS[0]:
                f1_false_cert = circa_values["conf_low"] > float(fixture["delta_star"])
            family_summary[family_id] = {
                "truth_delta": truth,
                "population_manski_interval": family.truth["manski_interval"],
                "population_circa_interval": family.truth["circa_interval"],
                "population_width_ratio": family.truth["width_ratio"],
                "circa_empirical_coverage": float(np.mean(circa_values["covered"])),
                "mean_sample_circa_width": float(np.mean(circa_values["width"])),
            }
        else:
            status = "INTERFERENCE_UNMODELED" if family_id == FAMILY_IDS[4] else "CORRUPTION_FIXTURE_REJECTED"
            for method in METHODS:
                sl = slice(cursor, cursor + replications)
                records["replication"][sl] = np.arange(replications, dtype=np.int32)
                records["family"][sl] = family_id.encode("ascii")
                records["method"][sl] = method.encode("ascii")
                records["status"][sl] = status.encode("ascii")
                cursor += replications
            family_summary[family_id] = {"numeric_certificate_forbidden": True, "status": status}

    if cursor != record_count:
        raise RuntimeError("record count mismatch")
    simultaneous = np.logical_and.reduce([circa_coverage_by_family[f] for f in FAMILY_IDS[:4]])
    coverage_successes = int(simultaneous.sum())
    false_successes = int(np.asarray(f1_false_cert).sum())
    corruption_checks = _corruption_checks()
    gates = {
        "population_identification_soundness_all_F1_to_F4": all(
            f.truth["circa_interval"][0] <= f.truth["delta"] <= f.truth["circa_interval"][1]
            for f in compiled.values()
        ),
        "simultaneous_coverage_empirical": float(np.mean(simultaneous)),
        "simultaneous_coverage_cp_lower": clopper_pearson_lower(coverage_successes, replications),
        "F2_population_width_ratio": float(compiled[FAMILY_IDS[1]].truth["width_ratio"]),
        "F4_population_width_ratio": float(compiled[FAMILY_IDS[3]].truth["width_ratio"]),
        "false_certification_empirical_F1": false_successes / replications,
        "false_certification_cp_upper_F1": clopper_pearson_upper(false_successes, replications),
        "all_registered_corruptions_fail_closed": all(c["passed"] for c in corruption_checks),
    }
    gate_pass = (
        gates["population_identification_soundness_all_F1_to_F4"]
        and gates["simultaneous_coverage_empirical"] >= 0.945
        and gates["simultaneous_coverage_cp_lower"] >= 0.94
        and gates["F2_population_width_ratio"] <= 0.75
        and gates["F4_population_width_ratio"] < 1.0
        and gates["false_certification_cp_upper_F1"] <= 0.06
        and gates["all_registered_corruptions_fail_closed"]
    )
    return records, {
        "family_summary": family_summary,
        "gates": gates,
        "gate_pass": bool(gate_pass),
        "coverage_successes": coverage_successes,
        "false_certification_successes": false_successes,
        "corruption_checks": corruption_checks,
    }


def _corruption_checks() -> list[dict]:
    specs = [
        ("F5_unregistered_interference_corruption", {"interference_registered": False}, "INTERFERENCE_UNMODELED"),
        ("bad_witness_hash", {"witness_hash_matches": False}, "INVALID_WITNESS"),
        ("nominal_policy_hash_mismatch", {"policy_hash_matches": False}, "INVALID_PROVENANCE"),
        ("constraint_hash_mismatch", {"constraint_hash_matches": False}, "INVALID_PROVENANCE"),
        ("horizon_mismatch", {"horizon_matches": False}, "INVALID_PROVENANCE"),
        ("early_termination_labeled_safe", {"outcome_complete": False}, "OUTCOME_CENSORED_INVALIDLY"),
        ("contradictory_L1_U0_witnesses", {"witnesses_contradictory": True}, "INVALID_WITNESS"),
    ]
    results = []
    for case_id, kwargs, expected in specs:
        check = verify_evidence_contract(**kwargs)
        results.append(
            {
                "case_id": case_id,
                "expected_status": expected,
                "observed_status": check.status,
                "numeric_certificate_allowed": check.numeric_certificate_allowed,
                "passed": check.status == expected and not check.numeric_certificate_allowed,
            }
        )
    return results


def validate_manifest(manifest: dict, repo_root: Path) -> None:
    expected = {
        "manifest_id": "circa-exact-truth-g0-v1",
        "status": "frozen_authorized_one_shot",
        "authorized": True,
        "retry_allowed": False,
        "replications": 5000,
        "trajectories_per_replication": 1000,
        "families": list(FAMILY_IDS),
        "methods": list(METHODS),
        "cpu_only": True,
        "gpu_count": 0,
        "wall_time_seconds_max": 1800,
        "output_size_bytes_max": 128 * 1024 * 1024,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"manifest mismatch for {key}")
    exclusions = manifest.get("authorizations", {})
    required_false = ("development", "g1", "gazebo", "pilot", "formal", "g2", "sealed", "hardware", "upload")
    if any(exclusions.get(key) is not False for key in required_false):
        raise ValueError("out-of-scope experiment authorization is not closed")
    for rel, expected_hash in manifest["integrity"]["files"].items():
        path = (repo_root / rel).resolve()
        if not path.is_relative_to(repo_root) or sha256(path) != expected_hash:
            raise ValueError(f"integrity mismatch: {rel}")


def run(manifest_path: Path, authorization_path: Path, repo_root: Path, output_path: Path, actual_argv: list[str]) -> dict:
    start = time.perf_counter()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"):
        raise ValueError("CUDA_VISIBLE_DEVICES must be absent, empty, or -1")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, repo_root)
    if authorization.get("status") != "authorized_exactly_once" or authorization.get("retry_allowed") is not False:
        raise ValueError("launch authorization is not exactly-once")
    if authorization.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("launch authorization manifest hash mismatch")
    if Path(sys.executable).resolve() != Path(authorization.get("python_executable", "")).resolve():
        raise ValueError("actual Python executable differs from frozen authorization")
    if authorization.get("exact_argv") != actual_argv:
        raise ValueError("actual argv differs from frozen authorization")

    expected_output = (repo_root / manifest["output_paths"]["summary"]).resolve()
    if output_path.resolve() != expected_output:
        raise ValueError("output path differs from frozen manifest")
    output_dir = output_path.parent
    temp_dir = output_dir.with_name(output_dir.name + ".tmp")
    if output_dir.exists() or temp_dir.exists():
        raise FileExistsError("exactly-once output or temp directory already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir()

    try:
        fixture_path = (repo_root / manifest["fixture_path"]).resolve()
        fixture = load_fixture(fixture_path)
        records, analysis = simulate_records(
            fixture,
            manifest["replications"],
            manifest["trajectories_per_replication"],
            manifest["master_seed"],
            manifest["alpha"],
        )
        arrays_path = temp_dir / "replication_arrays.npz"
        write_lossless_arrays(arrays_path, records)
        round_trip = verify_lossless_arrays(arrays_path, records)
        elapsed = time.perf_counter() - start
        summary = {
            "experiment_id": "circa-exact-truth-g0-v1",
            "status": "PASS" if analysis["gate_pass"] else "SCIENTIFIC_GATE_FAIL",
            "execution_mode": "CPU_ONLY_EXACT_TRUTH_CONSTRUCTED_FIXTURE",
            "scientific_run_count": 1,
            "retry_allowed": False,
            "manifest_sha256": sha256(manifest_path),
            "authorization_sha256": sha256(authorization_path),
            "fixture_sha256": sha256(fixture_path),
            "design": {
                "replications": manifest["replications"],
                "trajectories_per_replication": manifest["trajectories_per_replication"],
                "families": len(FAMILY_IDS),
                "methods": len(METHODS),
                "records": int(records.size),
                "alpha": manifest["alpha"],
                "delta_star": fixture["delta_star"],
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
            "claim_boundary": "Constructed exact-truth mechanism, coverage, and refusal G0 only; no development/G1, Gazebo, efficacy, natural-system, pilot, formal, G2, sealed, upload, GPU, hardware, or RESS performance claim.",
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
                "replication_arrays.npz": sha256(arrays_path),
                "run.log": sha256(log_path),
            },
        }
        receipt_path = temp_dir / "integrity_receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        total_bytes = sum(p.stat().st_size for p in temp_dir.iterdir() if p.is_file())
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
        # Preserve any partial evidence.  Exactly-once semantics forbid cleanup-and-retry.
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
