"""One-shot runner for the unsealed assurance-case corruption benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

from .assurance_case import (
    build_valid_assurance_case,
    inject_assurance_fault,
    verify_assurance_case,
)
from .g1a_runner import canonical_code_tree_hash


class AssuranceBenchmarkError(RuntimeError):
    """Raised when the frozen benchmark contract is violated."""


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_file(root: Path, relative: Any, expected: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise AssuranceBenchmarkError(f"{label} path is missing")
    if not isinstance(expected, str) or len(expected) != 64:
        raise AssuranceBenchmarkError(f"{label} hash is invalid")
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise AssuranceBenchmarkError(f"{label} path escapes repo root")
    if not candidate.is_file() or _digest(candidate) != expected.lower():
        raise AssuranceBenchmarkError(f"{label} hash mismatch")
    return candidate


def _validate_manifest(manifest: dict[str, Any], root: Path) -> None:
    required_exact = {
        "experiment_family": "assurance_case_corruption_benchmark",
        "stage": "unsealed_development",
        "execution_authorized": True,
        "sealed_data_allowed": False,
        "formal_experiment": False,
        "claim_generation_allowed": False,
        "run_count": 1,
        "scientific_failure_retry_allowed": False,
    }
    for key, expected in required_exact.items():
        if manifest.get(key) != expected:
            raise AssuranceBenchmarkError(f"manifest {key} must equal {expected!r}")
    if manifest.get("code_tree_sha256") != canonical_code_tree_hash(root):
        raise AssuranceBenchmarkError("current src+tests code tree does not match manifest")
    for prefix in ("authorization", "route_decision", "protocol"):
        _bound_file(
            root, manifest.get(f"{prefix}_path"), manifest.get(f"{prefix}_sha256"), prefix,
        )

    controls = manifest.get("valid_controls")
    families = manifest.get("fault_families")
    if controls != ["nominal", "filtered", "backup", "recovery"]:
        raise AssuranceBenchmarkError("valid controls must cover the four frozen modes")
    if not isinstance(families, dict) or not families:
        raise AssuranceBenchmarkError("fault_families must be a non-empty object")
    expected_families = {
        "missing_required_field", "malformed_digest", "expired_action",
        "monotonic_time_reversal", "latency_fingerprint_mismatch",
        "backup_invariant_mismatch", "constraint_contract_mismatch",
        "audit_chain_tamper",
    }
    if set(families) != expected_families:
        raise AssuranceBenchmarkError("fault family set differs from frozen protocol")
    if any(variants != [0, 1, 2] for variants in families.values()):
        raise AssuranceBenchmarkError("every fault family must freeze variants [0,1,2]")
    case_count = len(controls) + sum(len(value) for value in families.values())
    budget = manifest.get("budget", {})
    if not isinstance(budget, dict) or case_count > budget.get("max_cases", -1):
        raise AssuranceBenchmarkError("case count exceeds budget")
    if budget.get("max_runtime_seconds", 0) <= 0 or budget.get("max_output_bytes", 0) <= 0:
        raise AssuranceBenchmarkError("runtime/output budgets must be positive")
    if not isinstance(manifest.get("randomization_seed"), int):
        raise AssuranceBenchmarkError("randomization_seed must be an integer")


def run_assurance_case_benchmark(
    manifest_path: str | Path, repo_root: str | Path, output_path: str | Path,
) -> dict[str, Any]:
    """Run the frozen cases once and atomically write a bounded JSON result."""

    root = Path(repo_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    output_file = Path(output_path).resolve()
    if output_file.exists():
        raise AssuranceBenchmarkError("output already exists; scientific rerun refused")
    manifest_raw = manifest_file.read_bytes()
    manifest = json.loads(manifest_raw.decode("utf-8"))
    _validate_manifest(manifest, root)

    schedule: list[dict[str, Any]] = []
    for index, mode in enumerate(manifest["valid_controls"]):
        schedule.append({"kind": "control", "mode": mode, "variant": index})
    for family, variants in manifest["fault_families"].items():
        for variant in variants:
            schedule.append({"kind": "fault", "family": family, "variant": variant})
    random.Random(manifest["randomization_seed"]).shuffle(schedule)

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for order, case in enumerate(schedule):
        if time.perf_counter() - started > manifest["budget"]["max_runtime_seconds"]:
            raise AssuranceBenchmarkError("runtime budget exceeded")
        if case["kind"] == "control":
            case_id = f"control-{case['mode']}-{case['variant']}"
            bundle = build_valid_assurance_case(
                case_id=case_id, decision_mode=case["mode"], issued_at=10.0 + order,
            )
            expected_reason = "accepted"
        else:
            case_id = f"fault-{case['family']}-{case['variant']}"
            valid = build_valid_assurance_case(case_id=case_id, issued_at=10.0 + order)
            bundle = inject_assurance_fault(
                valid, family=case["family"], variant=case["variant"],
            )
            expected_reason = case["family"]
        verification = verify_assurance_case(bundle)
        rows.append({
            "run_order": order,
            "case_id": case_id,
            "kind": case["kind"],
            "family": case.get("family", "valid_control"),
            "variant": case["variant"],
            "accepted": verification.accepted,
            "detected": not verification.accepted if case["kind"] == "fault" else None,
            "pre_execution_blocked": verification.pre_execution_blocked,
            "actual_reason": verification.reason_code,
            "expected_reason": expected_reason,
            "reason_localized": verification.reason_code == expected_reason,
            "bundle_sha256": verification.bundle_sha256,
        })

    families: dict[str, dict[str, Any]] = {}
    for family in manifest["fault_families"]:
        family_rows = [row for row in rows if row["family"] == family]
        detected = sum(bool(row["detected"]) for row in family_rows)
        localized = sum(bool(row["reason_localized"]) for row in family_rows)
        families[family] = {
            "cases": len(family_rows),
            "detected": detected,
            "undetected_fault_rate": 1.0 - detected / len(family_rows),
            "reason_localization_accuracy": localized / len(family_rows),
        }
    control_rows = [row for row in rows if row["kind"] == "control"]
    fault_rows = [row for row in rows if row["kind"] == "fault"]
    summary = {
        "valid_bundle_acceptance_rate": sum(row["accepted"] for row in control_rows) / len(control_rows),
        "pre_execution_block_rate": sum(row["pre_execution_blocked"] for row in fault_rows) / len(fault_rows),
        "reason_localization_accuracy": sum(row["reason_localized"] for row in fault_rows) / len(fault_rows),
        "worst_family_undetected_fault_rate": max(
            value["undetected_fault_rate"] for value in families.values()
        ),
        "all_success_gates_passed": (
            all(row["accepted"] and row["reason_localized"] for row in control_rows)
            and all(row["detected"] and row["pre_execution_blocked"] and row["reason_localized"] for row in fault_rows)
        ),
    }
    result = {
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "code_tree_sha256": manifest["code_tree_sha256"],
        "randomization_seed": manifest["randomization_seed"],
        "schedule_case_ids": [row["case_id"] for row in rows],
        "case_count": len(rows),
        "valid_control_count": len(control_rows),
        "fault_case_count": len(fault_rows),
        "families": families,
        "summary": summary,
        "cases": rows,
        "duration_seconds": time.perf_counter() - started,
        "sealed_data_used": False,
        "formal_experiment_run": False,
        "claim_generation_allowed": False,
        "inference_boundary": (
            "Cases are deterministic mechanism checks, not independent policy seeds or a population fault-rate sample."
        ),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    if len(encoded) > manifest["budget"]["max_output_bytes"]:
        raise AssuranceBenchmarkError("output budget exceeded")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(output_file)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_assurance_case_benchmark(args.manifest, args.repo_root, args.output)


if __name__ == "__main__":
    main()
