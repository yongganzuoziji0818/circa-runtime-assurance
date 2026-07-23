"""One-shot known-probability coverage G0 for paired risk-reduction bounds.

This is synthetic method validation only.  It does not execute a controller,
Gazebo, a sealed experiment, or an efficacy comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np


class SyntheticCoverageError(RuntimeError):
    pass


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probabilities(values: Any, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (4,) or not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise SyntheticCoverageError(f"{label} must contain four positive finite probabilities")
    if not math.isclose(float(np.sum(array)), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise SyntheticCoverageError(f"{label} probabilities must sum to one")
    return array


def validate_manifest(manifest: dict[str, Any], root: Path) -> None:
    required = {
        "stage": "synthetic_known_probability_g0",
        "status": "frozen_authorized_one_shot",
        "authorized": True,
        "scientific_efficacy_experiment": False,
        "sealed_data_authorized": False,
        "formal_experiment_authorized": False,
        "pilot_authorized": False,
        "g2_authorized": False,
        "gpu_count": 0,
        "workers": 1,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise SyntheticCoverageError(f"manifest {key} must equal {expected!r}")
    if manifest.get("old_gazebo_g1_touched") is not False:
        raise SyntheticCoverageError("old Gazebo G1 boundary is not closed")
    alpha = manifest.get("alpha")
    rho = manifest.get("rho")
    if not isinstance(alpha, (int, float)) or not 0.0 < alpha < 1.0:
        raise SyntheticCoverageError("alpha must lie in (0, 1)")
    if not isinstance(rho, (int, float)) or not 0.0 < rho <= 1.0:
        raise SyntheticCoverageError("rho must lie in (0, 1]")
    replications = manifest.get("replications")
    sample_sizes = manifest.get("sample_sizes")
    if not isinstance(replications, int) or not 1000 <= replications <= 100000:
        raise SyntheticCoverageError("replications are outside the frozen G0 range")
    if not isinstance(sample_sizes, list) or not sample_sizes or any(not isinstance(n, int) or n < 2 for n in sample_sizes):
        raise SyntheticCoverageError("sample_sizes must be integers of at least two")
    minimum_coverage = manifest.get("minimum_empirical_coverage")
    maximum_bias = manifest.get("maximum_absolute_bias")
    if not isinstance(minimum_coverage, (int, float)) or not 0.0 < minimum_coverage <= 1.0:
        raise SyntheticCoverageError("minimum_empirical_coverage must lie in (0, 1]")
    if not isinstance(maximum_bias, (int, float)) or not 0.0 <= maximum_bias <= 1.0:
        raise SyntheticCoverageError("maximum_absolute_bias must lie in [0, 1]")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) < 2:
        raise SyntheticCoverageError("at least two registered synthetic families are required")
    names = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("family"), str) or not case["family"]:
            raise SyntheticCoverageError("each case needs a family name")
        names.append(case["family"])
        _probabilities(case.get("nominal"), f"{case['family']} nominal")
        _probabilities(case.get("surrogate"), f"{case['family']} surrogate")
    if len(set(names)) != len(names):
        raise SyntheticCoverageError("synthetic family names must be unique")
    for field, label in (
        ("core_path", "statistical core"),
        ("runner_path", "coverage runner"),
        ("resource_path", "resource receipt"),
    ):
        relative = manifest.get(field)
        expected = manifest.get(field.replace("_path", "_sha256"))
        if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
            raise SyntheticCoverageError(f"{label} binding is incomplete")
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file() or _digest(path) != expected.lower():
            raise SyntheticCoverageError(f"{label} hash mismatch")
    budgets = manifest.get("budgets")
    if not isinstance(budgets, dict):
        raise SyntheticCoverageError("budgets are missing")
    if budgets.get("max_runtime_seconds") != 60 or budgets.get("max_output_bytes") != 2097152:
        raise SyntheticCoverageError("synthetic G0 budgets differ from the frozen values")


def _case_statistics(
    rng: np.random.Generator,
    nominal: np.ndarray,
    surrogate: np.ndarray,
    *,
    rho: float,
    alpha_family: float,
    sample_size: int,
    replications: int,
) -> dict[str, Any]:
    proposal = rho * nominal + (1.0 - rho) * surrogate
    weights = nominal / proposal
    if np.any(weights > 1.0 / rho + 1e-12):
        raise SyntheticCoverageError("defensive-mixture weight bound failed")
    differences = np.array([0.0, 1.0, -1.0, 0.0])
    z = weights * differences
    counts = rng.multinomial(sample_size, proposal, size=replications)
    estimates = counts @ z / sample_size
    second_moments = counts @ (z * z)
    variances = (second_moments - sample_size * estimates * estimates) / (sample_size - 1)
    variances = np.maximum(variances, 0.0)
    log_term = math.log(2.0 / alpha_family)
    radii = np.sqrt(2.0 * variances * log_term / sample_size) + 14.0 * log_term / (
        3.0 * rho * (sample_size - 1)
    )
    lower = np.maximum(-1.0, estimates - radii)
    truth = float(nominal[1] - nominal[2])
    covered = lower <= truth + 1e-12
    return {
        "true_risk_reduction": truth,
        "mean_estimate": float(np.mean(estimates)),
        "bias": float(np.mean(estimates) - truth),
        "empirical_coverage": float(np.mean(covered)),
        "mean_lower_bound": float(np.mean(lower)),
        "maximum_weight": float(np.max(weights)),
        "proposal_probabilities": proposal.tolist(),
        "nominal_probabilities": nominal.tolist(),
        "surrogate_probabilities": surrogate.tolist(),
        "coverage_flags": covered,
    }


def simulate_coverage(manifest: dict[str, Any]) -> dict[str, Any]:
    seed = int(manifest["seed"])
    rng = np.random.default_rng(seed)
    alpha = float(manifest["alpha"])
    rho = float(manifest["rho"])
    cases = list(manifest["cases"])
    replications = int(manifest["replications"])
    family_alpha = alpha / len(cases)
    output_rows = []
    for sample_size in manifest["sample_sizes"]:
        simultaneous = np.ones(replications, dtype=bool)
        case_rows = []
        for case in cases:
            stats = _case_statistics(
                rng,
                _probabilities(case["nominal"], f"{case['family']} nominal"),
                _probabilities(case["surrogate"], f"{case['family']} surrogate"),
                rho=rho,
                alpha_family=family_alpha,
                sample_size=int(sample_size),
                replications=replications,
            )
            simultaneous &= stats.pop("coverage_flags")
            case_rows.append({"family": case["family"], **stats})
        output_rows.append(
            {
                "sample_size": int(sample_size),
                "family_results": case_rows,
                "simultaneous_empirical_coverage": float(np.mean(simultaneous)),
            }
        )
    minimum = float(manifest["minimum_empirical_coverage"])
    maximum_bias_allowed = float(manifest["maximum_absolute_bias"])
    marginal_coverages = [
        row["empirical_coverage"]
        for size_row in output_rows
        for row in size_row["family_results"]
    ]
    simultaneous_coverages = [row["simultaneous_empirical_coverage"] for row in output_rows]
    maximum_abs_bias = max(
        abs(row["bias"])
        for size_row in output_rows
        for row in size_row["family_results"]
    )
    coverage_gate = bool(
        min(marginal_coverages) >= minimum and min(simultaneous_coverages) >= minimum
    )
    bias_gate = bool(maximum_abs_bias <= maximum_bias_allowed)
    return {
        "rows": output_rows,
        "minimum_marginal_coverage": min(marginal_coverages),
        "minimum_simultaneous_coverage": min(simultaneous_coverages),
        "maximum_absolute_bias": float(maximum_abs_bias),
        "coverage_gate_passed": coverage_gate,
        "bias_gate_passed": bias_gate,
        "g0_gate_passed": bool(coverage_gate and bias_gate),
    }


def run(manifest_path: str | Path, repo_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest_raw = manifest_file.read_bytes()
    manifest = json.loads(manifest_raw.decode("utf-8"))
    validate_manifest(manifest, root)
    output = Path(output_path).resolve()
    expected = (root / manifest["output_path"]).resolve()
    if output != expected or root not in output.parents:
        raise SyntheticCoverageError("output differs from the manifest-bound path")
    if output.exists() or output.parent.exists():
        raise SyntheticCoverageError("synthetic G0 output path already exists; rerun refused")
    start = time.perf_counter()
    analysis = simulate_coverage(manifest)
    result = {
        "result_id": "p4-paired-risk-synthetic-g0-v1",
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "scientific_efficacy_experiment": False,
        "sealed_data_used": False,
        "formal_or_g2": False,
        "gpu_used": False,
        "old_gazebo_g1_touched": False,
        "independent_unit": "synthetic_replicated_experiment",
        "analysis": analysis,
        "elapsed_seconds": time.perf_counter() - start,
        "inference_boundary": "Known-probability synthetic G0 only; not controller efficacy, Gazebo evidence, formal certification, sealed evidence, or real-platform safety.",
    }
    body = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if result["elapsed_seconds"] > manifest["budgets"]["max_runtime_seconds"]:
        raise SyntheticCoverageError("runtime budget exceeded")
    if len(body) > manifest["budgets"]["max_output_bytes"]:
        raise SyntheticCoverageError("output budget exceeded")
    output.parent.mkdir(parents=True, exist_ok=False)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_bytes(body)
    temporary.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.manifest, args.repo_root, args.output)
    print(json.dumps({"elapsed_seconds": result["elapsed_seconds"], **result["analysis"]}, sort_keys=True))


if __name__ == "__main__":
    main()
