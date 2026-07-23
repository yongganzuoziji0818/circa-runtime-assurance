"""Exactly-once Route-B G0 for paired-risk betting mappings.

The run is known-probability method validation only.  Width and lower-bound
summaries are descriptive and cannot support an efficacy or novelty claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


CORE_PATH = Path(__file__).with_name("paired_risk_betting.py")
SPEC = importlib.util.spec_from_file_location("p4_paired_risk_betting_g0_core", CORE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load paired risk betting core")
CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)


class RouteBG0Error(RuntimeError):
    pass


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probabilities(values: Any, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (4,) or not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise RouteBG0Error(f"{label} must contain four positive finite probabilities")
    if not math.isclose(float(np.sum(array)), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RouteBG0Error(f"{label} probabilities must sum to one")
    return array


def _validate_bound_file(root: Path, relative: Any, expected: Any, label: str) -> None:
    if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
        raise RouteBG0Error(f"{label} binding is incomplete")
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file() or _digest(path) != expected.lower():
        raise RouteBG0Error(f"{label} hash mismatch")


def validate_manifest(manifest: dict[str, Any], root: Path) -> None:
    required = {
        "stage": "route_b_known_probability_g0",
        "status": "frozen_authorized_one_shot",
        "authorized": True,
        "claim_generation_allowed": False,
        "scientific_efficacy_experiment": False,
        "sealed_data_authorized": False,
        "formal_experiment_authorized": False,
        "pilot_authorized": False,
        "g2_authorized": False,
        "hardware_authorized": False,
        "external_upload_authorized": False,
        "old_gazebo_g1_touched": False,
        "consumed_synthetic_g0_rerun_allowed": False,
        "gpu_count": 0,
        "workers": 1,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise RouteBG0Error(f"manifest {key} must equal {expected!r}")
    if manifest.get("procedures") != [
        "empirical_bernstein_reference",
        "mope_off_policy_cs",
        "hedged_bounded_mean_ci",
    ]:
        raise RouteBG0Error("procedure order differs from the freeze")
    alpha = manifest.get("alpha")
    rho = manifest.get("rho")
    if not isinstance(alpha, (int, float)) or not 0.0 < alpha < 1.0:
        raise RouteBG0Error("alpha must lie in (0, 1)")
    if not isinstance(rho, (int, float)) or not 0.0 < rho < 1.0:
        raise RouteBG0Error("rho must lie strictly between zero and one")
    if not isinstance(manifest.get("replications"), int) or manifest["replications"] < 100:
        raise RouteBG0Error("replications must be at least 100")
    sample_sizes = manifest.get("sample_sizes")
    if not isinstance(sample_sizes, list) or not sample_sizes or any(
        not isinstance(value, int) or value < 2 for value in sample_sizes
    ):
        raise RouteBG0Error("sample_sizes are invalid")
    if manifest.get("mope_grid_size") != 501 or manifest.get("hedged_root_iterations") != 48:
        raise RouteBG0Error("numerical settings differ from the freeze")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 4:
        raise RouteBG0Error("exactly four synthetic families are required")
    names = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("family"), str):
            raise RouteBG0Error("each case requires a family name")
        names.append(case["family"])
        _probabilities(case.get("nominal"), f"{case['family']} nominal")
        _probabilities(case.get("surrogate"), f"{case['family']} surrogate")
    if len(set(names)) != 4:
        raise RouteBG0Error("synthetic family names must be unique")
    bindings = manifest.get("bindings")
    if not isinstance(bindings, list) or len(bindings) < 5:
        raise RouteBG0Error("manifest bindings are incomplete")
    for binding in bindings:
        _validate_bound_file(root, binding.get("path"), binding.get("sha256"), binding.get("label", "file"))
    budgets = manifest.get("budgets")
    if not isinstance(budgets, dict):
        raise RouteBG0Error("budgets are missing")
    if budgets.get("max_runtime_seconds") != 180 or budgets.get("max_output_bytes") != 2097152:
        raise RouteBG0Error("Route-B budgets differ from the freeze")


def _eb_interval(
    weighted: np.ndarray,
    *,
    rho: float,
    family_alpha: float,
) -> tuple[float, float]:
    n = int(weighted.size)
    estimate = float(np.mean(weighted))
    variance = float(np.var(weighted, ddof=1))
    log_term = math.log(2.0 / family_alpha)
    radius = math.sqrt(2.0 * variance * log_term / n) + 14.0 * log_term / (
        3.0 * rho * (n - 1)
    )
    return max(-1.0, estimate - radius), min(1.0, estimate + radius)


def _case_result(
    rng: np.random.Generator,
    case: dict[str, Any],
    *,
    rho: float,
    family_alpha: float,
    sample_size: int,
    replications: int,
    grid_size: int,
    root_iterations: int,
) -> dict[str, Any]:
    nominal = _probabilities(case["nominal"], f"{case['family']} nominal")
    surrogate = _probabilities(case["surrogate"], f"{case['family']} surrogate")
    proposal = rho * nominal + (1.0 - rho) * surrogate
    category_weights = nominal / proposal
    if np.any(category_weights > 1.0 / rho + 1e-12):
        raise RouteBG0Error("defensive-mixture weight bound failed")
    category_differences = np.array([0.0, 1.0, -1.0, 0.0])
    categories = rng.choice(4, size=(replications, sample_size), p=proposal)
    weights = category_weights[categories]
    differences = category_differences[categories]
    estimates = np.mean(weights * differences, axis=1)
    truth = float(nominal[1] - nominal[2])
    method_rows: dict[str, dict[str, Any]] = {}
    for procedure in (
        "empirical_bernstein_reference",
        "mope_off_policy_cs",
        "hedged_bounded_mean_ci",
    ):
        lower = np.empty(replications, dtype=float)
        upper = np.empty(replications, dtype=float)
        for index in range(replications):
            if procedure == "empirical_bernstein_reference":
                lower[index], upper[index] = _eb_interval(
                    weights[index] * differences[index],
                    rho=rho,
                    family_alpha=family_alpha,
                )
            elif procedure == "mope_off_policy_cs":
                interval = CORE.mope_paired_interval(
                    weights[index],
                    differences[index],
                    rho=rho,
                    family_alpha=family_alpha,
                    grid_size=grid_size,
                )
                lower[index], upper[index] = interval.delta_lower, interval.delta_upper
            else:
                interval = CORE.hedged_bounded_mean_interval(
                    weights[index],
                    differences[index],
                    rho=rho,
                    family_alpha=family_alpha,
                    root_iterations=root_iterations,
                )
                lower[index], upper[index] = interval.delta_lower, interval.delta_upper
        if np.any(lower > upper + 1e-12):
            raise RouteBG0Error(f"{procedure} produced an inverted interval")
        coverage = lower <= truth + 1e-12
        method_rows[procedure] = {
            "empirical_lower_coverage": float(np.mean(coverage)),
            "mean_lower_bound": float(np.mean(lower)),
            "mean_upper_bound": float(np.mean(upper)),
            "mean_interval_width": float(np.mean(upper - lower)),
            "median_interval_width": float(np.median(upper - lower)),
            "coverage_flags": coverage,
        }
    return {
        "family": case["family"],
        "true_risk_reduction": truth,
        "mean_estimate": float(np.mean(estimates)),
        "bias": float(np.mean(estimates) - truth),
        "maximum_weight": float(np.max(category_weights)),
        "nominal_probabilities": nominal.tolist(),
        "surrogate_probabilities": surrogate.tolist(),
        "proposal_probabilities": proposal.tolist(),
        "method_results": method_rows,
    }


def _refusal_checks(rho: float) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for name, function in (
        ("mope_weight_bound", CORE.mope_paired_interval),
        ("hedged_weight_bound", CORE.hedged_bounded_mean_interval),
    ):
        try:
            function([1.0 / rho + 0.1, 1.0], [1.0, 0.0], rho=rho, family_alpha=0.05)
        except CORE.PairedRiskBettingError:
            checks[name] = True
        else:
            checks[name] = False
    try:
        CORE.mope_paired_interval([1.0, 1.0], [2.0, 0.0], rho=rho, family_alpha=0.05)
    except CORE.PairedRiskBettingError:
        checks["invalid_difference"] = True
    else:
        checks["invalid_difference"] = False
    return checks


def simulate(manifest: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(int(manifest["seed"]))
    cases = list(manifest["cases"])
    alpha = float(manifest["alpha"])
    rho = float(manifest["rho"])
    replications = int(manifest["replications"])
    family_alpha = alpha / len(cases)
    rows = []
    all_marginal_coverages: list[float] = []
    all_simultaneous_coverages: list[float] = []
    all_biases: list[float] = []
    for sample_size in manifest["sample_sizes"]:
        simultaneous = {
            procedure: np.ones(replications, dtype=bool)
            for procedure in manifest["procedures"]
        }
        family_rows = []
        for case in cases:
            result = _case_result(
                rng,
                case,
                rho=rho,
                family_alpha=family_alpha,
                sample_size=int(sample_size),
                replications=replications,
                grid_size=int(manifest["mope_grid_size"]),
                root_iterations=int(manifest["hedged_root_iterations"]),
            )
            all_biases.append(abs(result["bias"]))
            for procedure, stats in result["method_results"].items():
                flags = stats.pop("coverage_flags")
                simultaneous[procedure] &= flags
                all_marginal_coverages.append(stats["empirical_lower_coverage"])
            family_rows.append(result)
        simultaneous_rows = {
            procedure: float(np.mean(flags)) for procedure, flags in simultaneous.items()
        }
        all_simultaneous_coverages.extend(simultaneous_rows.values())
        rows.append(
            {
                "sample_size": int(sample_size),
                "family_results": family_rows,
                "simultaneous_lower_coverage": simultaneous_rows,
            }
        )
    refusals = _refusal_checks(rho)
    minimum_marginal = min(all_marginal_coverages)
    minimum_simultaneous = min(all_simultaneous_coverages)
    maximum_bias = max(all_biases)
    coverage_gate = bool(
        minimum_marginal >= manifest["minimum_marginal_coverage"]
        and minimum_simultaneous >= manifest["minimum_simultaneous_coverage"]
    )
    bias_gate = bool(maximum_bias <= manifest["maximum_absolute_bias"])
    refusal_gate = bool(all(refusals.values()))
    return {
        "rows": rows,
        "minimum_marginal_lower_coverage": minimum_marginal,
        "minimum_simultaneous_lower_coverage": minimum_simultaneous,
        "maximum_absolute_bias": maximum_bias,
        "refusal_checks": refusals,
        "coverage_gate_passed": coverage_gate,
        "bias_gate_passed": bias_gate,
        "refusal_gate_passed": refusal_gate,
        "g0_gate_passed": bool(coverage_gate and bias_gate and refusal_gate),
        "tightness_is_descriptive_only": True,
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
        raise RouteBG0Error("output differs from the manifest-bound path")
    if output.exists() or output.parent.exists():
        raise RouteBG0Error("Route-B G0 output path already exists; rerun refused")
    start = time.perf_counter()
    analysis = simulate(manifest)
    result = {
        "result_id": "p4-paired-risk-betting-g0-v1",
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "claim_generation_allowed": False,
        "scientific_efficacy_experiment": False,
        "sealed_data_used": False,
        "formal_or_g2": False,
        "gpu_used": False,
        "workers": 1,
        "old_gazebo_g1_touched": False,
        "consumed_synthetic_g0_touched": False,
        "analysis": analysis,
        "elapsed_seconds": time.perf_counter() - start,
        "inference_boundary": "Route-B known-probability G0 only; no proposal efficiency, controller efficacy, Gazebo, sealed, formal, G2, pilot, hardware, or paper-performance evidence.",
    }
    body = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if result["elapsed_seconds"] > manifest["budgets"]["max_runtime_seconds"]:
        raise RouteBG0Error("runtime budget exceeded")
    if len(body) > manifest["budgets"]["max_output_bytes"]:
        raise RouteBG0Error("output budget exceeded")
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

