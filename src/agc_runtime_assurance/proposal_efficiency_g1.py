"""One-shot equal-call proposal-efficiency development experiment.

This runner compares proposal allocation only.  It applies the same attributed
fixed-time Hedged-CI to every arm, charges screening paths to learned proposals,
and never evaluates controller efficacy, Gazebo, sealed, formal or hardware data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np


def _isolated_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
CORE = _isolated_module("p4_proposal_efficiency_g1_core", HERE / "proposal_efficiency.py")
BETTING = _isolated_module("p4_proposal_efficiency_g1_betting", HERE / "paired_risk_betting.py")


METHODS = (
    "paired_nominal_mc",
    "baseline_failure_is",
    "union_failure_is",
    "positive_disagreement_is",
    "bidirectional_disagreement_is",
)
TARGET_BY_METHOD = {
    "baseline_failure_is": "baseline_failure",
    "union_failure_is": "union_failure",
    "positive_disagreement_is": "positive_disagreement",
    "bidirectional_disagreement_is": "bidirectional_disagreement",
}


class ProposalEfficiencyG1Error(RuntimeError):
    pass


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_bound_file(root: Path, binding: Any) -> None:
    if not isinstance(binding, dict):
        raise ProposalEfficiencyG1Error("manifest binding must be an object")
    relative = binding.get("path")
    expected = binding.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
        raise ProposalEfficiencyG1Error("manifest binding is incomplete")
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file() or _digest(path) != expected.lower():
        raise ProposalEfficiencyG1Error(f"binding mismatch: {binding.get('label', relative)}")


def validate_manifest(manifest: dict[str, Any], root: Path) -> None:
    required = {
        "stage": "route_b_proposal_efficiency_g1",
        "status": "frozen_awaiting_exact_launch_confirmation",
        "scientific_route_confirmed": True,
        "launch_authorized": False,
        "retry_allowed": False,
        "paper_efficacy_claim_allowed": False,
        "controller_efficacy_experiment": False,
        "sealed_data_authorized": False,
        "formal_experiment_authorized": False,
        "pilot_authorized": False,
        "g2_authorized": False,
        "hardware_authorized": False,
        "external_upload_authorized": False,
        "gpu_count": 0,
        "workers": 1,
        "certificate": "hedged_bounded_mean_ci",
        "methods": list(METHODS),
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ProposalEfficiencyG1Error(f"manifest {key} must equal {expected!r}")
    budgets = manifest.get("total_path_budgets")
    if budgets != sorted(set(budgets or [])) or len(budgets) < 2 or any(
        not isinstance(value, int) or value < 50 for value in budgets
    ):
        raise ProposalEfficiencyG1Error("total path budgets must be sorted unique integers")
    if manifest.get("primary_budget") != budgets[-1]:
        raise ProposalEfficiencyG1Error("primary budget must be the largest frozen budget")
    if manifest.get("replications") != 400:
        raise ProposalEfficiencyG1Error("replications must remain frozen at 400")
    if not math.isclose(float(manifest.get("screen_fraction", 0.0)), 0.2):
        raise ProposalEfficiencyG1Error("screen fraction must remain frozen at 0.2")
    if not math.isclose(float(manifest.get("rho", 0.0)), 0.5):
        raise ProposalEfficiencyG1Error("rho must remain frozen at 0.5")
    if manifest.get("bootstrap_resamples") != 10000:
        raise ProposalEfficiencyG1Error("bootstrap resamples must remain frozen at 10000")
    if manifest.get("positive_direction_guard_families") != [
        "separated_bidirectional",
        "negative_tail_guard",
    ]:
        raise ProposalEfficiencyG1Error("direction guard families changed")
    for binding in manifest.get("bindings", []):
        _validate_bound_file(root, binding)
    if len(manifest.get("bindings", [])) < 6:
        raise ProposalEfficiencyG1Error("manifest bindings are incomplete")
    output = manifest.get("output_path")
    if not isinstance(output, str) or output != "experiments/results/proposal_efficiency_g1_v1/result.json":
        raise ProposalEfficiencyG1Error("output path is not independently versioned")
    protected = manifest.get("protected_paths", [])
    required_protected = {
        "experiments/results/paired_risk_synthetic_g0_v1",
        "experiments/results/paired_risk_betting_g0_v1",
        "experiments/results/gazebo_second_system_g1_v1",
    }
    if not required_protected.issubset(set(protected)):
        raise ProposalEfficiencyG1Error("protected legacy result paths are incomplete")


def validate_launch_authorization(
    authorization: dict[str, Any], manifest_raw: bytes, manifest_path: Path
) -> None:
    if authorization.get("authorization_id") != "p4-proposal-efficiency-g1-v1-one-shot-launch":
        raise ProposalEfficiencyG1Error("launch authorization id mismatch")
    if authorization.get("authorized") is not True or authorization.get("retry_allowed") is not False:
        raise ProposalEfficiencyG1Error("one-shot launch authorization is absent or permits retry")
    if authorization.get("manifest_sha256") != hashlib.sha256(manifest_raw).hexdigest():
        raise ProposalEfficiencyG1Error("launch authorization does not bind the frozen manifest")
    if authorization.get("manifest_path") != manifest_path.as_posix():
        raise ProposalEfficiencyG1Error("launch authorization manifest path mismatch")
    if authorization.get("authorized_command") != (
        "C:/Users/liaoy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe "
        "src/agc_runtime_assurance/proposal_efficiency_g1.py "
        "--manifest experiments/manifests/proposal_efficiency_g1_v1.json "
        "--authorization experiments/manifests/proposal_efficiency_g1_v1_launch_authorization.json "
        "--repo-root . --output experiments/results/proposal_efficiency_g1_v1/result.json"
    ):
        raise ProposalEfficiencyG1Error("launch authorization command mismatch")


def _rng(master_seed: int, *tokens: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([master_seed, *tokens]))


def _ess_fraction(weights: np.ndarray) -> float:
    denominator = float(np.dot(weights, weights))
    if denominator <= 0.0:
        return 0.0
    return float(weights.sum()) ** 2 / (weights.size * denominator)


def _bootstrap_upper(
    paired_log_ratios: np.ndarray,
    *,
    confidence: float,
    resamples: int,
    rng: np.random.Generator,
) -> float:
    values = np.asarray(paired_log_ratios, dtype=float)
    if values.ndim != 1 or values.size < 20 or not np.all(np.isfinite(values)):
        raise ProposalEfficiencyG1Error("paired log ratios are invalid")
    means = np.empty(resamples, dtype=float)
    chunk = 500
    for start in range(0, resamples, chunk):
        stop = min(start + chunk, resamples)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = np.mean(values[indices], axis=1)
    return float(np.exp(np.quantile(means, confidence, method="higher")))


def _load_benchmark(root: Path, manifest: dict[str, Any]):
    path = root / manifest["benchmark_path"]
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("benchmark_id") != "p4-proposal-efficiency-known-probability-v1":
        raise ProposalEfficiencyG1Error("benchmark id mismatch")
    if raw.get("status") != "frozen_for_proposal_efficiency_g1":
        raise ProposalEfficiencyG1Error("benchmark is not frozen")
    if raw.get("category_order") != list(CORE.CATEGORY_ORDER):
        raise ProposalEfficiencyG1Error("benchmark category order mismatch")
    families = [CORE.family_from_mapping(item) for item in raw.get("families", [])]
    if [item.family for item in families] != manifest.get("registered_families"):
        raise ProposalEfficiencyG1Error("registered family order mismatch")
    return families


def simulate(manifest: dict[str, Any], families) -> dict[str, Any]:
    master_seed = int(manifest["seed"])
    replications = int(manifest["replications"])
    alpha = float(manifest["alpha"])
    family_alpha = alpha / len(families)
    rho = float(manifest["rho"])
    screen_fraction = float(manifest["screen_fraction"])
    prior = float(manifest["beta_prior_half_count"])
    score_floor = float(manifest["score_floor"])
    iterations = int(manifest["hedged_root_iterations"])
    records: list[dict[str, Any]] = []
    widths: dict[tuple[int, str, str], np.ndarray] = {}
    coverages: list[float] = []
    biases: list[float] = []
    simultaneous_coverages: list[float] = []
    minimum_ess_fraction = 1.0
    maximum_weight = 0.0

    for budget_index, budget in enumerate(manifest["total_path_budgets"]):
        per_cell: dict[tuple[str, str], list[dict[str, Any]]] = {
            (method, family.family): [] for method in METHODS for family in families
        }
        simultaneous_flags = {
            method: np.ones(replications, dtype=bool) for method in METHODS
        }
        simultaneous_positive = {
            method: np.ones(replications, dtype=bool) for method in METHODS
        }
        for replication in range(replications):
            for family_index, family in enumerate(families):
                screen_n, _, calls = CORE.equal_call_counts(
                    int(budget), screen_fraction, learned=True
                )
                screen_rng = _rng(master_seed, budget_index, replication, family_index, 1)
                screen_strata, screen_categories = CORE.sample_paths(
                    family,
                    family.nominal_stratum_probabilities,
                    screen_n,
                    screen_rng,
                )
                order_rng = _rng(master_seed, budget_index, replication, family_index, 2)
                method_order = list(np.asarray(METHODS)[order_rng.permutation(len(METHODS))])
                order_hash = hashlib.sha256("|".join(method_order).encode("utf-8")).hexdigest()
                for method_index, method in enumerate(method_order):
                    learned = method != "paired_nominal_mc"
                    charged_screen, evaluation_n, charged_calls = CORE.equal_call_counts(
                        int(budget), screen_fraction, learned=learned
                    )
                    if charged_calls != calls:
                        raise ProposalEfficiencyG1Error("equal-call accounting failed")
                    eval_rng = _rng(
                        master_seed,
                        budget_index,
                        replication,
                        family_index,
                        10 + METHODS.index(method),
                    )
                    if learned:
                        fitted = CORE.fit_defensive_proposal(
                            family,
                            screen_strata,
                            screen_categories,
                            target=TARGET_BY_METHOD[method],
                            rho=rho,
                            beta_prior_half_count=prior,
                            score_floor=score_floor,
                        )
                        q = fitted.defensive_stratum_probabilities
                        stratum_weights = fitted.importance_weights
                        interval_rho = rho
                        proposal_hash = fitted.fingerprint
                    else:
                        q = family.nominal_stratum_probabilities
                        stratum_weights = np.ones_like(q)
                        interval_rho = 1.0
                        proposal_hash = hashlib.sha256(q.astype("<f8").tobytes()).hexdigest()
                    strata, categories = CORE.sample_paths(family, q, evaluation_n, eval_rng)
                    weights = stratum_weights[strata]
                    differences = CORE.signed_differences(categories)
                    interval = BETTING.hedged_bounded_mean_interval(
                        weights,
                        differences,
                        rho=interval_rho,
                        family_alpha=family_alpha,
                        root_iterations=iterations,
                    )
                    estimate = float(np.mean(weights * differences))
                    width = float(interval.delta_upper - interval.delta_lower)
                    covered = bool(interval.delta_lower <= family.true_risk_reduction + 1e-15)
                    positive = bool(interval.delta_lower > 0.0)
                    ess_fraction = _ess_fraction(weights)
                    max_weight = float(np.max(weights))
                    minimum_ess_fraction = min(minimum_ess_fraction, ess_fraction)
                    maximum_weight = max(maximum_weight, max_weight)
                    simultaneous_flags[method][replication] &= covered
                    simultaneous_positive[method][replication] &= positive
                    per_cell[(method, family.family)].append(
                        {
                            "replication": replication,
                            "estimate": estimate,
                            "lower": float(interval.delta_lower),
                            "upper": float(interval.delta_upper),
                            "width": width,
                            "covered": covered,
                            "positive_lower": positive,
                            "ess_fraction": ess_fraction,
                            "maximum_weight": max_weight,
                            "screen_paths": charged_screen,
                            "evaluation_paths": evaluation_n,
                            "simulator_calls": charged_calls,
                            "proposal_sha256": proposal_hash,
                            "method_order_sha256": order_hash,
                        }
                    )
        summaries = []
        for method in METHODS:
            simultaneous_coverage = float(np.mean(simultaneous_flags[method]))
            simultaneous_certification = float(np.mean(simultaneous_positive[method]))
            simultaneous_coverages.append(simultaneous_coverage)
            for family in families:
                cell = per_cell[(method, family.family)]
                estimates = np.asarray([item["estimate"] for item in cell], dtype=float)
                cell_widths = np.asarray([item["width"] for item in cell], dtype=float)
                widths[(int(budget), method, family.family)] = cell_widths
                coverage = float(np.mean([item["covered"] for item in cell]))
                bias = float(np.mean(estimates) - family.true_risk_reduction)
                coverages.append(coverage)
                biases.append(abs(bias))
                summaries.append(
                    {
                        "method": method,
                        "family": family.family,
                        "true_risk_reduction": family.true_risk_reduction,
                        "mean_estimate": float(np.mean(estimates)),
                        "bias": bias,
                        "empirical_lower_coverage": coverage,
                        "mean_interval_width": float(np.mean(cell_widths)),
                        "median_interval_width": float(np.median(cell_widths)),
                        "mean_lower": float(np.mean([item["lower"] for item in cell])),
                        "mean_ess_fraction": float(np.mean([item["ess_fraction"] for item in cell])),
                        "maximum_weight": float(max(item["maximum_weight"] for item in cell)),
                    }
                )
        records.append(
            {
                "total_path_budget": int(budget),
                "standalone_simulator_calls_per_method_family_replication": 2 * int(budget),
                "cell_summaries": summaries,
                "simultaneous_lower_coverage": {
                    method: float(np.mean(simultaneous_flags[method])) for method in METHODS
                },
                "simultaneous_positive_certification_probability": {
                    method: float(np.mean(simultaneous_positive[method])) for method in METHODS
                },
                "replication_records": [
                    {"method": method, "family": family.family, **item}
                    for method in METHODS
                    for family in families
                    for item in per_cell[(method, family.family)]
                ],
            }
        )

    primary_budget = int(manifest["primary_budget"])
    proposed = "bidirectional_disagreement_is"
    comparisons = []
    comparison_alpha = float(manifest["efficiency_familywise_alpha"]) / 4.0
    bootstrap_confidence = 1.0 - comparison_alpha
    for comparison_index, comparator in enumerate(
        [
            "paired_nominal_mc",
            "baseline_failure_is",
            "union_failure_is",
            "positive_disagreement_is",
        ]
    ):
        if comparator == "positive_disagreement_is":
            comparison_families = manifest["positive_direction_guard_families"]
            ratio_threshold = float(manifest["positive_direction_width_ratio_threshold"])
        else:
            comparison_families = [family.family for family in families]
            ratio_threshold = float(manifest["primary_width_ratio_threshold"])
        paired_logs = np.zeros(replications, dtype=float)
        for family_name in comparison_families:
            numerator = widths[(primary_budget, proposed, family_name)]
            denominator = widths[(primary_budget, comparator, family_name)]
            if np.any(numerator <= 0.0) or np.any(denominator <= 0.0):
                raise ProposalEfficiencyG1Error("non-positive interval width")
            paired_logs += np.log(numerator / denominator)
        paired_logs /= len(comparison_families)
        geometric_ratio = float(np.exp(np.mean(paired_logs)))
        upper = _bootstrap_upper(
            paired_logs,
            confidence=bootstrap_confidence,
            resamples=int(manifest["bootstrap_resamples"]),
            rng=_rng(master_seed, 900, comparison_index),
        )
        comparisons.append(
            {
                "comparator": comparator,
                "families": comparison_families,
                "geometric_mean_width_ratio": geometric_ratio,
                "bonferroni_bootstrap_upper_ratio": upper,
                "ratio_threshold": ratio_threshold,
                "multiplicity_adjusted_confidence": bootstrap_confidence,
                "passed": bool(geometric_ratio <= ratio_threshold and upper < 1.0),
            }
        )

    coverage_gate = bool(
        min(coverages) >= float(manifest["minimum_marginal_coverage"])
        and min(simultaneous_coverages) >= float(manifest["minimum_simultaneous_coverage"])
    )
    bias_gate = bool(max(biases) <= float(manifest["maximum_absolute_bias"]))
    diagnostic_gate = bool(
        minimum_ess_fraction >= float(manifest["minimum_ess_fraction"])
        and maximum_weight <= 1.0 / rho + 1e-12
    )
    efficiency_gate = bool(all(item["passed"] for item in comparisons))
    return {
        "budget_results": records,
        "efficiency_comparisons": comparisons,
        "minimum_marginal_lower_coverage": min(coverages),
        "minimum_simultaneous_lower_coverage": min(simultaneous_coverages),
        "maximum_absolute_bias": max(biases),
        "minimum_ess_fraction": minimum_ess_fraction,
        "maximum_weight": maximum_weight,
        "coverage_gate_passed": coverage_gate,
        "bias_gate_passed": bias_gate,
        "diagnostic_gate_passed": diagnostic_gate,
        "efficiency_gate_passed": efficiency_gate,
        "proposal_efficiency_g1_passed": bool(
            coverage_gate and bias_gate and diagnostic_gate and efficiency_gate
        ),
    }


def run(
    manifest_path: str | Path,
    authorization_path: str | Path,
    repo_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest_raw = manifest_file.read_bytes()
    manifest = json.loads(manifest_raw.decode("utf-8"))
    validate_manifest(manifest, root)
    authorization_file = Path(authorization_path).resolve()
    authorization = json.loads(authorization_file.read_text(encoding="utf-8"))
    validate_launch_authorization(
        authorization,
        manifest_raw,
        manifest_file.relative_to(root),
    )
    output = Path(output_path).resolve()
    expected = (root / manifest["output_path"]).resolve()
    if output != expected or root not in output.parents:
        raise ProposalEfficiencyG1Error("output differs from the manifest-bound path")
    if output.exists() or output.parent.exists():
        raise ProposalEfficiencyG1Error("proposal-efficiency output path already exists; rerun refused")
    families = _load_benchmark(root, manifest)
    start = time.perf_counter()
    analysis = simulate(manifest, families)
    result = {
        "result_id": "p4-proposal-efficiency-g1-v1",
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "benchmark_sha256": _digest(root / manifest["benchmark_path"]),
        "scientific_route_confirmed": True,
        "paper_efficacy_claim_allowed": False,
        "controller_efficacy_experiment": False,
        "development_proposal_efficiency_evidence": True,
        "sealed_data_used": False,
        "formal_or_g2": False,
        "gpu_used": False,
        "workers": 1,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "analysis": analysis,
        "elapsed_seconds": time.perf_counter() - start,
        "inference_boundary": "Known-probability equal-call proposal-efficiency development evidence only; no controller efficacy, Gazebo, 1U1G, sealed, formal, G2, pilot, hardware, or real-platform claim.",
    }
    body = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if result["elapsed_seconds"] > manifest["budgets"]["max_runtime_seconds"]:
        raise ProposalEfficiencyG1Error("runtime budget exceeded")
    if len(body) > manifest["budgets"]["max_output_bytes"]:
        raise ProposalEfficiencyG1Error("output budget exceeded")
    output.parent.mkdir(parents=True, exist_ok=False)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_bytes(body)
    temporary.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.manifest, args.authorization, args.repo_root, args.output)
    summary = {
        "elapsed_seconds": result["elapsed_seconds"],
        **{key: value for key, value in result["analysis"].items() if key.endswith("passed")},
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
