"""Versioned engineering repair for proposal-efficiency G1.

V1 failed before producing scientific output because the audited betting helper
excluded the mathematically valid nominal-MC boundary rho=1.  This wrapper keeps
the V1 scientific design unchanged and adds exact rho=1 handling only.
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
from typing import Any, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "p4_proposal_efficiency_g1_v1_preserved", HERE / "proposal_efficiency_g1.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load preserved proposal-efficiency G1 v1 runner")
BASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)

ORIGINAL_HEDGED = BASE.BETTING.hedged_bounded_mean_interval
ORIGINAL_VALIDATE_MANIFEST = BASE.validate_manifest


def hedged_bounded_mean_interval_allow_nominal(
    weights: Sequence[float] | np.ndarray,
    differences: Sequence[float] | np.ndarray,
    *,
    rho: float,
    family_alpha: float,
    root_iterations: int = 48,
):
    if rho < 1.0:
        return ORIGINAL_HEDGED(
            weights,
            differences,
            rho=rho,
            family_alpha=family_alpha,
            root_iterations=root_iterations,
        )
    if rho != 1.0 or not math.isfinite(rho):
        raise BASE.BETTING.PairedRiskBettingError("rho must lie in (0, 1]")
    if not 0.0 < family_alpha < 1.0 or not math.isfinite(family_alpha):
        raise BASE.BETTING.PairedRiskBettingError("family_alpha must lie strictly between zero and one")
    if not isinstance(root_iterations, int) or root_iterations < 24:
        raise BASE.BETTING.PairedRiskBettingError("root_iterations must be an integer of at least 24")
    w = np.asarray(weights, dtype=float)
    d = np.asarray(differences, dtype=float)
    if w.ndim != 1 or d.ndim != 1 or w.shape != d.shape or w.size < 2:
        raise BASE.BETTING.PairedRiskBettingError("weights and differences must be paired non-empty vectors")
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(d)):
        raise BASE.BETTING.PairedRiskBettingError("weights and differences contain non-finite values")
    if np.any(w < 0.0) or np.any(w > 1.0 + 1e-12):
        raise BASE.BETTING.PairedRiskBettingError("weights exceed the nominal rho=1 bound")
    if np.any(d < -1.0) or np.any(d > 1.0):
        raise BASE.BETTING.PairedRiskBettingError("differences must lie in [-1, 1]")

    bounded = (w * d + 1.0) / 2.0
    theta = 0.5
    positive_bets = BASE.BETTING._fixed_time_predmix_bets(
        bounded, family_alpha * theta
    )
    negative_bets = BASE.BETTING._fixed_time_predmix_bets(
        bounded, family_alpha * (1.0 - theta)
    )
    threshold = math.log(1.0 / family_alpha)

    def accepted(candidate: float) -> bool:
        return BASE.BETTING._hedged_log_capital(
            bounded, candidate, positive_bets, negative_bets, theta
        ) <= threshold

    center = float(np.mean(bounded))
    if not accepted(center):
        search = np.linspace(0.0, 1.0, 1001)
        accepted_values = [float(value) for value in search if accepted(float(value))]
        if not accepted_values:
            raise BASE.BETTING.PairedRiskBettingError("hedged confidence set is numerically empty")
        center = accepted_values[len(accepted_values) // 2]
    lower_mean = BASE.BETTING._conservative_root(
        accepted, center, lower=True, iterations=root_iterations
    )
    upper_mean = BASE.BETTING._conservative_root(
        accepted, center, lower=False, iterations=root_iterations
    )
    return BASE.BETTING.PairedBettingInterval(
        procedure="hedged_bounded_mean_ci",
        sample_size=int(w.size),
        delta_lower=max(-1.0, 2.0 * lower_mean - 1.0),
        delta_upper=min(1.0, 2.0 * upper_mean - 1.0),
        transformed_lower=lower_mean,
        transformed_upper=upper_mean,
        family_alpha=float(family_alpha),
        rho=1.0,
        numerical_resolution=2.0 * 2.0 ** (-root_iterations),
    )


BASE.BETTING.hedged_bounded_mean_interval = hedged_bounded_mean_interval_allow_nominal


def validate_manifest_v2(manifest: dict[str, Any], root: Path) -> None:
    if manifest.get("manifest_id") != "p4-proposal-efficiency-g1-v2-2026-07-19":
        raise BASE.ProposalEfficiencyG1Error("v2 manifest id mismatch")
    if manifest.get("engineering_revision") != "exact_nominal_rho_one_boundary_only":
        raise BASE.ProposalEfficiencyG1Error("v2 engineering revision expanded")
    if manifest.get("supersedes_failed_manifest_sha256") != (
        "60bbe401ebb1353acc77b5e89258dd347c72211914d5679f0d7a0713f4247e9f"
    ):
        raise BASE.ProposalEfficiencyG1Error("v2 does not bind the failed v1 manifest")
    if manifest.get("output_path") != "experiments/results/proposal_efficiency_g1_v2/result.json":
        raise BASE.ProposalEfficiencyG1Error("v2 output path mismatch")
    if manifest.get("launch_authorization_path") != (
        "experiments/manifests/proposal_efficiency_g1_v2_launch_authorization.json"
    ):
        raise BASE.ProposalEfficiencyG1Error("v2 launch path mismatch")
    transformed = dict(manifest)
    transformed["output_path"] = "experiments/results/proposal_efficiency_g1_v1/result.json"
    ORIGINAL_VALIDATE_MANIFEST(transformed, root)
    if "experiments/results/proposal_efficiency_g1_v1" not in manifest["protected_paths"]:
        raise BASE.ProposalEfficiencyG1Error("failed v1 output namespace is not protected")


def validate_launch_authorization_v2(
    authorization: dict[str, Any], manifest_raw: bytes, manifest_path: Path
) -> None:
    expected_command = (
        "C:/Users/liaoy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe "
        "src/agc_runtime_assurance/proposal_efficiency_g1_v2.py "
        "--manifest experiments/manifests/proposal_efficiency_g1_v2.json "
        "--authorization experiments/manifests/proposal_efficiency_g1_v2_launch_authorization.json "
        "--repo-root . --output experiments/results/proposal_efficiency_g1_v2/result.json"
    )
    if authorization.get("authorization_id") != "p4-proposal-efficiency-g1-v2-one-shot-reverification":
        raise BASE.ProposalEfficiencyG1Error("v2 authorization id mismatch")
    if authorization.get("authorized") is not True or authorization.get("retry_allowed") is not False:
        raise BASE.ProposalEfficiencyG1Error("v2 one-shot authorization is invalid")
    if authorization.get("manifest_sha256") != hashlib.sha256(manifest_raw).hexdigest():
        raise BASE.ProposalEfficiencyG1Error("v2 authorization does not bind the manifest")
    if authorization.get("manifest_path") != manifest_path.as_posix():
        raise BASE.ProposalEfficiencyG1Error("v2 authorization manifest path mismatch")
    if authorization.get("authorized_command") != expected_command:
        raise BASE.ProposalEfficiencyG1Error("v2 authorization command mismatch")


BASE.validate_manifest = validate_manifest_v2
BASE.validate_launch_authorization = validate_launch_authorization_v2


def run_v2(
    manifest_path: str | Path,
    authorization_path: str | Path,
    repo_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest_raw = manifest_file.read_bytes()
    manifest = json.loads(manifest_raw.decode("utf-8"))
    validate_manifest_v2(manifest, root)
    authorization = json.loads(Path(authorization_path).read_text(encoding="utf-8"))
    validate_launch_authorization_v2(
        authorization, manifest_raw, manifest_file.relative_to(root)
    )
    output = Path(output_path).resolve()
    expected = (root / manifest["output_path"]).resolve()
    if output != expected or root not in output.parents:
        raise BASE.ProposalEfficiencyG1Error("v2 output differs from the manifest-bound path")
    if output.exists() or output.parent.exists():
        raise BASE.ProposalEfficiencyG1Error("v2 output path already exists; re-execution refused")
    families = BASE._load_benchmark(root, manifest)
    start = time.perf_counter()
    analysis = BASE.simulate(manifest, families)
    result = {
        "result_id": "p4-proposal-efficiency-g1-v2",
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "benchmark_sha256": BASE._digest(root / manifest["benchmark_path"]),
        "engineering_revision": "exact_nominal_rho_one_boundary_only",
        "failed_v1_manifest_sha256": manifest["supersedes_failed_manifest_sha256"],
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
        raise BASE.ProposalEfficiencyG1Error("v2 runtime budget exceeded")
    if len(body) > manifest["budgets"]["max_output_bytes"]:
        raise BASE.ProposalEfficiencyG1Error("v2 output budget exceeded")
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
    result = run_v2(args.manifest, args.authorization, args.repo_root, args.output)
    summary = {
        "elapsed_seconds": result["elapsed_seconds"],
        **{key: value for key, value in result["analysis"].items() if key.endswith("passed")},
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
