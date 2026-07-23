"""Fail-closed G1a mechanism-validation runner for the unsealed P4 sandbox.

G1a records deterministic falsification and contract checks.  It is not a
performance comparison and cannot authorize development, formal, or sealed work.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np
import scipy

from .counterexamples import (
    coupling_before_individual_failure,
    marginal_coverage_selection_failure,
    stale_observation_deadline_failure,
)
from .filtering import FilterStatus
from .preflight import PreflightError, verify_development_manifest
from .sandbox_baselines import (
    SandboxACoFiAdapter,
    SandboxConformalCBFAdapter,
    SandboxNominalCBFAdapter,
)
from .sandbox_fallback import (
    SandboxFallbackSafeMPCAdapter,
    sandbox_backup_equilibrium,
    sandbox_backup_invariant_radius,
)
from .sandbox_task import SandboxComparisonTask
from .scenario_manifest import load_g0_scenario_manifest


class G1ARunnerError(RuntimeError):
    pass


_REQUIRED_CHECKS = {
    "coupling_precedes_individual_failure",
    "selected_action_coverage_failure",
    "aoi_debit_prevents_overrun",
    "nominal_cbf_shared_contract",
    "conformal_cbf_shared_contract",
    "acofi_unsafe_task_policy_visible",
    "acofi_feedback_switches_safe",
    "fallback_invariant_binding",
    "fallback_unrecoverable_fail_closed",
}


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_code_tree_hash(repo_root: str | Path) -> str:
    root = Path(repo_root).resolve()
    files = sorted(
        (
            path for folder in (root / "src", root / "tests")
            for path in folder.rglob("*")
            if path.is_file()
            and path.suffix != ".pyc"
            and "__pycache__" not in path.parts
            and ".pytest_cache" not in path.parts
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    lines = [
        f"{path.relative_to(root).as_posix()}\t{sha256_file(path)}"
        for path in files
    ]
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise G1ARunnerError(f"JSON root must be an object: {path}")
    return value


def _repo_artifact(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise G1ARunnerError(f"{label} path is missing")
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise G1ARunnerError(f"{label} escapes repo root")
    if not candidate.is_file():
        raise G1ARunnerError(f"{label} does not exist: {relative}")
    return candidate


def _verify_bound_artifact(
    manifest: dict[str, Any], root: Path, prefix: str,
) -> Path:
    path = _repo_artifact(root, manifest.get(f"{prefix}_path"), prefix)
    expected = manifest.get(f"{prefix}_hash")
    if not isinstance(expected, str) or sha256_file(path) != expected.lower():
        raise G1ARunnerError(f"{prefix} hash mismatch")
    return path


def _validate_authorization(
    authorization_path: Path, repo_root: Path,
) -> tuple[dict[str, Any], Path, Path]:
    authorization = verify_development_manifest(authorization_path)
    if authorization.experiment_family != "synthetic_counterexample":
        raise G1ARunnerError("G1a requires the synthetic_counterexample family")
    manifest = _load_json(authorization_path)
    if canonical_code_tree_hash(repo_root) != authorization.code_hash.lower():
        raise G1ARunnerError("current src+tests code tree does not match authorization")
    split_paths = manifest.get("split_artifact_paths")
    if not isinstance(split_paths, dict) or set(split_paths) != {
        "calibration", "development",
    }:
        raise G1ARunnerError("split_artifact_paths must bind calibration and development")
    for split, expected in (
        ("calibration", authorization.calibration_hash),
        ("development", authorization.development_hash),
    ):
        path = _repo_artifact(repo_root, split_paths[split], f"{split} split")
        if sha256_file(path) != expected.lower():
            raise G1ARunnerError(f"{split} split hash mismatch")
        split_manifest = _load_json(path)
        if split_manifest.get("split") != split or split_manifest.get("sealed") is not False:
            raise G1ARunnerError(f"{split} split boundary is invalid")
    protocol_path = _verify_bound_artifact(manifest, repo_root, "protocol")
    scenario_path = _verify_bound_artifact(manifest, repo_root, "scenario_manifest")
    _verify_bound_artifact(manifest, repo_root, "autonomy_manifest")
    _verify_bound_artifact(manifest, repo_root, "resource_snapshot")
    protocol = _load_json(protocol_path)
    if protocol.get("experiment_family") != "synthetic_counterexample":
        raise G1ARunnerError("protocol experiment family mismatch")
    if protocol.get("sealed_data_allowed") is not False:
        raise G1ARunnerError("protocol must forbid sealed data")
    if protocol.get("formal_experiment") is not False:
        raise G1ARunnerError("protocol must not be formal")
    if protocol.get("claim_generation_allowed") is not False:
        raise G1ARunnerError("protocol must forbid claim generation")
    if protocol.get("run_count") != 1:
        raise G1ARunnerError("G1a protocol must authorize exactly one run")
    required = protocol.get("required_checks")
    if not isinstance(required, dict) or set(required) != _REQUIRED_CHECKS:
        raise G1ARunnerError("protocol required-check set is not frozen G1a v1")
    return manifest, protocol_path, scenario_path


def evaluate_g1a(
    authorization_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    authorization_path = Path(authorization_path).resolve()
    manifest, protocol_path, scenario_path = _validate_authorization(
        authorization_path, root,
    )
    scenario_manifest = load_g0_scenario_manifest(scenario_path)
    expected_scenario_hash = manifest["scenario_manifest_hash"].lower()
    if scenario_manifest.fingerprint != expected_scenario_hash:
        raise G1ARunnerError("scenario loader fingerprint mismatch")

    checks = {name: False for name in sorted(_REQUIRED_CHECKS)}
    observations: dict[str, Any] = {}
    errors: list[dict[str, str]] = []

    try:
        example = coupling_before_individual_failure()
        checks["coupling_precedes_individual_failure"] = bool(
            example.team_first_violation_time < example.individual_censored_horizon
        )
        observations["coupling"] = {
            "team_first_violation_time": example.team_first_violation_time,
            "individual_censored_horizon": example.individual_censored_horizon,
        }
    except Exception as error:  # evidence must retain unexpected failures
        errors.append({"check": "coupling", "type": type(error).__name__, "message": str(error)})

    try:
        example = marginal_coverage_selection_failure()
        checks["selected_action_coverage_failure"] = bool(
            example.marginal_coverage >= 0.9
            and example.selected_hard_coverage == 0.0
        )
        observations["selection"] = {
            "marginal_coverage": example.marginal_coverage,
            "selected_hard_coverage": example.selected_hard_coverage,
            "selected_fraction": example.selected_fraction,
            "optimism_correction": example.optimism_correction,
        }
    except Exception as error:
        errors.append({"check": "selection", "type": type(error).__name__, "message": str(error)})

    try:
        example = stale_observation_deadline_failure()
        checks["aoi_debit_prevents_overrun"] = bool(
            example.duration_without_age_debit > example.true_remaining_safe_time
            and example.duration_with_all_debits <= example.true_remaining_safe_time
        )
        observations["staleness"] = {
            "true_remaining_safe_time": example.true_remaining_safe_time,
            "duration_without_age_debit": example.duration_without_age_debit,
            "duration_with_all_debits": example.duration_with_all_debits,
        }
    except Exception as error:
        errors.append({"check": "staleness", "type": type(error).__name__, "message": str(error)})

    try:
        scenario = next(
            item for item in scenario_manifest.scenarios if item.name == "near_collision"
        )
        _, state = scenario.instantiate()
        task = SandboxComparisonTask(shift=scenario.shift)
        fallback = np.array([-2.0, 0.0, 0.0, 2.0, 0.0])
        nominal = SandboxNominalCBFAdapter(task).decide(
            state, fallback_action=fallback,
        )
        checks["nominal_cbf_shared_contract"] = bool(
            nominal.filter_result.status == FilterStatus.FILTERED
            and np.all(
                nominal.constraint_bundle.A @ nominal.action
                <= nominal.constraint_bundle.b + 1e-7
            )
            and task.postcheck_next_state(state, nominal.action)
        )
        conformal = SandboxConformalCBFAdapter(
            target_loss=-0.05,
            learning_rate=0.1,
            initial_value=-0.01,
            task=task,
        ).decide(state, fallback_action=fallback)
        checks["conformal_cbf_shared_contract"] = bool(
            conformal.constraint_contract_fingerprint
            == nominal.constraint_contract_fingerprint
            and np.all(
                conformal.conformal_step.affine_A @ conformal.action
                <= conformal.conformal_step.affine_b + 1e-7
            )
            and task.postcheck_next_state(state, conformal.action)
        )
        acofi = SandboxACoFiAdapter(
            target_alpha=0.1,
            learning_rate=0.05,
            gamma=0.9,
            safety_threshold=0.1,
            task=task,
        )
        unsafe = acofi.decide(
            state,
            step_index=0,
            predicted_task_q=1.0,
            fallback_action=fallback,
        )
        checks["acofi_unsafe_task_policy_visible"] = bool(
            unsafe.acofi_decision.source == "task_policy"
            and unsafe.exact_next_step_postcheck is False
        )
        _, feedback_error = acofi.observe_transition(
            previous_predicted_q=1.0,
            previous_local_margin=0.01,
            next_learned_value=-1.0,
        )
        safe = acofi.decide(
            state,
            step_index=0,
            predicted_task_q=100.0,
            fallback_action=fallback,
        )
        checks["acofi_feedback_switches_safe"] = bool(
            feedback_error
            and safe.acofi_decision.source == "learned_safe_policy"
            and safe.exact_next_step_postcheck is True
        )
        observations["shared_task"] = {
            "scenario_manifest_hash": scenario_manifest.fingerprint,
            "nominal_policy_fingerprint": task.nominal_policy_fingerprint,
            "constraint_contract_fingerprint": task.constraint_contract_fingerprint,
            "nominal_filter_status": nominal.filter_result.status.value,
            "nominal_intervention_norm": nominal.filter_result.intervention_norm,
            "conformal_intervention_norm": conformal.conformal_step.filter_result.intervention_norm,
            "acofi_initial_source": unsafe.acofi_decision.source,
            "acofi_initial_exact_postcheck": unsafe.exact_next_step_postcheck,
            "acofi_after_feedback_source": safe.acofi_decision.source,
            "acofi_after_feedback_exact_postcheck": safe.exact_next_step_postcheck,
        }
    except Exception as error:
        errors.append({"check": "shared_task", "type": type(error).__name__, "message": str(error)})

    try:
        fallback = SandboxFallbackSafeMPCAdapter()
        inside = sandbox_backup_equilibrium() + 0.1 * sandbox_backup_invariant_radius()
        inside_decision = fallback.decide(inside, horizon=1)
        checks["fallback_invariant_binding"] = bool(
            inside_decision.feasible
            and inside_decision.bound_result is not None
            and inside_decision.bound_result.feasible
            and inside_decision.backup_invariant_fingerprint
            == inside_decision.bound_result.backup_invariant_fingerprint
        )
        far = np.zeros(15)
        far[2] = 2.0
        far_decision = fallback.decide(far, horizon=1)
        checks["fallback_unrecoverable_fail_closed"] = bool(
            far_decision.feasible is False
            and far_decision.bound_result is None
        )
        observations["fallback"] = {
            "inside_feasible": inside_decision.feasible,
            "inside_solver_status": inside_decision.solution.solver_status,
            "backup_invariant_fingerprint": inside_decision.backup_invariant_fingerprint,
            "far_feasible": far_decision.feasible,
            "far_solver_status": far_decision.solution.solver_status,
        }
    except Exception as error:
        errors.append({"check": "fallback", "type": type(error).__name__, "message": str(error)})

    all_passed = all(checks.values()) and not errors
    return {
        "result_id": "p4-g1a-mechanism-validation-v1",
        "stage": "g1a_mechanism_validation",
        "status": "passed" if all_passed else "scientific_failure",
        "all_required_checks_passed": all_passed,
        "formal_experiment_run": False,
        "sealed_data_used": False,
        "claim_generation_allowed": False,
        "authorization": {
            "manifest_id": manifest["manifest_id"],
            "manifest_hash": sha256_file(authorization_path),
            "code_tree_hash": manifest["code_hash"],
            "calibration_split_hash": manifest["split_hashes"]["calibration"],
            "development_split_hash": manifest["split_hashes"]["development"],
            "protocol_hash": sha256_file(protocol_path),
            "scenario_manifest_hash": scenario_manifest.fingerprint,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "checks": checks,
        "observations": observations,
        "errors": errors,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation_boundary": (
            "Deterministic G1a mechanism/falsification evidence only; not method "
            "superiority, platform certification, or confirmatory evidence."
        ),
    }


def run_and_write_g1a(
    authorization_path: str | Path,
    repo_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    output = Path(output_path).resolve()
    results_root = (root / "experiments" / "results").resolve()
    if results_root not in output.parents:
        raise G1ARunnerError("G1a output must be under experiments/results")
    if "sealed" in {part.lower() for part in output.parts}:
        raise G1ARunnerError("G1a output path must not reference sealed data")
    if output.exists():
        raise G1ARunnerError("G1a refuses to overwrite an existing result")
    result = evaluate_g1a(authorization_path, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise G1ARunnerError("temporary output already exists")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_and_write_g1a(
            args.authorization, args.repo_root, args.output,
        )
    except (G1ARunnerError, PreflightError, ValueError, json.JSONDecodeError) as error:
        print(f"G1A_PREFLIGHT_FAILED: {error}")
        return 3
    print(json.dumps({
        "status": result["status"],
        "all_required_checks_passed": result["all_required_checks_passed"],
        "output": str(Path(args.output).resolve()),
    }, sort_keys=True))
    return 0 if result["all_required_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
