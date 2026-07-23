"""Fail-closed one-shot Gazebo second-system development benchmark."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from .assurance_case import verify_assurance_case
from .dynamic_l0 import (
    FAMILIES,
    METHODS,
    RolloutResult,
    _assurance_bundle,
    _family_configuration,
    _inject_runtime_fault,
    _summarize,
)
from .environment import CompoundShift
from .g1a_runner import canonical_code_tree_hash
from .gazebo_second_system import GazeboAirGroundEnv
from .predictor import NominalRolloutHorizonPredictor
from .sandbox_baselines import SandboxBaselineInfeasible, SandboxNominalCBFAdapter
from .sandbox_fallback import sandbox_backup_equilibrium, sandbox_backup_gain
from .sandbox_task import SandboxComparisonTask


class GazeboBenchmarkError(RuntimeError):
    pass


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_file(root: Path, relative: Any, expected: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise GazeboBenchmarkError(f"{label} path is missing")
    if not isinstance(expected, str) or len(expected) != 64:
        raise GazeboBenchmarkError(f"{label} hash is invalid")
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise GazeboBenchmarkError(f"{label} escapes repository root")
    if not candidate.is_file() or _digest(candidate) != expected.lower():
        raise GazeboBenchmarkError(f"{label} hash mismatch")
    return candidate


def _contains_pending(value: Any) -> bool:
    if isinstance(value, str):
        return "PENDING" in value.upper()
    if isinstance(value, list):
        return any(_contains_pending(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_pending(item) for item in value.values())
    return False


def validate_gazebo_manifest(manifest: dict[str, Any], root: Path) -> Path:
    required = {
        "stage": "development_g1_second_system",
        "status": "frozen_g0_verified",
        "authorized": True,
        "formal_experiment_authorized": False,
        "sealed_data_authorized": False,
        "claim_generation_allowed": False,
        "scientific_failure_retry_allowed": False,
        "seed_top_up_allowed": False,
        "scientific_output_consumed": False,
        "gpu_required": False,
        "independent_unit": "scenario_seed",
        "primary_baseline": "nominal_cbf",
        "critical_component_baseline": "unbound_filter",
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise GazeboBenchmarkError(f"manifest {key} must equal {expected!r}")
    if _contains_pending(manifest):
        raise GazeboBenchmarkError("manifest contains an unresolved PENDING field")
    if tuple(manifest.get("methods", ())) != METHODS:
        raise GazeboBenchmarkError("method order differs from frozen Route-A protocol")
    if tuple(manifest.get("scenario_families", ())) != FAMILIES:
        raise GazeboBenchmarkError("scenario family order differs from frozen Route-A protocol")
    seeds = manifest.get("scenario_seeds")
    if not isinstance(seeds, list) or len(seeds) != 20 or len(set(seeds)) != 20:
        raise GazeboBenchmarkError("exactly 20 unique scenario seeds are required")
    if manifest.get("horizon_steps") != 80 or manifest.get("physics_iterations_per_step") != 10:
        raise GazeboBenchmarkError("horizon or Gazebo integration count differs from protocol")
    if manifest.get("controller_dt_seconds") != 0.1 or manifest.get("bootstrap_replicates") != 5000:
        raise GazeboBenchmarkError("controller period or bootstrap count differs from protocol")
    if manifest.get("code_tree_sha256") != canonical_code_tree_hash(root):
        raise GazeboBenchmarkError("current src+tests code tree does not match manifest")
    for prefix, label in (
        ("authorization", "authorization"),
        ("protocol", "protocol"),
        ("world", "Gazebo world"),
        ("g0_receipt", "Gazebo G0 receipt"),
    ):
        _bound_file(root, manifest.get(f"{prefix}_path"), manifest.get(f"{prefix}_sha256"), label)
    budgets = manifest.get("budgets", {})
    if budgets.get("max_rollouts") != 500 or budgets.get("gpu_count") != 0:
        raise GazeboBenchmarkError("rollout or GPU budget differs from protocol")
    if not isinstance(budgets.get("max_runtime_seconds"), int) or budgets["max_runtime_seconds"] <= 0:
        raise GazeboBenchmarkError("runtime budget is not frozen")
    if not isinstance(budgets.get("max_output_bytes"), int) or budgets["max_output_bytes"] <= 0:
        raise GazeboBenchmarkError("output budget is not frozen")
    receipt_path = (root / manifest["g0_receipt_path"]).resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("receipt_kind") != "gazebo_second_system_engineering_g0":
        raise GazeboBenchmarkError("wrong G0 receipt kind")
    if receipt.get("determinism_passed") is not True or receipt.get("maximum_replay_error", float("inf")) > 1e-9:
        raise GazeboBenchmarkError("Gazebo deterministic replay gate failed")
    if receipt.get("scientific_experiment_run") is not False or receipt.get("sealed_data_used") is not False:
        raise GazeboBenchmarkError("G0 receipt crossed the evidence boundary")
    return (root / manifest["world_path"]).resolve()


def _run_rollout(
    method: str,
    family: str,
    seed: int,
    world_path: Path,
    horizon: int,
) -> tuple[RolloutResult, dict[str, float]]:
    shift, observation_steps, communication_steps, fault = _family_configuration(family, seed)
    env = GazeboAirGroundEnv(world_path, shift, horizon=horizon)
    env.reset(seed=seed, interaction_close=family in {"interaction_risk", "compound_shift"})
    task = SandboxComparisonTask()
    cbf = SandboxNominalCBFAdapter(task)
    backup_center = sandbox_backup_equilibrium()
    backup_gain = sandbox_backup_gain()
    predictor = NominalRolloutHorizonPredictor(CompoundShift(), max_steps=40)
    states = [env.state.copy()]
    applied = [env._applied_action.copy()]
    initial_distance = float(np.linalg.norm(env.state[:3] - task.uav_goal) + np.linalg.norm(env.state[6:8] - task.ugv_goal))
    interventions = blocks = backups = 0
    action_cost = 0.0
    reasons: dict[str, int] = {}
    violated = False
    violation_step: int | None = None
    compute_seconds: list[float] = []
    dispatch_seconds: list[float] = []
    physics_seconds: list[float] = []
    return_seconds: list[float] = []
    for step in range(horizon):
        decision_start = time.perf_counter()
        index = max(0, len(states) - 1 - observation_steps)
        observed = states[index].copy()
        observed[[0, 1, 2, 6, 7]] += shift.sensor_bias
        augmented = np.concatenate([observed, applied[index]])
        nominal = task.nominal_action(augmented)
        true_augmented = np.concatenate([env.state, env._applied_action])
        backup = np.clip(backup_gain @ (true_augmented - backup_center), task.action_lower, task.action_upper)
        try:
            filtered = cbf.decide(augmented, fallback_action=backup).action
        except (SandboxBaselineInfeasible, ValueError):
            filtered = backup.copy()
        age = (observation_steps + communication_steps) * env.dt + 0.04
        predicted = predictor.predict(observed, filtered, previous_applied_action=applied[index], step_index=step)
        duration = max(0.0, predicted - 0.30 - age - 0.05)
        candidate = nominal.copy() if method in {"no_runtime_assurance", "fixed_ttl"} else filtered.copy()
        fault_now = fault is not None and step > 0 and step % 7 == 0
        bundle = _assurance_bundle(candidate, step * env.dt, duration, f"gazebo-{method}-{family}-{seed}-{step}")
        transmitted = candidate.copy()
        if fault_now:
            transmitted = _inject_runtime_fault(bundle, fault, step)
        reason = "executed"
        if method == "fixed_ttl" and age > 0.30:
            transmitted, reason = backup, "fixed_ttl_expired"
        elif method == "unbound_filter" and duration <= 0.0:
            transmitted, reason = backup, "adaptive_expiry"
        elif method == "full_assurance_case":
            verification = verify_assurance_case(bundle)
            if duration <= 0.0 or not verification.accepted:
                transmitted = backup
                reason = "zero_horizon" if duration <= 0.0 else verification.reason_code
                blocks += int(not verification.accepted)
        compute_seconds.append(time.perf_counter() - decision_start)
        if not np.allclose(transmitted, nominal):
            interventions += 1
        if np.allclose(transmitted, backup):
            backups += 1
        reasons[reason] = reasons.get(reason, 0) + 1
        action_cost += float(np.dot(transmitted, transmitted))
        _, _, terminated, truncated, info = env.step(transmitted)
        latency = info["latency_receipt"].as_seconds()
        dispatch_seconds.append(float(latency["dispatch_to_first_pre_update"]))
        physics_seconds.append(float(latency["first_pre_to_last_post_update"]))
        return_seconds.append(float(latency["post_update_to_return"]))
        states.append(env.state.copy())
        applied.append(env._applied_action.copy())
        if info["constraint_violation"]:
            violated, violation_step = True, step + 1
            break
        if terminated or truncated:
            break
    final_distance = float(np.linalg.norm(env.state[:3] - task.uav_goal) + np.linalg.norm(env.state[6:8] - task.ugv_goal))
    progress = (initial_distance - final_distance) / max(initial_distance, 1e-12)
    count = max(1, env.step_index)
    row = RolloutResult(method, seed, family, violated, violation_step, interventions, blocks, backups, count, float(progress), action_cost / count, reasons)
    latency_summary = {
        "decision_compute_max_seconds": max(compute_seconds, default=0.0),
        "decision_compute_mean_seconds": float(np.mean(compute_seconds)) if compute_seconds else 0.0,
        "dispatch_max_seconds": max(dispatch_seconds, default=0.0),
        "physics_callback_max_seconds": max(physics_seconds, default=0.0),
        "return_max_seconds": max(return_seconds, default=0.0),
        "deterministic_compute_bound_available": False,
    }
    return row, latency_summary


def _critical_component_summary(rows: list[RolloutResult], manifest: dict[str, Any]) -> dict[str, Any]:
    seeds = list(manifest["scenario_seeds"])
    index = {(r.method, r.scenario_seed, r.scenario_family): int(r.constraint_violation) for r in rows}

    def worst(method: str, chosen: list[int]) -> float:
        return max(float(np.mean([index[(method, int(seed), family)] for seed in chosen])) for family in FAMILIES)

    observed = worst("full_assurance_case", seeds) - worst("unbound_filter", seeds)
    rng = np.random.default_rng(int(manifest["bootstrap_seed"]))
    samples = []
    for _ in range(int(manifest["bootstrap_replicates"])):
        chosen = [int(value) for value in rng.choice(seeds, size=len(seeds), replace=True)]
        samples.append(worst("full_assurance_case", chosen) - worst("unbound_filter", chosen))
    return {
        "effect_full_minus_unbound_filter": observed,
        "paired_seed_cluster_bootstrap_95ci": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "claim_rule": "A null component effect restricts evidence binding to integrity, diagnosis, and traceability claims.",
    }


def run_gazebo_benchmark(manifest_path: str | Path, repo_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest_raw = manifest_file.read_bytes()
    manifest = json.loads(manifest_raw.decode("utf-8"))
    world_path = validate_gazebo_manifest(manifest, root)
    output = Path(output_path).resolve()
    expected_output = (root / manifest["output_path"]).resolve()
    if output != expected_output or root not in output.parents:
        raise GazeboBenchmarkError("output differs from manifest-bound path")
    if output.exists() or output.parent.exists():
        raise GazeboBenchmarkError("scientific output path already exists; rerun refused")
    schedule = [(seed, family, method) for seed in manifest["scenario_seeds"] for family in FAMILIES for method in METHODS]
    random.Random(int(manifest["randomization_seed"])).shuffle(schedule)
    start = time.perf_counter()
    rows: list[RolloutResult] = []
    latency_rows: list[dict[str, Any]] = []
    for order, (seed, family, method) in enumerate(schedule):
        if time.perf_counter() - start > manifest["budgets"]["max_runtime_seconds"]:
            raise GazeboBenchmarkError("runtime budget exceeded")
        row, latency = _run_rollout(method, family, int(seed), world_path, int(manifest["horizon_steps"]))
        rows.append(row)
        latency_rows.append({"order": order, "scenario_seed": seed, "scenario_family": family, "method": method, **latency})
    summary = _summarize(rows, manifest)
    summary["critical_component"] = _critical_component_summary(rows, manifest)
    result = {
        "result_id": "p4-gazebo-second-system-g1-v1",
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "code_tree_sha256": manifest["code_tree_sha256"],
        "world_sha256": manifest["world_sha256"],
        "formal_experiment_run": False,
        "sealed_data_used": False,
        "gpu_used": False,
        "claim_generation_allowed": False,
        "independent_unit": "scenario_seed",
        "schedule": [{"order": order, "scenario_seed": seed, "scenario_family": family, "method": method} for order, (seed, family, method) in enumerate(schedule)],
        "rows": [asdict(row) for row in rows],
        "latency_rows": latency_rows,
        "summary": summary,
        "elapsed_seconds": time.perf_counter() - start,
        "inference_boundary": "Unsealed two-simulator development evidence; not PX4/ROS2 SITL, hardware, formal certification, deterministic platform latency, arbitrary-shift safety, or G2 confirmation.",
    }
    body = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(body) > manifest["budgets"]["max_output_bytes"]:
        raise GazeboBenchmarkError("output budget exceeded")
    output.parent.mkdir(parents=True, exist_ok=False)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_bytes(body)
    temporary.replace(output)
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_gazebo_benchmark(args.manifest, args.repo_root, args.output)
    print(json.dumps({"rows": len(result["rows"]), "elapsed_seconds": result["elapsed_seconds"], "summary": result["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
