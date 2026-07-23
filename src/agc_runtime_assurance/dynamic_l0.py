"""One-shot unsealed Route-A dynamic L0 benchmark.

The independent unit is a scenario seed.  This module deliberately contains no
training and makes no platform or confirmatory claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from .assurance_case import (
    assurance_audit_payload,
    build_valid_assurance_case,
    verify_assurance_case,
)
from .environment import AirGroundRuntimeEnv, CompoundShift
from .g1a_runner import canonical_code_tree_hash
from .predictor import NominalRolloutHorizonPredictor
from .sandbox_baselines import SandboxBaselineInfeasible, SandboxNominalCBFAdapter
from .sandbox_fallback import sandbox_backup_equilibrium, sandbox_backup_gain
from .sandbox_task import SandboxComparisonTask


METHODS = (
    "no_runtime_assurance", "fixed_ttl", "unbound_filter",
    "nominal_cbf", "full_assurance_case",
)
FAMILIES = (
    "dynamics_shift", "measurement_shift", "communication_shift",
    "interaction_risk", "compound_shift",
)


class DynamicL0Error(RuntimeError):
    pass


@dataclass(frozen=True)
class RolloutResult:
    method: str
    scenario_seed: int
    scenario_family: str
    constraint_violation: bool
    violation_step: int | None
    intervention_count: int
    pre_execution_block_count: int
    backup_count: int
    action_count: int
    task_progress: float
    action_cost: float
    reason_counts: dict[str, int]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_file(root: Path, relative: Any, expected: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise DynamicL0Error(f"{label} path is missing")
    if not isinstance(expected, str) or len(expected) != 64:
        raise DynamicL0Error(f"{label} hash is invalid")
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise DynamicL0Error(f"{label} escapes repo root")
    if not candidate.is_file() or _digest(candidate) != expected.lower():
        raise DynamicL0Error(f"{label} hash mismatch")
    return candidate


def validate_dynamic_l0_manifest(manifest: dict[str, Any], root: Path) -> None:
    required = {
        "stage": "development_l0",
        "authorized": True,
        "experiment_family": "route_a_dynamic_contract_benchmark",
        "formal_experiment_authorized": False,
        "sealed_data_authorized": False,
        "claim_generation_allowed": False,
        "gpu_required": False,
        "scientific_failure_retry_allowed": False,
        "seed_top_up_allowed": False,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise DynamicL0Error(f"manifest {key} must equal {value!r}")
    if tuple(manifest.get("methods", ())) != METHODS:
        raise DynamicL0Error("Tier-1 method order differs from frozen protocol")
    if tuple(manifest.get("scenario_families", ())) != FAMILIES:
        raise DynamicL0Error("scenario family order differs from frozen protocol")
    seeds = manifest.get("scenario_seeds")
    if not isinstance(seeds, list) or len(seeds) != 20 or len(set(seeds)) != 20:
        raise DynamicL0Error("exactly 20 unique scenario seeds are required")
    if not all(isinstance(seed, int) for seed in seeds):
        raise DynamicL0Error("scenario seeds must be integers")
    if manifest.get("horizon_steps") != 80 or manifest.get("bootstrap_replicates") != 5000:
        raise DynamicL0Error("horizon or bootstrap count differs from frozen protocol")
    if manifest.get("primary_baseline") != "nominal_cbf":
        raise DynamicL0Error("primary baseline must be nominal_cbf")
    if manifest.get("code_tree_sha256") != canonical_code_tree_hash(root):
        raise DynamicL0Error("current src+tests code tree does not match manifest")
    for prefix, label in (
        ("route_decision", "route decision"),
        ("authorization", "authorization"),
        ("protocol", "protocol"),
    ):
        _bound_file(root, manifest.get(f"{prefix}_path"), manifest.get(f"{prefix}_sha256"), label)
    budgets = manifest.get("budgets", {})
    if budgets.get("max_rollouts") != 500 or budgets.get("max_runtime_seconds") != 120:
        raise DynamicL0Error("runtime/rollout budget differs from frozen protocol")
    if budgets.get("max_output_bytes") != 5 * 1024 * 1024:
        raise DynamicL0Error("output budget differs from frozen protocol")


def _family_configuration(family: str, seed: int) -> tuple[CompoundShift, int, int, str | None]:
    rng = np.random.default_rng(seed + 1009 * FAMILIES.index(family))
    if family == "dynamics_shift":
        return CompoundShift(float(rng.uniform(.82, 1.22)), float(rng.uniform(.05, .22)), float(rng.uniform(.05, .25)), float(rng.uniform(.05, .32)), 0.0), int(rng.integers(0, 2)), int(rng.integers(0, 2)), None
    if family == "measurement_shift":
        return CompoundShift(sensor_bias=float(rng.uniform(-.22, .22))), int(rng.integers(1, 4)), int(rng.integers(0, 2)), "latency_fingerprint_mismatch"
    if family == "communication_shift":
        return CompoundShift(actuator_lag=float(rng.uniform(.10, .30))), int(rng.integers(1, 4)), int(rng.integers(2, 6)), "action_digest_mismatch"
    if family == "interaction_risk":
        return CompoundShift(actuator_lag=float(rng.uniform(.0, .18)), sensor_bias=float(rng.uniform(-.10, .10))), int(rng.integers(0, 3)), int(rng.integers(0, 3)), "constraint_contract_mismatch"
    return CompoundShift(float(rng.uniform(.82, 1.22)), float(rng.uniform(.08, .22)), float(rng.uniform(.08, .25)), float(rng.uniform(.15, .35)), float(rng.uniform(-.18, .18))), int(rng.integers(2, 5)), int(rng.integers(2, 6)), "action_digest_mismatch"


def _refresh_bundle(bundle: dict[str, Any]) -> None:
    values = bundle["action"]["values"]
    body = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    bundle["action"]["action_sha256"] = hashlib.sha256(body).hexdigest()
    payload = assurance_audit_payload(bundle)
    bundle["audit"]["payload"] = payload
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    bundle["audit"]["hash"] = hashlib.sha256((bundle["audit"]["previous_hash"] + canonical).encode()).hexdigest()


def _assurance_bundle(action: np.ndarray, now: float, duration: float, case_id: str) -> dict[str, Any]:
    bundle = build_valid_assurance_case(case_id=case_id, decision_mode="filtered", issued_at=now)
    bundle["action"]["values"] = [float(value) for value in action]
    bundle["action"]["valid_until"] = now + max(duration, 1e-6)
    bundle["action"]["consumed_at"] = now
    _refresh_bundle(bundle)
    return bundle


def _inject_runtime_fault(bundle: dict[str, Any], fault: str, step: int) -> np.ndarray:
    if fault == "action_digest_mismatch":
        changed = np.asarray(bundle["action"]["values"], dtype=float)
        changed = -changed[[3, 4, 2, 0, 1]]
        bundle["action"]["values"] = changed.tolist()
        return changed
    if fault == "latency_fingerprint_mismatch":
        bundle["evidence"]["recoverability_latency_sha256"] = hashlib.sha256(f"latency-{step}".encode()).hexdigest()
    elif fault == "constraint_contract_mismatch":
        bundle["evidence"]["filter_constraint_sha256"] = hashlib.sha256(f"constraint-{step}".encode()).hexdigest()
    return np.asarray(bundle["action"]["values"], dtype=float)


def run_rollout(method: str, family: str, seed: int, horizon: int = 80) -> RolloutResult:
    if method not in METHODS or family not in FAMILIES:
        raise ValueError("unknown method or family")
    shift, observation_steps, communication_steps, fault = _family_configuration(family, seed)
    env = AirGroundRuntimeEnv(shift, horizon=horizon)
    env.reset(seed=seed)
    if family in {"interaction_risk", "compound_shift"}:
        jitter = np.random.default_rng(seed).normal(0.0, .08, 2)
        env.state[:2] = np.array([-1.15, 0.0]) + jitter
        env.state[6:8] = np.array([1.15, 0.0]) - jitter
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
    for step in range(horizon):
        index = max(0, len(states) - 1 - observation_steps)
        observed = states[index].copy()
        observed[[0, 1, 2, 6, 7]] += shift.sensor_bias
        augmented = np.concatenate([observed, applied[index]])
        nominal = task.nominal_action(augmented)
        true_augmented = np.concatenate([env.state, env._applied_action])
        backup = np.clip(
            backup_gain @ (true_augmented - backup_center),
            task.action_lower, task.action_upper,
        )
        try:
            filtered = cbf.decide(augmented, fallback_action=backup).action
        except (SandboxBaselineInfeasible, ValueError):
            filtered = backup.copy()
        age = (observation_steps + communication_steps) * env.dt + .04
        predicted = predictor.predict(observed, filtered, previous_applied_action=applied[index], step_index=step)
        duration = max(0.0, predicted - .30 - age - .05)
        candidate = nominal.copy() if method in {"no_runtime_assurance", "fixed_ttl"} else filtered.copy()
        fault_now = fault is not None and step > 0 and step % 7 == 0
        bundle = _assurance_bundle(candidate, step * env.dt, duration, f"{method}-{family}-{seed}-{step}")
        transmitted = candidate.copy()
        if fault_now:
            transmitted = _inject_runtime_fault(bundle, fault, step)
        reason = "executed"
        if method == "fixed_ttl" and age > .30:
            transmitted, reason = backup, "fixed_ttl_expired"
        elif method == "unbound_filter" and duration <= 0.0:
            transmitted, reason = backup, "adaptive_expiry"
        elif method == "full_assurance_case":
            verification = verify_assurance_case(bundle)
            if duration <= 0.0 or not verification.accepted:
                transmitted = backup
                reason = "zero_horizon" if duration <= 0.0 else verification.reason_code
                blocks += int(not verification.accepted)
        if not np.allclose(transmitted, nominal):
            interventions += 1
        if np.allclose(transmitted, backup):
            backups += 1
        reasons[reason] = reasons.get(reason, 0) + 1
        action_cost += float(np.dot(transmitted, transmitted))
        _, _, terminated, truncated, info = env.step(transmitted)
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
    return RolloutResult(method, seed, family, violated, violation_step, interventions, blocks, backups, count, float(progress), action_cost / count, reasons)


def _summarize(rows: list[RolloutResult], manifest: dict[str, Any]) -> dict[str, Any]:
    rates: dict[str, dict[str, float]] = {}
    worst: dict[str, float] = {}
    guardrails: dict[str, dict[str, float]] = {}
    for method in METHODS:
        subset = [row for row in rows if row.method == method]
        rates[method] = {
            family: float(np.mean([row.constraint_violation for row in subset if row.scenario_family == family]))
            for family in FAMILIES
        }
        worst[method] = max(rates[method].values())
        guardrails[method] = {
            "intervention_rate": float(sum(row.intervention_count for row in subset) / sum(row.action_count for row in subset)),
            "backup_rate": float(sum(row.backup_count for row in subset) / sum(row.action_count for row in subset)),
            "mean_task_progress": float(np.mean([row.task_progress for row in subset])),
            "mean_action_cost": float(np.mean([row.action_cost for row in subset])),
        }
    primary = "nominal_cbf"
    effect = worst["full_assurance_case"] - worst[primary]
    seeds = list(manifest["scenario_seeds"])
    rng = np.random.default_rng(int(manifest["bootstrap_seed"]))
    samples = []
    index = {(r.method, r.scenario_seed, r.scenario_family): int(r.constraint_violation) for r in rows}
    for _ in range(int(manifest["bootstrap_replicates"])):
        chosen = rng.choice(seeds, size=len(seeds), replace=True)
        full_worst = max(float(np.mean([index[("full_assurance_case", int(s), f)] for s in chosen])) for f in FAMILIES)
        base_worst = max(float(np.mean([index[(primary, int(s), f)] for s in chosen])) for f in FAMILIES)
        samples.append(full_worst - base_worst)
    interval = [float(x) for x in np.quantile(samples, [.025, .975])]
    full_guard = guardrails["full_assurance_case"]
    progress_ratio = full_guard["mean_task_progress"] / max(guardrails[primary]["mean_task_progress"], 1e-12)
    gates = {
        "effect_at_or_below_sesoi": effect <= float(manifest["success_gates"]["max_primary_effect"]),
        "bootstrap_upper_below_zero": interval[1] < 0.0,
        "intervention_rate_within_limit": full_guard["intervention_rate"] <= float(manifest["success_gates"]["max_intervention_rate"]),
        "task_progress_ratio_within_limit": progress_ratio >= float(manifest["success_gates"]["min_progress_ratio_to_primary"]),
    }
    return {"family_violation_rates": rates, "worst_family_violation_rates": worst, "primary_effect_full_minus_nominal_cbf": effect, "paired_seed_cluster_bootstrap_95ci": interval, "guardrails": guardrails, "task_progress_ratio_to_primary": progress_ratio, "success_gates": gates, "all_success_gates_passed": all(gates.values())}


def run_dynamic_l0(manifest_path: str | Path, repo_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest_raw = manifest_file.read_bytes()
    manifest = json.loads(manifest_raw.decode("utf-8"))
    validate_dynamic_l0_manifest(manifest, root)
    output = Path(output_path).resolve()
    if output.exists():
        raise DynamicL0Error("output already exists; scientific rerun refused")
    if root not in output.parents:
        raise DynamicL0Error("output path escapes repo root")
    schedule = [(seed, family, method) for seed in manifest["scenario_seeds"] for family in FAMILIES for method in METHODS]
    random.Random(int(manifest["randomization_seed"])).shuffle(schedule)
    start = time.perf_counter()
    rows: list[RolloutResult] = []
    for seed, family, method in schedule:
        if time.perf_counter() - start > manifest["budgets"]["max_runtime_seconds"]:
            raise DynamicL0Error("runtime budget exceeded")
        rows.append(run_rollout(method, family, int(seed), int(manifest["horizon_steps"])))
    result = {
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "code_tree_sha256": manifest["code_tree_sha256"],
        "formal_experiment_run": False,
        "sealed_data_used": False,
        "gpu_used": False,
        "claim_generation_allowed": False,
        "independent_unit": "scenario_seed",
        "schedule": [{"order": i, "scenario_seed": s, "scenario_family": f, "method": m} for i, (s, f, m) in enumerate(schedule)],
        "rows": [asdict(row) for row in rows],
        "summary": _summarize(rows, manifest),
        "elapsed_seconds": time.perf_counter() - start,
        "inference_boundary": "Unsealed development sandbox; scenario seeds are paired independent units. No platform, arbitrary-shift, author-method-superiority, formal or sealed claim.",
    }
    body = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if len(body) > manifest["budgets"]["max_output_bytes"]:
        raise DynamicL0Error("output budget exceeded")
    output.parent.mkdir(parents=True, exist_ok=False)
    temporary = output.with_suffix(output.suffix + ".tmp")
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
    run_dynamic_l0(args.manifest, args.repo_root, args.output)


if __name__ == "__main__":
    main()
