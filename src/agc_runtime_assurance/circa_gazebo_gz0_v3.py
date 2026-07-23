"""Traceable claim-ineligible CIRCA Gazebo GZ0-v3 runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from .assurance_case import verify_assurance_case
from .circa_gazebo_gz0 import SceneCandidate
from .dynamic_l0 import _assurance_bundle
from .environment import CompoundShift
from .gazebo_second_system import GazeboAirGroundEnv, constraint_margins
from .predictor import NominalRolloutHorizonPredictor
from .sandbox_baselines import SandboxBaselineInfeasible, SandboxNominalCBFAdapter
from .sandbox_fallback import sandbox_backup_equilibrium, sandbox_backup_gain
from .sandbox_task import SandboxComparisonTask


class CircaGazeboGZ0V3Error(RuntimeError):
    pass


@dataclass(frozen=True)
class OperationalEnvelope:
    hard_separation_m: float
    design_relative_speed_mps: float
    reaction_time_s: float
    relative_braking_deceleration_mps2: float
    uncertainty_allowance_m: float
    operational_separation_m: float


@dataclass(frozen=True)
class V3RegimeSummary:
    candidate_id: str
    family_id: str
    scenario_seed: int
    hazard_active: bool
    regime: str
    operational_separation_m: float
    operational_first_violation: int
    operational_first_violation_step: int | None
    hard_first_violation: int
    hard_first_violation_step: int | None
    applied_intervention: int
    applied_intervention_steps: int
    counterfactual_action_diverged: int
    first_action_divergence_step: int | None
    minimum_planar_separation_m: float
    minimum_planar_separation_step: int
    initial_operational_margin_m: float
    minimum_operational_margin_m: float
    maximum_relative_speed_mps: float
    design_speed_envelope_exceeded: bool
    minimum_uav_margin: float
    minimum_ugv_margin: float
    minimum_hard_separation_margin: float
    final_relative_x_m: float
    final_relative_y_m: float
    completed_steps: int


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_operational_envelope(candidate: SceneCandidate, assumptions: dict[str, float]) -> OperationalEnvelope:
    hard = float(assumptions["hard_separation_m"])
    speed_floor = float(assumptions["design_relative_speed_floor_mps"])
    compute = float(assumptions["compute_dispatch_budget_s"])
    braking = float(assumptions["relative_braking_deceleration_mps2"])
    uncertainty = float(assumptions["uncertainty_allowance_m"])
    values = np.asarray([hard, speed_floor, compute, braking, uncertainty], dtype=float)
    if not np.all(np.isfinite(values)) or hard <= 0.0 or speed_floor <= 0.0 or compute < 0.0 or braking <= 0.0 or uncertainty < 0.0:
        raise CircaGazeboGZ0V3Error("invalid operational-envelope assumptions")
    speed = max(speed_floor, 2.0 * candidate.closing_speed)
    reaction = compute + 0.1 * (candidate.observation_delay_steps + candidate.communication_delay_steps)
    distance = hard + speed * reaction + speed * speed / (2.0 * braking) + uncertainty
    return OperationalEnvelope(hard, speed, reaction, braking, uncertainty, distance)


def _task(
    uav_goal_y: float,
    ugv_goal_y: float,
    minimum_separation: float,
    action_limit: float,
) -> SandboxComparisonTask:
    return SandboxComparisonTask(
        uav_goal=(4.0, float(uav_goal_y), 2.0),
        ugv_goal=(-4.0, float(ugv_goal_y)),
        minimum_separation=float(minimum_separation),
        action_limit=float(action_limit),
    )


def validate_manifest(manifest: dict[str, Any], root: Path) -> tuple[Path, tuple[SceneCandidate, ...]]:
    required = {
        "stage": "claim_ineligible_gazebo_gz0_v3_traceable_operational_envelope_development",
        "status": "authorized_exactly_once",
        "authorized": True,
        "scientific_claim_allowed": False,
        "circa_gz1_authorized": False,
        "retry_allowed": False,
        "seed_top_up_allowed": False,
        "sealed_data_authorized": False,
        "formal_experiment_authorized": False,
        "gpu_count": 0,
        "route": "prospective_operational_envelope_with_traceable_conflict_corridor",
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise CircaGazeboGZ0V3Error(f"manifest {key} must equal {expected!r}")
    if manifest.get("prior_result_reuse_allowed") is not False:
        raise CircaGazeboGZ0V3Error("prior GZ0 result reuse must be forbidden")
    if not np.isfinite(manifest.get("task_action_limit", np.nan)) or manifest["task_action_limit"] <= 0.0:
        raise CircaGazeboGZ0V3Error("task action limit must be positive and finite")
    world = (root / manifest["world_path"]).resolve()
    if root not in world.parents or not world.is_file() or _sha256(world) != manifest["world_sha256"]:
        raise CircaGazeboGZ0V3Error("world source lock failed")
    for relative, expected in manifest.get("source_files", {}).items():
        source = (root / relative).resolve()
        if root not in source.parents or not source.is_file() or _sha256(source) != expected:
            raise CircaGazeboGZ0V3Error(f"source lock failed for {relative}")
    candidates = tuple(SceneCandidate.from_dict(value) for value in manifest["candidates"])
    if not candidates or len({item.candidate_id for item in candidates}) != len(candidates):
        raise CircaGazeboGZ0V3Error("candidate IDs must be non-empty and unique")
    families = tuple(manifest["families"])
    if set(item.family_id for item in candidates) != set(families):
        raise CircaGazeboGZ0V3Error("every registered family must have candidates")
    per_family = int(manifest["candidates_per_family"])
    if any(sum(item.family_id == family for item in candidates) != per_family for family in families):
        raise CircaGazeboGZ0V3Error("candidate count differs across families")
    expected_rows = len(candidates) * int(manifest["seeds_per_candidate"]) * 2
    if expected_rows != int(manifest["max_regime_rollouts"]):
        raise CircaGazeboGZ0V3Error("rollout budget does not match the frozen grid")
    hazard_indices = tuple(int(value) for value in manifest["hazard_active_seed_indices"])
    if (
        not hazard_indices
        or len(set(hazard_indices)) != len(hazard_indices)
        or min(hazard_indices) < 0
        or max(hazard_indices) >= int(manifest["seeds_per_candidate"])
        or len(hazard_indices) == int(manifest["seeds_per_candidate"])
    ):
        raise CircaGazeboGZ0V3Error("hazard-active indices must register conflict and control roles")
    for candidate in candidates:
        envelope = derive_operational_envelope(candidate, manifest["operational_envelope_assumptions"])
        initial = candidate.initial_state()
        initial_separation = float(np.linalg.norm(initial[:2] - initial[6:8]))
        if initial_separation - envelope.operational_separation_m < manifest["initial_operational_margin_min_m"]:
            raise CircaGazeboGZ0V3Error(f"initial operational margin too small for {candidate.candidate_id}")
        threshold_time = (2.0 * candidate.half_gap - envelope.operational_separation_m) / (
            2.0 * candidate.closing_speed
        )
        if not 0.0 < threshold_time < int(manifest["horizon_steps"]) * 0.1:
            raise CircaGazeboGZ0V3Error(f"operational threshold is not straight-line reachable for {candidate.candidate_id}")
    return world, candidates


def _run_regime(
    candidate: SceneCandidate,
    seed: int,
    hazard_active: bool,
    regime: str,
    world: Path,
    manifest: dict[str, Any],
) -> V3RegimeSummary:
    if regime not in {"active_assurance", "shadow_no_override"}:
        raise CircaGazeboGZ0V3Error("unknown regime")
    horizon = int(manifest["horizon_steps"])
    envelope = derive_operational_envelope(candidate, manifest["operational_envelope_assumptions"])
    env = GazeboAirGroundEnv(world, candidate.shift(), horizon=horizon)
    env.reset(seed=seed, initial_state=candidate.initial_state(), position_jitter_scale=candidate.position_jitter_scale)
    corridor = float(manifest["corridor_lateral_goal"])
    action_limit = float(manifest["task_action_limit"])
    task = (
        _task(corridor, corridor, envelope.operational_separation_m, action_limit)
        if hazard_active
        else _task(
            float(manifest["safe_uav_lateral_goal"]),
            float(manifest["safe_ugv_lateral_goal"]),
            envelope.operational_separation_m,
            action_limit,
        )
    )
    cbf = SandboxNominalCBFAdapter(task)
    backup_center, backup_gain = sandbox_backup_equilibrium(), sandbox_backup_gain()
    predictor = NominalRolloutHorizonPredictor(CompoundShift(), max_steps=40)
    states, applied = [env.state.copy()], [env._applied_action.copy()]
    initial_separation = float(np.linalg.norm(env.state[:2] - env.state[6:8]))
    min_sep, min_sep_step = initial_separation, 0
    min_operational = initial_separation - envelope.operational_separation_m
    initial_operational = min_operational
    max_relative_speed = float(np.linalg.norm(env.state[3:5] - env.state[8:10]))
    margins0 = constraint_margins(env.state, 0)
    min_uav, min_ugv, min_hard = margins0.uav_local, margins0.ugv_local, margins0.coupling
    operational_step = hard_step = divergence_step = None
    applied_intervention = applied_intervention_steps = 0
    counterfactual_diverged = 0
    for step in range(horizon):
        index = max(0, len(states) - 1 - candidate.observation_delay_steps)
        observed = states[index].copy()
        observed[[0, 1, 2, 6, 7]] += candidate.sensor_bias
        augmented = np.concatenate([observed, applied[index]])
        nominal = task.nominal_action(augmented)
        true_augmented = np.concatenate([env.state, env._applied_action])
        backup = np.clip(backup_gain @ (true_augmented - backup_center), task.action_lower, task.action_upper)
        try:
            filtered = cbf.decide(augmented, fallback_action=backup).action
        except (SandboxBaselineInfeasible, ValueError):
            filtered = backup.copy()
        age = (candidate.observation_delay_steps + candidate.communication_delay_steps) * env.dt + 0.04
        predicted = predictor.predict(observed, filtered, previous_applied_action=applied[index], step_index=step)
        duration = max(0.0, predicted - 0.30 - age - 0.05)
        bundle = _assurance_bundle(filtered, step * env.dt, duration, f"circa-gz0-v3-{candidate.candidate_id}-{seed}-{step}")
        verification = verify_assurance_case(bundle)
        active_action = filtered if duration > 0.0 and verification.accepted else backup
        changed = not np.allclose(active_action, nominal)
        counterfactual_diverged |= int(changed)
        if changed and divergence_step is None:
            divergence_step = step + 1
        action = active_action if regime == "active_assurance" else nominal
        if regime == "active_assurance":
            applied_intervention |= int(changed)
            applied_intervention_steps += int(changed)
        _, _, _, truncated, info = env.step(action)
        states.append(env.state.copy())
        applied.append(env._applied_action.copy())
        separation = float(np.linalg.norm(env.state[:2] - env.state[6:8]))
        relative_speed = float(np.linalg.norm(env.state[3:5] - env.state[8:10]))
        current = info["margins"]
        if separation < min_sep:
            min_sep, min_sep_step = separation, step + 1
        min_operational = min(min_operational, separation - envelope.operational_separation_m)
        max_relative_speed = max(max_relative_speed, relative_speed)
        min_uav = min(min_uav, current.uav_local)
        min_ugv = min(min_ugv, current.ugv_local)
        min_hard = min(min_hard, current.coupling)
        if separation < envelope.operational_separation_m and operational_step is None:
            operational_step = step + 1
        if current.coupling < 0.0 and hard_step is None:
            hard_step = step + 1
        if truncated:
            break
    relative = env.state[:2] - env.state[6:8]
    return V3RegimeSummary(
        candidate.candidate_id,
        candidate.family_id,
        seed,
        hazard_active,
        regime,
        envelope.operational_separation_m,
        int(operational_step is not None),
        operational_step,
        int(hard_step is not None),
        hard_step,
        int(applied_intervention),
        applied_intervention_steps,
        int(counterfactual_diverged),
        divergence_step,
        min_sep,
        min_sep_step,
        initial_operational,
        min_operational,
        max_relative_speed,
        bool(max_relative_speed > envelope.design_relative_speed_mps + manifest["design_speed_tolerance_mps"]),
        min_uav,
        min_ugv,
        min_hard,
        float(relative[0]),
        float(relative[1]),
        env.step_index,
    )


def _diagnostic(rows: list[V3RegimeSummary], families: tuple[str, ...], manifest: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for family in families:
        candidates = []
        for candidate_id in sorted({row.candidate_id for row in rows if row.family_id == family}):
            subset = [row for row in rows if row.candidate_id == candidate_id]
            active = [row for row in subset if row.regime == "active_assurance"]
            shadow = [row for row in subset if row.regime == "shadow_no_override"]
            item = {
                "candidate_id": candidate_id,
                "shadow_operational_violation_rate": float(np.mean([row.operational_first_violation for row in shadow])),
                "active_operational_violation_rate": float(np.mean([row.operational_first_violation for row in active])),
                "active_intervention_rate": float(np.mean([row.applied_intervention for row in active])),
                "speed_envelope_exceedance_rate": float(np.mean([row.design_speed_envelope_exceeded for row in subset])),
            }
            gates = manifest["development_gates"]
            item["event_floor_pass"] = bool(
                gates["shadow_operational_violation_rate_min"]
                <= item["shadow_operational_violation_rate"]
                <= gates["shadow_operational_violation_rate_max"]
                and item["active_intervention_rate"] >= gates["active_intervention_rate_min"]
                and item["active_operational_violation_rate"] <= gates["active_operational_violation_rate_max"]
                and item["speed_envelope_exceedance_rate"] <= gates["speed_envelope_exceedance_rate_max"]
            )
            candidates.append(item)
        passing = [item for item in candidates if item["event_floor_pass"]]
        output[family] = {
            "all_candidates": candidates,
            "passing_candidate_count": len(passing),
            "selected": sorted(
                passing,
                key=lambda item: (
                    abs(item["shadow_operational_violation_rate"] - manifest["target_shadow_operational_violation_rate"]),
                    item["active_operational_violation_rate"],
                    item["candidate_id"],
                ),
            )[0] if passing else None,
        }
    return output


def run_gz0_v3(manifest_path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest_raw = manifest_file.read_bytes()
    manifest = json.loads(manifest_raw)
    world, candidates = validate_manifest(manifest, root)
    output = (root / manifest["output_path"]).resolve()
    if root not in output.parents or output.exists():
        raise CircaGazeboGZ0V3Error("exactly-once output path already exists or is unsafe")
    hazard_indices = {int(value) for value in manifest["hazard_active_seed_indices"]}
    schedule = []
    for candidate in candidates:
        for index in range(int(manifest["seeds_per_candidate"])):
            digest = hashlib.sha256(
                f"circa-gz0-v3|{manifest['master_seed']}|{candidate.candidate_id}|{index}".encode()
            ).digest()
            seed = int.from_bytes(digest[:8], "big") % (2**31 - 1)
            hazard_active = index in hazard_indices
            schedule.extend([
                (candidate, seed, hazard_active, "active_assurance"),
                (candidate, seed, hazard_active, "shadow_no_override"),
            ])
    random.Random(int(manifest["schedule_seed"])).shuffle(schedule)
    start = time.perf_counter()
    rows = []
    for candidate, seed, hazard_active, regime in schedule:
        if time.perf_counter() - start > manifest["wall_time_seconds_max"]:
            raise CircaGazeboGZ0V3Error("GZ0-v3 wall-time budget exceeded")
        rows.append(_run_regime(candidate, seed, hazard_active, regime, world, manifest))
    elapsed = time.perf_counter() - start
    diagnostic = _diagnostic(rows, tuple(manifest["families"]), manifest)
    result = {
        "result_id": manifest["manifest_id"],
        "claim_eligible": False,
        "circa_gz1_run": False,
        "sealed_data_used": False,
        "formal_experiment_run": False,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "world_sha256": manifest["world_sha256"],
        "rows": [asdict(row) for row in rows],
        "development_diagnostic": diagnostic,
        "all_families_have_passing_candidate": all(
            value["passing_candidate_count"] > 0 for value in diagnostic.values()
        ),
        "elapsed_seconds": elapsed,
        "boundary": "claim-ineligible GZ0-v3 traceable operational-envelope development only",
    }
    body = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if len(body) > manifest["output_bytes_max"]:
        raise CircaGazeboGZ0V3Error("GZ0-v3 output budget exceeded")
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_bytes(body)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    result = run_gz0_v3(args.manifest, args.repo_root)
    print(json.dumps({
        "result_id": result["result_id"],
        "rows": len(result["rows"]),
        "elapsed_seconds": result["elapsed_seconds"],
        "all_families_have_passing_candidate": result["all_families_have_passing_candidate"],
        "claim_eligible": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
