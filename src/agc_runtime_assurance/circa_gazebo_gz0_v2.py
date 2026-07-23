"""Claim-ineligible CIRCA Gazebo GZ0-v2 conflict-corridor runner."""

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
from .gazebo_second_system import GazeboAirGroundEnv
from .predictor import NominalRolloutHorizonPredictor
from .sandbox_baselines import SandboxBaselineInfeasible, SandboxNominalCBFAdapter
from .sandbox_fallback import sandbox_backup_equilibrium, sandbox_backup_gain
from .sandbox_task import SandboxComparisonTask


class CircaGazeboGZ0V2Error(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class V2RegimeSummary:
    candidate_id: str
    family_id: str
    scenario_seed: int
    hazard_active: bool
    regime: str
    first_violation: int
    first_violation_step: int | None
    intervened: int
    intervention_steps: int
    completed_steps: int


def _task(uav_goal_y: float, ugv_goal_y: float) -> SandboxComparisonTask:
    return SandboxComparisonTask(
        uav_goal=(4.0, float(uav_goal_y), 2.0),
        ugv_goal=(-4.0, float(ugv_goal_y)),
    )


def analytic_reachability(candidate: SceneCandidate, corridor_y: float, horizon_steps: int) -> dict[str, Any]:
    """Necessary route checks only; this function generates no Gazebo outcome."""
    task = _task(corridor_y, corridor_y)
    initial = candidate.initial_state()
    augmented = np.concatenate([initial, np.zeros(5, dtype=float)])
    nominal = task.nominal_action(augmented)
    initial_separation = float(np.linalg.norm(initial[:2] - initial[6:8]))
    longitudinal_time = (2.0 * candidate.half_gap - task.minimum_separation) / (
        2.0 * candidate.closing_speed
    )
    uav_lateral_error = corridor_y - initial[1]
    ugv_lateral_error = corridor_y - initial[7]
    lateral_convergent = bool(
        (initial[1] - initial[7]) * (nominal[1] - nominal[4]) < 0.0
        or abs(initial[1] - initial[7]) <= 1e-12
    )
    longitudinal_conflict = bool(nominal[0] > 0.0 and nominal[3] < 0.0)
    return {
        "candidate_id": candidate.candidate_id,
        "initial_planar_separation": initial_separation,
        "straight_line_threshold_crossing_seconds": longitudinal_time,
        "horizon_seconds": horizon_steps * 0.1,
        "nominal_initial_action": nominal.tolist(),
        "uav_lateral_goal_error": float(uav_lateral_error),
        "ugv_lateral_goal_error": float(ugv_lateral_error),
        "lateral_convergent": lateral_convergent,
        "longitudinal_conflict": longitudinal_conflict,
        "necessary_reachability_pass": bool(
            initial_separation > task.minimum_separation
            and 0.0 < longitudinal_time < horizon_steps * 0.1
            and lateral_convergent
            and longitudinal_conflict
        ),
    }


def validate_manifest(manifest: dict[str, Any], root: Path) -> tuple[Path, tuple[SceneCandidate, ...]]:
    required = {
        "stage": "claim_ineligible_gazebo_gz0_v2_conflict_corridor_development",
        "status": "authorized_exactly_once",
        "authorized": True,
        "scientific_claim_allowed": False,
        "circa_gz1_authorized": False,
        "retry_allowed": False,
        "seed_top_up_allowed": False,
        "sealed_data_authorized": False,
        "formal_experiment_authorized": False,
        "gpu_count": 0,
        "route": "shared_lateral_corridor_conflicting_waypoints",
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise CircaGazeboGZ0V2Error(f"manifest {key} must equal {expected!r}")
    if manifest.get("supersedes_result_reuse_allowed") is not False:
        raise CircaGazeboGZ0V2Error("GZ0-v1 result reuse must be forbidden")
    world = (root / manifest["world_path"]).resolve()
    if root not in world.parents or not world.is_file() or _sha256(world) != manifest["world_sha256"]:
        raise CircaGazeboGZ0V2Error("world source lock failed")
    for relative, expected in manifest.get("source_files", {}).items():
        source = (root / relative).resolve()
        if root not in source.parents or not source.is_file() or _sha256(source) != expected:
            raise CircaGazeboGZ0V2Error(f"source lock failed for {relative}")
    candidates = tuple(SceneCandidate.from_dict(value) for value in manifest["candidates"])
    if not candidates or len({item.candidate_id for item in candidates}) != len(candidates):
        raise CircaGazeboGZ0V2Error("candidate IDs must be non-empty and unique")
    families = tuple(manifest["families"])
    if set(item.family_id for item in candidates) != set(families):
        raise CircaGazeboGZ0V2Error("every registered family must have candidates")
    per_family = int(manifest["candidates_per_family"])
    if any(sum(item.family_id == family for item in candidates) != per_family for family in families):
        raise CircaGazeboGZ0V2Error("candidate count differs across families")
    expected_rows = len(candidates) * int(manifest["seeds_per_candidate"]) * 2
    if expected_rows != int(manifest["max_regime_rollouts"]):
        raise CircaGazeboGZ0V2Error("rollout budget does not match the frozen grid")
    hazard_indices = tuple(int(value) for value in manifest["hazard_active_seed_indices"])
    if (
        not hazard_indices
        or len(set(hazard_indices)) != len(hazard_indices)
        or min(hazard_indices) < 0
        or max(hazard_indices) >= int(manifest["seeds_per_candidate"])
        or len(hazard_indices) == int(manifest["seeds_per_candidate"])
    ):
        raise CircaGazeboGZ0V2Error("hazard-active indices must register both conflict and control roles")
    audits = [
        analytic_reachability(item, float(manifest["corridor_lateral_goal"]), int(manifest["horizon_steps"]))
        for item in candidates
    ]
    if not all(item["necessary_reachability_pass"] for item in audits):
        raise CircaGazeboGZ0V2Error("analytic conflict-corridor reachability gate failed")
    return world, candidates


def _run_regime(
    candidate: SceneCandidate,
    seed: int,
    regime: str,
    world: Path,
    horizon: int,
    corridor_y: float,
    hazard_active: bool,
    safe_uav_goal_y: float,
    safe_ugv_goal_y: float,
) -> V2RegimeSummary:
    if regime not in {"active_assurance", "shadow_no_override"}:
        raise CircaGazeboGZ0V2Error("unknown regime")
    env = GazeboAirGroundEnv(world, candidate.shift(), horizon=horizon)
    env.reset(seed=seed, initial_state=candidate.initial_state(), position_jitter_scale=candidate.position_jitter_scale)
    task = (
        _task(corridor_y, corridor_y)
        if hazard_active
        else _task(safe_uav_goal_y, safe_ugv_goal_y)
    )
    cbf = SandboxNominalCBFAdapter(task)
    backup_center, backup_gain = sandbox_backup_equilibrium(), sandbox_backup_gain()
    predictor = NominalRolloutHorizonPredictor(CompoundShift(), max_steps=40)
    states, applied = [env.state.copy()], [env._applied_action.copy()]
    intervened = intervention_steps = 0
    first_violation_step: int | None = None
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
        bundle = _assurance_bundle(filtered, step * env.dt, duration, f"circa-gz0-v2-{candidate.candidate_id}-{seed}-{step}")
        verification = verify_assurance_case(bundle)
        active_action = filtered if duration > 0.0 and verification.accepted else backup
        action = active_action if regime == "active_assurance" else nominal
        changed = not np.allclose(active_action, nominal)
        intervened |= int(changed)
        intervention_steps += int(changed)
        _, _, terminated, truncated, info = env.step(action)
        states.append(env.state.copy())
        applied.append(env._applied_action.copy())
        if info["constraint_violation"] and first_violation_step is None:
            first_violation_step = step + 1
        if terminated or truncated:
            break
    return V2RegimeSummary(
        candidate.candidate_id,
        candidate.family_id,
        seed,
        hazard_active,
        regime,
        int(first_violation_step is not None),
        first_violation_step,
        int(intervened),
        intervention_steps,
        env.step_index,
    )


def _diagnostic(rows: list[V2RegimeSummary], families: tuple[str, ...], manifest: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for family in families:
        candidates = []
        for candidate_id in sorted({row.candidate_id for row in rows if row.family_id == family}):
            subset = [row for row in rows if row.candidate_id == candidate_id]
            active = [row for row in subset if row.regime == "active_assurance"]
            shadow = [row for row in subset if row.regime == "shadow_no_override"]
            item = {
                "candidate_id": candidate_id,
                "shadow_violation_rate": float(np.mean([row.first_violation for row in shadow])),
                "active_violation_rate": float(np.mean([row.first_violation for row in active])),
                "active_intervention_rate": float(np.mean([row.intervened for row in active])),
            }
            item["event_floor_pass"] = bool(
                manifest["development_gates"]["shadow_violation_rate_min"]
                <= item["shadow_violation_rate"]
                <= manifest["development_gates"]["shadow_violation_rate_max"]
                and item["active_intervention_rate"] >= manifest["development_gates"]["active_intervention_rate_min"]
                and item["active_violation_rate"] <= manifest["development_gates"]["active_violation_rate_max"]
            )
            candidates.append(item)
        passing = [item for item in candidates if item["event_floor_pass"]]
        output[family] = {
            "all_candidates": candidates,
            "passing_candidate_count": len(passing),
            "selected": sorted(
                passing,
                key=lambda item: (
                    abs(item["shadow_violation_rate"] - manifest["target_shadow_violation_rate"]),
                    item["active_violation_rate"],
                    item["candidate_id"],
                ),
            )[0] if passing else None,
        }
    return output


def run_gz0_v2(manifest_path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest_raw = manifest_file.read_bytes()
    manifest = json.loads(manifest_raw)
    world, candidates = validate_manifest(manifest, root)
    output = (root / manifest["output_path"]).resolve()
    if root not in output.parents or output.exists():
        raise CircaGazeboGZ0V2Error("exactly-once output path already exists or is unsafe")
    schedule = []
    hazard_indices = {int(value) for value in manifest["hazard_active_seed_indices"]}
    for candidate in candidates:
        for index in range(int(manifest["seeds_per_candidate"])):
            digest = hashlib.sha256(
                f"circa-gz0-v2|{manifest['master_seed']}|{candidate.candidate_id}|{index}".encode()
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
            raise CircaGazeboGZ0V2Error("GZ0-v2 wall-time budget exceeded")
        rows.append(_run_regime(
            candidate,
            seed,
            regime,
            world,
            int(manifest["horizon_steps"]),
            float(manifest["corridor_lateral_goal"]),
            hazard_active,
            float(manifest["safe_uav_lateral_goal"]),
            float(manifest["safe_ugv_lateral_goal"]),
        ))
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
        "boundary": "claim-ineligible GZ0-v2 conflict-corridor development and budget measurement only",
    }
    body = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if len(body) > manifest["output_bytes_max"]:
        raise CircaGazeboGZ0V2Error("GZ0-v2 output budget exceeded")
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_bytes(body)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    result = run_gz0_v2(args.manifest, args.repo_root)
    print(json.dumps({
        "result_id": result["result_id"],
        "rows": len(result["rows"]),
        "elapsed_seconds": result["elapsed_seconds"],
        "all_families_have_passing_candidate": result["all_families_have_passing_candidate"],
        "claim_eligible": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
