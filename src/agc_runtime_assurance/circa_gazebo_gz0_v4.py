"""Exactly-once, claim-ineligible CIRCA Gazebo GZ0-v4 post-repair runner."""

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
from .circa_gazebo_gz0 import SceneCandidate
from .circa_gazebo_gz0_v3 import (
    V3RegimeSummary,
    _diagnostic,
    _task,
    derive_operational_envelope,
    validate_manifest as validate_v3_manifest,
)
from .dynamic_l0 import _assurance_bundle
from .environment import CompoundShift
from .gazebo_second_system import constraint_margins
from .gazebo_second_system_v3 import GazeboAirGroundEnvV3
from .predictor import NominalRolloutHorizonPredictor
from .sandbox_baselines import SandboxBaselineInfeasible, SandboxNominalCBFAdapter
from .sandbox_fallback import sandbox_backup_equilibrium, sandbox_backup_gain


class CircaGazeboGZ0V4Error(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_candidate(value: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(value)
    normalized.pop("candidate_id", None)
    return normalized


def _assert_scientific_parameters_unchanged(manifest: dict[str, Any], frozen: dict[str, Any]) -> None:
    scalar_keys = (
        "families",
        "candidates_per_family",
        "seeds_per_candidate",
        "hazard_active_seed_indices",
        "horizon_steps",
        "corridor_lateral_goal",
        "safe_uav_lateral_goal",
        "safe_ugv_lateral_goal",
        "task_action_limit",
        "operational_envelope_assumptions",
        "initial_operational_margin_min_m",
        "design_speed_tolerance_mps",
        "target_shadow_operational_violation_rate",
        "development_gates",
        "max_regime_rollouts",
        "wall_time_seconds_max",
        "output_bytes_max",
    )
    changed = [key for key in scalar_keys if manifest.get(key) != frozen.get(key)]
    if changed:
        raise CircaGazeboGZ0V4Error(f"scientific/operational parameters changed: {changed}")
    current_candidates = [_normalized_candidate(value) for value in manifest.get("candidates", [])]
    frozen_candidates = [_normalized_candidate(value) for value in frozen.get("candidates", [])]
    if current_candidates != frozen_candidates:
        raise CircaGazeboGZ0V4Error("candidate parameters or order changed from frozen GZ0-v3")


def validate_manifest(manifest: dict[str, Any], root: Path) -> tuple[Path, tuple[SceneCandidate, ...]]:
    required = {
        "stage": "claim_ineligible_gazebo_gz0_v4_post_repair_operational_envelope_development",
        "status": "authorized_exactly_once",
        "authorized": True,
        "scientific_claim_allowed": False,
        "circa_gz1_authorized": False,
        "retry_allowed": False,
        "seed_top_up_allowed": False,
        "sealed_data_authorized": False,
        "formal_experiment_authorized": False,
        "gpu_count": 0,
        "resource_class": "CPU-SHARED",
        "route": "binding_repair_driver_with_unchanged_operational_envelope",
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise CircaGazeboGZ0V4Error(f"manifest {key} must equal {expected!r}")
    if manifest.get("prior_result_reuse_allowed") is not False:
        raise CircaGazeboGZ0V4Error("prior GZ0 result reuse must be forbidden")
    frozen_path = (root / manifest["frozen_scientific_source_manifest_path"]).resolve()
    if root not in frozen_path.parents or not frozen_path.is_file():
        raise CircaGazeboGZ0V4Error("frozen scientific source manifest is unsafe or absent")
    if _sha256(frozen_path) != manifest["frozen_scientific_source_manifest_sha256"]:
        raise CircaGazeboGZ0V4Error("frozen GZ0-v3 manifest lock failed")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    _assert_scientific_parameters_unchanged(manifest, frozen)
    driver_path = (root / manifest["bound_driver_path"]).resolve()
    if root not in driver_path.parents or not driver_path.is_file():
        raise CircaGazeboGZ0V4Error("bound driver path is unsafe or absent")
    if _sha256(driver_path) != manifest["bound_driver_sha256"]:
        raise CircaGazeboGZ0V4Error("bound post-repair driver hash failed")
    delegated = dict(manifest)
    delegated.update({
        "stage": "claim_ineligible_gazebo_gz0_v3_traceable_operational_envelope_development",
        "route": "prospective_operational_envelope_with_traceable_conflict_corridor",
    })
    try:
        return validate_v3_manifest(delegated, root)
    except Exception as exc:
        raise CircaGazeboGZ0V4Error(str(exc)) from exc


def _run_regime(
    candidate: SceneCandidate,
    seed: int,
    hazard_active: bool,
    regime: str,
    world: Path,
    manifest: dict[str, Any],
) -> V3RegimeSummary:
    if regime not in {"active_assurance", "shadow_no_override"}:
        raise CircaGazeboGZ0V4Error("unknown regime")
    horizon = int(manifest["horizon_steps"])
    envelope = derive_operational_envelope(candidate, manifest["operational_envelope_assumptions"])
    env = GazeboAirGroundEnvV3(world, candidate.shift(), horizon=horizon)
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
        bundle = _assurance_bundle(filtered, step * env.dt, duration, f"circa-gz0-v4-{candidate.candidate_id}-{seed}-{step}")
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
        candidate.candidate_id, candidate.family_id, seed, hazard_active, regime,
        envelope.operational_separation_m, int(operational_step is not None), operational_step,
        int(hard_step is not None), hard_step, int(applied_intervention), applied_intervention_steps,
        int(counterfactual_diverged), divergence_step, min_sep, min_sep_step, initial_operational,
        min_operational, max_relative_speed,
        bool(max_relative_speed > envelope.design_relative_speed_mps + manifest["design_speed_tolerance_mps"]),
        min_uav, min_ugv, min_hard, float(relative[0]), float(relative[1]), env.step_index,
    )


def run_gz0_v4(manifest_path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest_raw = manifest_file.read_bytes()
    manifest = json.loads(manifest_raw)
    world, candidates = validate_manifest(manifest, root)
    output = (root / manifest["output_path"]).resolve()
    if root not in output.parents or output.exists():
        raise CircaGazeboGZ0V4Error("exactly-once output path already exists or is unsafe")
    hazard_indices = {int(value) for value in manifest["hazard_active_seed_indices"]}
    schedule = []
    for candidate in candidates:
        for index in range(int(manifest["seeds_per_candidate"])):
            digest = hashlib.sha256(
                f"circa-gz0-v4|{manifest['master_seed']}|{candidate.candidate_id}|{index}".encode()
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
            raise CircaGazeboGZ0V4Error("GZ0-v4 wall-time budget exceeded")
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
        "bound_driver_path": manifest["bound_driver_path"],
        "bound_driver_sha256": manifest["bound_driver_sha256"],
        "frozen_scientific_source_manifest_sha256": manifest["frozen_scientific_source_manifest_sha256"],
        "rows": [asdict(row) for row in rows],
        "development_diagnostic": diagnostic,
        "all_families_have_passing_candidate": all(
            value["passing_candidate_count"] > 0 for value in diagnostic.values()
        ),
        "elapsed_seconds": elapsed,
        "boundary": "claim-ineligible GZ0-v4 post-repair development only; no GZ1 authorization",
    }
    body = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if len(body) > manifest["output_bytes_max"]:
        raise CircaGazeboGZ0V4Error("GZ0-v4 output budget exceeded")
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_bytes(body)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    result = run_gz0_v4(args.manifest, args.repo_root)
    print(json.dumps({
        "result_id": result["result_id"],
        "rows": len(result["rows"]),
        "elapsed_seconds": result["elapsed_seconds"],
        "all_families_have_passing_candidate": result["all_families_have_passing_candidate"],
        "claim_eligible": False,
        "circa_gz1_run": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
