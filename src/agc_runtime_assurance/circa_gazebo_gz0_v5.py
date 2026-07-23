"""Claim-ineligible factorial mechanism runner for CIRCA Gazebo GZ0-v5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from .assurance_case import verify_assurance_case
from .circa_gazebo_gz0 import SceneCandidate
from .circa_gazebo_gz0_v3 import _task, derive_operational_envelope
from .dynamic_l0 import _assurance_bundle
from .environment import CompoundShift
from .gazebo_predictive_shield import PredictiveSeparationShield
from .gazebo_second_system import UAV_QUADRATIC_DRAG, UGV_QUADRATIC_DRAG, constraint_margins
from .gazebo_second_system_v3 import GazeboAirGroundEnvV3
from .gazebo_second_system_v4 import GazeboAirGroundEnvV4
from .predictor import NominalRolloutHorizonPredictor
from .sandbox_baselines import SandboxBaselineInfeasible, SandboxNominalCBFAdapter
from .sandbox_fallback import sandbox_backup_equilibrium, sandbox_backup_gain


class CircaGazeboGZ0V5Error(RuntimeError):
    pass


DRIVERS = ("command_persistent_unbounded_v3", "planar_speed_projected_v4")
CONTROLLERS = ("shadow_no_override", "registered_one_step_cbf", "predictive_separation_shield")
TRACE_FIELDS = (
    "candidate_id", "family_id", "scenario_seed", "hazard_active", "driver_id", "controller_id",
    "operational_separation_m", "design_relative_speed_mps", "operational_first_violation",
    "operational_first_violation_step", "hard_first_violation", "hard_first_violation_step",
    "applied_intervention", "applied_intervention_steps", "counterfactual_action_diverged",
    "first_action_divergence_step", "shield_triggered_steps", "maximum_shield_trigger_distance_m",
    "speed_projection_count", "minimum_planar_separation_m", "minimum_planar_separation_step",
    "initial_operational_margin_m", "minimum_operational_margin_m", "maximum_relative_speed_mps",
    "design_speed_envelope_exceeded", "minimum_uav_margin", "minimum_ugv_margin",
    "minimum_hard_separation_margin", "final_relative_x_m", "final_relative_y_m", "completed_steps",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_candidates(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for value in values:
        item = dict(value)
        item.pop("candidate_id", None)
        output.append(item)
    return output


def validate_design_manifest(
    manifest: dict[str, Any], root: Path
) -> tuple[Path, tuple[SceneCandidate, ...]]:
    required = {
        "stage": "claim_ineligible_gazebo_gz0_v5_factorial_mechanism_development",
        "route_authorized": True,
        "scientific_claim_allowed": False,
        "circa_gz1_authorized": False,
        "retry_allowed": False,
        "seed_top_up_allowed": False,
        "sealed_data_authorized": False,
        "formal_experiment_authorized": False,
        "gpu_count": 0,
        "resource_class": "CPU-SHARED",
        "route": "factorial_speed_projection_by_predictive_shield_mechanism_screen",
        "prior_result_reuse_allowed": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise CircaGazeboGZ0V5Error(f"manifest {key} must equal {expected!r}")
    if tuple(manifest.get("drivers", ())) != DRIVERS:
        raise CircaGazeboGZ0V5Error("driver levels changed")
    if tuple(manifest.get("regimes_per_driver", ())) != CONTROLLERS:
        raise CircaGazeboGZ0V5Error("controller/control levels changed")
    if tuple(manifest.get("active_controllers", ())) != CONTROLLERS[1:]:
        raise CircaGazeboGZ0V5Error("active-controller levels changed")
    frozen_path = (root / manifest["frozen_candidate_source_manifest_path"]).resolve()
    if root not in frozen_path.parents or not frozen_path.is_file():
        raise CircaGazeboGZ0V5Error("frozen GZ0-v4 manifest is unsafe or absent")
    if _sha256(frozen_path) != manifest["frozen_candidate_source_manifest_sha256"]:
        raise CircaGazeboGZ0V5Error("frozen GZ0-v4 manifest lock failed")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    invariant_keys = (
        "families", "candidates_per_family", "seeds_per_candidate", "hazard_active_seed_indices",
        "horizon_steps", "corridor_lateral_goal", "safe_uav_lateral_goal", "safe_ugv_lateral_goal",
        "task_action_limit", "operational_envelope_assumptions", "initial_operational_margin_min_m",
        "design_speed_tolerance_mps", "target_shadow_operational_violation_rate",
    )
    changed = [key for key in invariant_keys if manifest.get(key) != frozen.get(key)]
    if changed or _normalized_candidates(manifest["candidates"]) != _normalized_candidates(frozen["candidates"]):
        raise CircaGazeboGZ0V5Error(f"GZ0-v4 candidate/operational invariance failed: {changed}")
    world = (root / manifest["world_path"]).resolve()
    if root not in world.parents or not world.is_file() or _sha256(world) != manifest["world_sha256"]:
        raise CircaGazeboGZ0V5Error("world source lock failed")
    for relative, expected in manifest.get("source_files", {}).items():
        source = (root / relative).resolve()
        if root not in source.parents or not source.is_file() or _sha256(source) != expected:
            raise CircaGazeboGZ0V5Error(f"source lock failed for {relative}")
    candidates = tuple(SceneCandidate.from_dict(value) for value in manifest["candidates"])
    if len(candidates) != len(manifest["families"]) * int(manifest["candidates_per_family"]):
        raise CircaGazeboGZ0V5Error("candidate grid size changed")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise CircaGazeboGZ0V5Error("candidate IDs must be unique")
    if set(candidate.family_id for candidate in candidates) != set(manifest["families"]):
        raise CircaGazeboGZ0V5Error("candidate family registry changed")
    expected = len(candidates) * int(manifest["seeds_per_candidate"]) * len(DRIVERS) * len(CONTROLLERS)
    if expected != int(manifest["max_regime_rollouts"]):
        raise CircaGazeboGZ0V5Error("factorial rollout budget mismatch")
    shield = manifest["predictive_shield"]
    command_lower_bound = (
        float(manifest["task_action_limit"]) / max(item.uav_mass for item in candidates)
        + float(manifest["task_action_limit"]) / (1.0 + max(item.ugv_friction for item in candidates))
    )
    maximum_vehicle_speed = 0.5 * max(
        max(1.0, 2.0 * item.closing_speed) for item in candidates
    )
    worst_adverse_drag = (
        (
            max(item.uav_drag for item in candidates) * maximum_vehicle_speed
            + UAV_QUADRATIC_DRAG * maximum_vehicle_speed**2
        ) / min(item.uav_mass for item in candidates)
        + UGV_QUADRATIC_DRAG * maximum_vehicle_speed**2
    )
    worst_case_lower_bound = command_lower_bound - worst_adverse_drag
    if not 0.0 < shield["minimum_relative_deceleration_mps2"] <= worst_case_lower_bound:
        raise CircaGazeboGZ0V5Error("shield deceleration is not a conservative actuation lower bound")
    if shield["outward_action_magnitude"] != manifest["task_action_limit"]:
        raise CircaGazeboGZ0V5Error("shield action magnitude must equal the frozen action limit")
    return world, candidates


def validate_runnable_manifest(
    manifest: dict[str, Any], root: Path
) -> tuple[Path, tuple[SceneCandidate, ...]]:
    world, candidates = validate_design_manifest(manifest, root)
    required = {
        "status": "authorized_exactly_once",
        "scientific_run_authorized": True,
        "exactly_once_authorization": True,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise CircaGazeboGZ0V5Error(f"runnable manifest {key} must equal {expected!r}")
    return world, candidates


def _run_regime(
    candidate: SceneCandidate,
    seed: int,
    hazard_active: bool,
    driver_id: str,
    controller_id: str,
    world: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if driver_id not in DRIVERS or controller_id not in CONTROLLERS:
        raise CircaGazeboGZ0V5Error("unknown factorial level")
    horizon = int(manifest["horizon_steps"])
    envelope = derive_operational_envelope(candidate, manifest["operational_envelope_assumptions"])
    design_speed = envelope.design_relative_speed_mps
    if driver_id == "planar_speed_projected_v4":
        env = GazeboAirGroundEnvV4(
            world,
            candidate.shift(),
            horizon=horizon,
            uav_planar_speed_limit_mps=0.5 * design_speed,
            ugv_planar_speed_limit_mps=0.5 * design_speed,
        )
    else:
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
    shield = PredictiveSeparationShield(
        minimum_relative_deceleration_mps2=float(
            manifest["predictive_shield"]["minimum_relative_deceleration_mps2"]
        ),
        action_limit=action_limit,
    )
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
    applied_intervention = applied_intervention_steps = counterfactual_diverged = 0
    shield_triggered_steps = 0
    maximum_shield_trigger_distance = 0.0
    for step in range(horizon):
        index = max(0, len(states) - 1 - candidate.observation_delay_steps)
        observed = states[index].copy()
        observed[[0, 1, 2, 6, 7]] += candidate.sensor_bias
        augmented = np.concatenate([observed, applied[index]])
        nominal = task.nominal_action(augmented)
        if controller_id == "shadow_no_override":
            action = nominal
            changed = False
        elif controller_id == "predictive_separation_shield":
            decision = shield.decide(
                observed,
                nominal,
                operational_separation_m=envelope.operational_separation_m,
                reaction_time_s=envelope.reaction_time_s,
            )
            action = decision.action
            changed = not np.allclose(action, nominal)
            shield_triggered_steps += int(decision.intervened)
            maximum_shield_trigger_distance = max(
                maximum_shield_trigger_distance, decision.trigger_distance_m
            )
        else:
            true_augmented = np.concatenate([env.state, env._applied_action])
            backup = np.clip(
                backup_gain @ (true_augmented - backup_center),
                task.action_lower,
                task.action_upper,
            )
            try:
                filtered = cbf.decide(augmented, fallback_action=backup).action
            except (SandboxBaselineInfeasible, ValueError):
                filtered = backup.copy()
            age = (candidate.observation_delay_steps + candidate.communication_delay_steps) * env.dt + 0.04
            predicted = predictor.predict(
                observed, filtered, previous_applied_action=applied[index], step_index=step
            )
            duration = max(0.0, predicted - 0.30 - age - 0.05)
            bundle = _assurance_bundle(
                filtered,
                step * env.dt,
                duration,
                f"circa-gz0-v5-{driver_id}-{candidate.candidate_id}-{seed}-{step}",
            )
            verification = verify_assurance_case(bundle)
            action = filtered if duration > 0.0 and verification.accepted else backup
            changed = not np.allclose(action, nominal)
        if changed and divergence_step is None:
            divergence_step = step + 1
        if controller_id != "shadow_no_override":
            applied_intervention |= int(changed)
            applied_intervention_steps += int(changed)
            counterfactual_diverged |= int(changed)
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
    row = {
        "candidate_id": candidate.candidate_id,
        "family_id": candidate.family_id,
        "scenario_seed": seed,
        "hazard_active": hazard_active,
        "driver_id": driver_id,
        "controller_id": controller_id,
        "operational_separation_m": envelope.operational_separation_m,
        "design_relative_speed_mps": design_speed,
        "operational_first_violation": int(operational_step is not None),
        "operational_first_violation_step": operational_step,
        "hard_first_violation": int(hard_step is not None),
        "hard_first_violation_step": hard_step,
        "applied_intervention": int(applied_intervention),
        "applied_intervention_steps": applied_intervention_steps,
        "counterfactual_action_diverged": int(counterfactual_diverged),
        "first_action_divergence_step": divergence_step,
        "shield_triggered_steps": shield_triggered_steps,
        "maximum_shield_trigger_distance_m": maximum_shield_trigger_distance,
        "speed_projection_count": int(getattr(env, "speed_projection_count", 0)),
        "minimum_planar_separation_m": min_sep,
        "minimum_planar_separation_step": min_sep_step,
        "initial_operational_margin_m": initial_operational,
        "minimum_operational_margin_m": min_operational,
        "maximum_relative_speed_mps": max_relative_speed,
        "design_speed_envelope_exceeded": bool(
            max_relative_speed > design_speed + manifest["design_speed_tolerance_mps"]
        ),
        "minimum_uav_margin": min_uav,
        "minimum_ugv_margin": min_ugv,
        "minimum_hard_separation_margin": min_hard,
        "final_relative_x_m": float(relative[0]),
        "final_relative_y_m": float(relative[1]),
        "completed_steps": env.step_index,
    }
    if set(row) != set(TRACE_FIELDS):
        raise CircaGazeboGZ0V5Error("trace schema drifted from the frozen field registry")
    return row


def _rate(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        raise CircaGazeboGZ0V5Error("empty factorial cell")
    return float(np.mean([row[field] for row in rows]))


def _diagnostic(
    rows: list[dict[str, Any]], families: tuple[str, ...], manifest: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    gates = manifest["development_gates"]
    manipulation = {}
    for driver in DRIVERS:
        subset = [row for row in rows if row["driver_id"] == driver]
        manipulation[driver] = {
            "row_count": len(subset),
            "speed_envelope_exceedance_rate": _rate(subset, "design_speed_envelope_exceeded"),
        }
    manipulation["pass"] = bool(
        manipulation["command_persistent_unbounded_v3"]["speed_envelope_exceedance_rate"]
        >= gates["unbounded_driver_speed_exceedance_rate_min"]
        and manipulation["planar_speed_projected_v4"]["speed_envelope_exceedance_rate"]
        <= gates["projected_driver_speed_exceedance_rate_max"]
    )
    output: dict[str, Any] = {}
    for family in families:
        candidates = []
        for candidate_id in sorted({row["candidate_id"] for row in rows if row["family_id"] == family}):
            candidate_rows = [row for row in rows if row["candidate_id"] == candidate_id]
            unbounded = [row for row in candidate_rows if row["driver_id"] == "command_persistent_unbounded_v3"]
            unbounded_shadow = [row for row in unbounded if row["controller_id"] == "shadow_no_override"]
            unbounded_shield = [row for row in unbounded if row["controller_id"] == "predictive_separation_shield"]
            unbounded_cbf = [row for row in unbounded if row["controller_id"] == "registered_one_step_cbf"]
            projected = [row for row in candidate_rows if row["driver_id"] == "planar_speed_projected_v4"]
            shadow = [row for row in projected if row["controller_id"] == "shadow_no_override"]
            shield = [row for row in projected if row["controller_id"] == "predictive_separation_shield"]
            cbf = [row for row in projected if row["controller_id"] == "registered_one_step_cbf"]
            item = {
                "candidate_id": candidate_id,
                "unbounded_shadow_operational_violation_rate": _rate(unbounded_shadow, "operational_first_violation"),
                "unbounded_predictive_shield_operational_violation_rate": _rate(unbounded_shield, "operational_first_violation"),
                "unbounded_registered_cbf_operational_violation_rate": _rate(unbounded_cbf, "operational_first_violation"),
                "projected_shadow_operational_violation_rate": _rate(shadow, "operational_first_violation"),
                "projected_shadow_hard_violation_rate": _rate(shadow, "hard_first_violation"),
                "predictive_shield_operational_violation_rate": _rate(shield, "operational_first_violation"),
                "predictive_shield_hard_violation_rate": _rate(shield, "hard_first_violation"),
                "predictive_shield_intervention_rate": _rate(shield, "applied_intervention"),
                "registered_cbf_operational_violation_rate": _rate(cbf, "operational_first_violation"),
                "registered_cbf_hard_violation_rate": _rate(cbf, "hard_first_violation"),
                "projected_driver_speed_exceedance_rate": _rate(projected, "design_speed_envelope_exceeded"),
            }
            item["predictive_shield_paired_risk_reduction"] = (
                item["projected_shadow_operational_violation_rate"]
                - item["predictive_shield_operational_violation_rate"]
            )
            item["unbounded_predictive_shield_paired_risk_reduction"] = (
                item["unbounded_shadow_operational_violation_rate"]
                - item["unbounded_predictive_shield_operational_violation_rate"]
            )
            item["driver_by_predictive_shield_interaction"] = (
                item["predictive_shield_paired_risk_reduction"]
                - item["unbounded_predictive_shield_paired_risk_reduction"]
            )
            item["unbounded_registered_cbf_paired_risk_reduction"] = (
                item["unbounded_shadow_operational_violation_rate"]
                - item["unbounded_registered_cbf_operational_violation_rate"]
            )
            item["projected_registered_cbf_paired_risk_reduction"] = (
                item["projected_shadow_operational_violation_rate"]
                - item["registered_cbf_operational_violation_rate"]
            )
            item["predictive_shield_hard_risk_increase"] = (
                item["predictive_shield_hard_violation_rate"]
                - item["projected_shadow_hard_violation_rate"]
            )
            item["gate_pass"] = bool(
                gates["projected_shadow_operational_violation_rate_min"]
                <= item["projected_shadow_operational_violation_rate"]
                <= gates["projected_shadow_operational_violation_rate_max"]
                and item["predictive_shield_intervention_rate"]
                >= gates["predictive_shield_intervention_rate_min"]
                and item["predictive_shield_operational_violation_rate"]
                <= gates["predictive_shield_operational_violation_rate_max"]
                and item["predictive_shield_paired_risk_reduction"]
                >= gates["predictive_shield_paired_risk_reduction_min"]
                and item["predictive_shield_hard_risk_increase"]
                <= gates["predictive_shield_hard_risk_increase_max"]
                and item["projected_driver_speed_exceedance_rate"]
                <= gates["projected_driver_speed_exceedance_rate_max"]
            )
            candidates.append(item)
        passing = [item for item in candidates if item["gate_pass"]]
        output[family] = {
            "all_candidates": candidates,
            "passing_candidate_count": len(passing),
            "selected": sorted(
                passing,
                key=lambda item: (
                    abs(item["projected_shadow_operational_violation_rate"] - manifest["target_shadow_operational_violation_rate"]),
                    -item["predictive_shield_paired_risk_reduction"],
                    item["predictive_shield_operational_violation_rate"],
                    item["candidate_id"],
                ),
            )[0] if passing else None,
        }
    return manipulation, output


def run_gz0_v5(manifest_path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest_raw = manifest_file.read_bytes()
    manifest = json.loads(manifest_raw)
    world, candidates = validate_runnable_manifest(manifest, root)
    output = (root / manifest["output_path"]).resolve()
    if root not in output.parents or output.exists():
        raise CircaGazeboGZ0V5Error("exactly-once output path already exists or is unsafe")
    hazard_indices = {int(value) for value in manifest["hazard_active_seed_indices"]}
    schedule = []
    for candidate in candidates:
        for index in range(int(manifest["seeds_per_candidate"])):
            digest = hashlib.sha256(
                f"circa-gz0-v5|{manifest['master_seed']}|{candidate.candidate_id}|{index}".encode()
            ).digest()
            seed = int.from_bytes(digest[:8], "big") % (2**31 - 1)
            for driver in DRIVERS:
                for controller in CONTROLLERS:
                    schedule.append((candidate, seed, index in hazard_indices, driver, controller))
    if len(schedule) != int(manifest["max_regime_rollouts"]):
        raise CircaGazeboGZ0V5Error("generated schedule does not match frozen budget")
    random.Random(int(manifest["schedule_seed"])).shuffle(schedule)
    start = time.perf_counter()
    rows = []
    for candidate, seed, hazard_active, driver, controller in schedule:
        if time.perf_counter() - start > manifest["wall_time_seconds_max"]:
            raise CircaGazeboGZ0V5Error("GZ0-v5 wall-time budget exceeded")
        rows.append(
            _run_regime(candidate, seed, hazard_active, driver, controller, world, manifest)
        )
    elapsed = time.perf_counter() - start
    manipulation, diagnostic = _diagnostic(rows, tuple(manifest["families"]), manifest)
    all_families = all(value["passing_candidate_count"] > 0 for value in diagnostic.values())
    result = {
        "result_id": manifest["manifest_id"],
        "claim_eligible": False,
        "circa_gz1_run": False,
        "sealed_data_used": False,
        "formal_experiment_run": False,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "world_sha256": manifest["world_sha256"],
        "rows": rows,
        "factorial_manipulation": manipulation,
        "development_diagnostic": diagnostic,
        "all_families_have_passing_candidate": all_families,
        "route_gate_pass": bool(manipulation["pass"] and all_families),
        "elapsed_seconds": elapsed,
        "boundary": "claim-ineligible GZ0-v5 factorial mechanism development only; no GZ1 authorization",
    }
    body = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if len(body) > manifest["output_bytes_max"]:
        raise CircaGazeboGZ0V5Error("GZ0-v5 output budget exceeded")
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_bytes(body)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    result = run_gz0_v5(args.manifest, args.repo_root)
    print(json.dumps({
        "result_id": result["result_id"],
        "rows": len(result["rows"]),
        "elapsed_seconds": result["elapsed_seconds"],
        "factorial_manipulation_pass": result["factorial_manipulation"]["pass"],
        "all_families_have_passing_candidate": result["all_families_have_passing_candidate"],
        "route_gate_pass": result["route_gate_pass"],
        "claim_eligible": False,
        "circa_gz1_run": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
