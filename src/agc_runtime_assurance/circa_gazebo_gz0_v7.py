"""Design-only validation kernel for the frozen CIRCA-GZ0-v7 Route A.

There is deliberately no scientific runner in this module.  It validates the
role construction, unchanged v6 endpoints/gates, the minimal viability-based
half-gap adjustment, and the robust-backup fixture.  The CLI prints a transient
non-evidence audit to stdout and never creates an output directory.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .circa_gazebo_gz0 import SceneCandidate
from .circa_gazebo_gz0_v3 import derive_operational_envelope
from .gazebo_role_valid_task import build_role_valid_scenario
from .gazebo_robust_backup_filter import (
    GazeboPlanarPlant,
    RobustBackupConfig,
    RobustBackupSafetyFilter,
    propagate_planar_state,
)


class CircaGazeboGZ0V7Error(RuntimeError):
    pass


DRIVERS = ("command_persistent_unbounded_v3", "planar_speed_projected_v4")
CONTROLLERS = ("shadow_no_override", "registered_one_step_cbf", "robust_backup_filter_v7")
ALLOWED_HALF_GAP_GRID_M = 0.05


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _backup_filter(
    candidate: SceneCandidate, manifest: dict[str, Any]
) -> tuple[RobustBackupSafetyFilter, float]:
    envelope = derive_operational_envelope(candidate, manifest["operational_envelope_assumptions"])
    parameters = manifest["robust_backup_filter"]
    plant = GazeboPlanarPlant(
        uav_mass=candidate.uav_mass,
        uav_drag=candidate.uav_drag,
        ugv_friction=candidate.ugv_friction,
        actuator_lag=candidate.actuator_lag,
        dt=float(parameters["sample_period_s"]),
        speed_limit_mps=0.5 * envelope.design_relative_speed_mps,
    )
    config = RobustBackupConfig(
        operational_separation_m=envelope.operational_separation_m,
        action_limit=float(manifest["task_action_limit"]),
        horizon_steps=int(parameters["backup_horizon_steps"]),
        terminal_margin_m=float(parameters["terminal_margin_m"]),
        position_error_bound_m=float(parameters["position_error_bound_m"]),
        velocity_error_bound_mps=float(parameters["velocity_error_bound_mps"]),
        relative_acceleration_error_bound_mps2=float(
            parameters["relative_acceleration_error_bound_mps2"]
        ),
        barrier_retention=float(parameters["barrier_retention"]),
    )
    return RobustBackupSafetyFilter(plant, config), envelope.operational_separation_m


def _scenario(
    candidate: SceneCandidate, manifest: dict[str, Any], *, hazard_active: bool
):
    envelope = derive_operational_envelope(candidate, manifest["operational_envelope_assumptions"])
    role = manifest["role_validity"]
    return build_role_valid_scenario(
        candidate,
        hazard_active=hazard_active,
        operational_separation_m=envelope.operational_separation_m,
        action_limit=float(manifest["task_action_limit"]),
        corridor_lateral_goal=float(manifest["corridor_lateral_goal"]),
        control_uav_lateral_goal=float(manifest["safe_uav_lateral_goal"]),
        control_ugv_lateral_goal=float(manifest["safe_ugv_lateral_goal"]),
        longitudinal_goal_magnitude=float(role["longitudinal_goal_magnitude_m"]),
    )


def evaluate_reference_role(
    candidate: SceneCandidate, manifest: dict[str, Any], *, hazard_active: bool
) -> dict[str, Any]:
    """Run a deterministic reference-model fixture, not a Gazebo experiment."""

    scenario = _scenario(candidate, manifest, hazard_active=hazard_active)
    envelope = derive_operational_envelope(candidate, manifest["operational_envelope_assumptions"])
    plant = GazeboPlanarPlant(
        candidate.uav_mass,
        candidate.uav_drag,
        candidate.ugv_friction,
        candidate.actuator_lag,
        dt=float(manifest["robust_backup_filter"]["sample_period_s"]),
        speed_limit_mps=0.5 * envelope.design_relative_speed_mps,
    )
    state = scenario.initial_state.copy()
    applied = np.zeros(5, dtype=float)
    state_history = [state.copy()]
    applied_history = [applied.copy()]
    minimum_separation = float(np.linalg.norm(state[:2] - state[6:8]))
    first_violation_step = None
    for step in range(int(manifest["horizon_steps"])):
        index = max(0, len(state_history) - 1 - candidate.observation_delay_steps)
        observed = state_history[index].copy()
        observed[[0, 1, 2, 6, 7]] += candidate.sensor_bias
        nominal = scenario.task.nominal_action(
            np.concatenate([observed, applied_history[index]])
        )
        state, applied = propagate_planar_state(state, applied, nominal, plant)
        state_history.append(state.copy())
        applied_history.append(applied.copy())
        separation = float(np.linalg.norm(state[:2] - state[6:8]))
        minimum_separation = min(minimum_separation, separation)
        if separation < envelope.operational_separation_m and first_violation_step is None:
            first_violation_step = step + 1
    return {
        "role_id": scenario.role_id,
        "initial_radial_separation_rate_mps": scenario.initial_radial_separation_rate_mps,
        "minimum_separation_m": minimum_separation,
        "operational_separation_m": envelope.operational_separation_m,
        "operational_violation": first_violation_step is not None,
        "first_violation_step": first_violation_step,
    }


def evaluate_initial_backup(candidate: SceneCandidate, manifest: dict[str, Any]) -> dict[str, Any]:
    scenario = _scenario(candidate, manifest, hazard_active=True)
    safety_filter, _ = _backup_filter(candidate, manifest)
    action = safety_filter.backup_action(scenario.initial_state)
    plan = safety_filter.evaluate_plan(scenario.initial_state, np.zeros(5), action)
    return {
        "feasible": plan.feasible,
        "terminal_reached": plan.terminal_reached,
        "terminal_step": plan.terminal_step,
        "minimum_tightened_margin_m": plan.minimum_tightened_margin_m,
        "minimum_barrier_residual_m": plan.minimum_barrier_residual_m,
        "reason": plan.reason,
    }


def _grid_aligned(value: float) -> bool:
    return bool(abs(value / ALLOWED_HALF_GAP_GRID_M - round(value / ALLOWED_HALF_GAP_GRID_M)) <= 1e-9)


def validate_design_manifest(
    manifest: dict[str, Any], root: Path
) -> tuple[tuple[SceneCandidate, ...], dict[str, Any]]:
    required = {
        "stage": "claim_ineligible_gazebo_gz0_v7_role_valid_robust_backup_design",
        "status": "frozen_route_local_implementation_only_nonrunnable",
        "route_authorized": True,
        "local_implementation_authorized": True,
        "scientific_run_authorized": False,
        "exactly_once_authorization": False,
        "scientific_claim_allowed": False,
        "circa_gz1_authorized": False,
        "retry_allowed": False,
        "seed_top_up_allowed": False,
        "seed_material_generated": False,
        "sealed_data_authorized": False,
        "formal_experiment_authorized": False,
        "gpu_count": 0,
        "resource_class": "CPU-SHARED",
        "route": "role_valid_negative_control_by_robust_backup_filter",
        "prior_result_reuse_allowed": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise CircaGazeboGZ0V7Error(f"manifest {key} must equal {expected!r}")
    forbidden_material = ("master_seed", "schedule_seed", "output_path", "admission_path")
    if any(key in manifest for key in forbidden_material):
        raise CircaGazeboGZ0V7Error("scientific seed/output material must not exist in design-only v7")
    if tuple(manifest.get("drivers", ())) != DRIVERS or tuple(
        manifest.get("regimes_per_driver", ())
    ) != CONTROLLERS:
        raise CircaGazeboGZ0V7Error("v7 factorial levels drifted")
    for relative, expected in manifest.get("source_files", {}).items():
        source = (root / relative).resolve()
        if root not in source.parents or not source.is_file() or _sha256(source) != expected:
            raise CircaGazeboGZ0V7Error(f"v7 source lock failed for {relative}")
    for relative, expected in manifest.get("protected_files", {}).items():
        protected = (root / relative).resolve()
        if root not in protected.parents or not protected.is_file() or _sha256(protected) != expected:
            raise CircaGazeboGZ0V7Error(f"protected evidence lock failed for {relative}")
    if not manifest.get("source_files") or not manifest.get("protected_files"):
        raise CircaGazeboGZ0V7Error("source and protected-evidence locks must be nonempty")

    prior_path = (root / manifest["v6_design_manifest_path"]).resolve()
    if root not in prior_path.parents or not prior_path.is_file():
        raise CircaGazeboGZ0V7Error("v6 design manifest is unsafe or absent")
    if _sha256(prior_path) != manifest["v6_design_manifest_sha256"]:
        raise CircaGazeboGZ0V7Error("v6 design source lock failed")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    unchanged = (
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
    drift = [key for key in unchanged if manifest.get(key) != prior.get(key)]
    if drift:
        raise CircaGazeboGZ0V7Error(f"v6 endpoint/gate payload drifted: {drift}")

    prior_by_family = {
        (item["family_id"], item["candidate_id"][-1]): item for item in prior["candidates"]
    }
    candidates = tuple(SceneCandidate.from_dict(value) for value in manifest["candidates"])
    if len(candidates) != 12 or len({item.candidate_id for item in candidates}) != 12:
        raise CircaGazeboGZ0V7Error("v7 must contain 12 unique candidates")
    if set(item.family_id for item in candidates) != set(manifest["families"]):
        raise CircaGazeboGZ0V7Error("v7 family registry drifted")

    role_audit: dict[str, Any] = {}
    reserve = float(manifest["robust_backup_filter"]["initial_viability_reserve_m"])
    for candidate in candidates:
        suffix = candidate.candidate_id[-1]
        key = (candidate.family_id, suffix)
        if key not in prior_by_family or f"{candidate.family_id}-V7{suffix}" != candidate.candidate_id:
            raise CircaGazeboGZ0V7Error(f"candidate identity drift for {candidate.candidate_id}")
        old = dict(prior_by_family[key])
        new = next(item for item in manifest["candidates"] if item["candidate_id"] == candidate.candidate_id)
        for field in old:
            if field not in {"candidate_id", "half_gap"} and new.get(field) != old[field]:
                raise CircaGazeboGZ0V7Error(f"candidate field {field} changed for {candidate.candidate_id}")
        if candidate.half_gap < float(old["half_gap"]) or not _grid_aligned(candidate.half_gap):
            raise CircaGazeboGZ0V7Error(f"half-gap rule failed for {candidate.candidate_id}")

        hazard = evaluate_reference_role(candidate, manifest, hazard_active=True)
        control = evaluate_reference_role(candidate, manifest, hazard_active=False)
        backup = evaluate_initial_backup(candidate, manifest)
        if not hazard["operational_violation"] or control["operational_violation"]:
            raise CircaGazeboGZ0V7Error(f"role-validity fixture failed for {candidate.candidate_id}")
        if not backup["feasible"] or backup["minimum_tightened_margin_m"] < reserve - 1e-9:
            raise CircaGazeboGZ0V7Error(f"initial backup viability failed for {candidate.candidate_id}")

        previous_gap = candidate.half_gap - ALLOWED_HALF_GAP_GRID_M
        if previous_gap >= float(old["half_gap"]) - 1e-9:
            previous = replace(candidate, half_gap=previous_gap)
            prior_backup = evaluate_initial_backup(previous, manifest)
            if prior_backup["feasible"] and prior_backup["minimum_tightened_margin_m"] >= reserve - 1e-9:
                raise CircaGazeboGZ0V7Error(f"half-gap is not the minimal registered grid point for {candidate.candidate_id}")
        role_audit[candidate.candidate_id] = {
            "hazard": hazard,
            "negative_control": control,
            "initial_backup": backup,
            "v6_half_gap_m": float(old["half_gap"]),
            "v7_half_gap_m": candidate.half_gap,
        }

    predicted_rate = len(manifest["hazard_active_seed_indices"]) / int(
        manifest["seeds_per_candidate"]
    )
    lower = manifest["development_gates"]["projected_shadow_operational_violation_rate_min"]
    upper = manifest["development_gates"]["projected_shadow_operational_violation_rate_max"]
    if not lower <= predicted_rate <= upper or predicted_rate != manifest["target_shadow_operational_violation_rate"]:
        raise CircaGazeboGZ0V7Error("role allocation does not satisfy the unchanged event-rate gate")
    expected = len(candidates) * int(manifest["seeds_per_candidate"]) * len(DRIVERS) * len(CONTROLLERS)
    if expected != 360 or expected != int(manifest["max_regime_rollouts"]):
        raise CircaGazeboGZ0V7Error("v7 design size drifted")
    return candidates, role_audit


def audit_design(manifest_path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    path = Path(manifest_path).resolve()
    if root not in path.parents or not path.is_file():
        raise CircaGazeboGZ0V7Error("manifest path is unsafe or absent")
    raw = path.read_bytes()
    manifest = json.loads(raw)
    candidates, role_audit = validate_design_manifest(manifest, root)
    return {
        "audit_type": "local_non_evidence_design_fixture",
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "candidate_count": len(candidates),
        "reference_role_pairs_passed": len(role_audit),
        "initial_backup_corners_passed": sum(
            int(item["initial_backup"]["feasible"]) for item in role_audit.values()
        ),
        "predicted_shadow_role_rate": len(manifest["hazard_active_seed_indices"])
        / int(manifest["seeds_per_candidate"]),
        "scientific_run_authorized": False,
        "scientific_output_generated": False,
        "circa_gz1_authorized": False,
        "role_audit": role_audit,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate the nonrunnable GZ0-v7 design only")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    print(json.dumps(audit_design(args.manifest, args.repo_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
