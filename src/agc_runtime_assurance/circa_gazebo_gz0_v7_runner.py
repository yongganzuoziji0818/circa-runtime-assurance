"""Exactly-once runner for claim-ineligible CIRCA Gazebo GZ0-v7 Route A."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from .assurance_case import verify_assurance_case
from .circa_gazebo_gz0 import SceneCandidate
from .circa_gazebo_gz0_v3 import derive_operational_envelope
from .circa_gazebo_gz0_v7 import (
    _backup_filter,
    _scenario,
    validate_design_manifest as validate_design_only_manifest,
)
from .dynamic_l0 import _assurance_bundle
from .environment import CompoundShift
from .gazebo_second_system import constraint_margins
from .gazebo_second_system_v3 import GazeboAirGroundEnvV3
from .gazebo_second_system_v4 import GazeboAirGroundEnvV4
from .predictor import NominalRolloutHorizonPredictor
from .sandbox_baselines import SandboxBaselineInfeasible, SandboxNominalCBFAdapter
from .sandbox_fallback import sandbox_backup_equilibrium, sandbox_backup_gain


class CircaGazeboGZ0V7RunnerError(RuntimeError):
    pass


SEED_NAMESPACE = "circa-gz0-v7"
DRIVERS = ("command_persistent_unbounded_v3", "planar_speed_projected_v4")
CONTROLLERS = ("shadow_no_override", "registered_one_step_cbf", "robust_backup_filter_v7")
TRACE_FIELDS = (
    "candidate_id", "family_id", "scenario_seed", "hazard_active", "role_id",
    "initial_radial_separation_rate_mps", "driver_id", "controller_id",
    "operational_separation_m", "design_relative_speed_mps",
    "operational_first_violation", "operational_first_violation_step",
    "hard_first_violation", "hard_first_violation_step", "applied_intervention",
    "applied_intervention_steps", "counterfactual_action_diverged",
    "first_action_divergence_step", "robust_filter_triggered_steps",
    "robust_filter_fail_closed_steps", "robust_filter_reason_counts",
    "minimum_nominal_tube_margin_m", "minimum_backup_tube_margin_m",
    "speed_projection_count", "minimum_planar_separation_m",
    "minimum_planar_separation_step", "initial_operational_margin_m",
    "minimum_operational_margin_m", "minimum_absolute_operational_margin_m",
    "maximum_relative_speed_mps", "design_speed_envelope_exceeded",
    "minimum_uav_margin", "minimum_ugv_margin", "minimum_hard_separation_margin",
    "minimum_absolute_hard_margin", "final_relative_x_m", "final_relative_y_m",
    "completed_steps",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root not in path.parents:
        raise CircaGazeboGZ0V7RunnerError(f"unsafe path outside repository: {relative}")
    return path


def scenario_seeds(manifest: dict[str, Any]) -> tuple[int, ...]:
    seeds: list[int] = []
    for candidate in manifest["candidates"]:
        for index in range(int(manifest["seeds_per_candidate"])):
            digest = hashlib.sha256(
                f"{SEED_NAMESPACE}|{manifest['master_seed']}|{candidate['candidate_id']}|{index}".encode()
            ).digest()
            seeds.append(int.from_bytes(digest[:8], "big") % (2**31 - 1))
    return tuple(seeds)


def validate_design_manifest(
    manifest: dict[str, Any], root: Path
) -> tuple[Path, tuple[SceneCandidate, ...]]:
    required = {
        "stage": "claim_ineligible_gazebo_gz0_v7_role_valid_robust_backup_exactly_once",
        "route_authorized": True,
        "scientific_claim_allowed": False,
        "circa_gz1_authorized": False,
        "retry_allowed": False,
        "seed_top_up_allowed": False,
        "seed_material_generated": True,
        "sealed_data_authorized": False,
        "formal_experiment_authorized": False,
        "gpu_count": 0,
        "resource_class": "CPU-SHARED",
        "route": "role_valid_negative_control_by_robust_backup_filter",
        "prior_result_reuse_allowed": False,
        "seed_namespace": SEED_NAMESPACE,
        "legacy_v2_fixture_disposition": "retained_out_of_route_failure_no_rerun",
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise CircaGazeboGZ0V7RunnerError(f"manifest {key} must equal {expected!r}")
    if tuple(manifest.get("drivers", ())) != DRIVERS:
        raise CircaGazeboGZ0V7RunnerError("v7 driver levels drifted")
    if tuple(manifest.get("regimes_per_driver", ())) != CONTROLLERS:
        raise CircaGazeboGZ0V7RunnerError("v7 controller levels drifted")
    if tuple(manifest.get("active_controllers", ())) != CONTROLLERS[1:]:
        raise CircaGazeboGZ0V7RunnerError("v7 active-controller levels drifted")

    design_path = _safe_child(root, manifest["design_source_manifest_path"])
    if not design_path.is_file() or _sha256(design_path) != manifest["design_source_manifest_sha256"]:
        raise CircaGazeboGZ0V7RunnerError("v7 design-only manifest lock failed")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    design_candidates, _ = validate_design_only_manifest(design, root)
    invariant = (
        "drivers", "active_controllers", "regimes_per_driver", "role_validity",
        "robust_backup_filter", "families", "candidates_per_family", "seeds_per_candidate",
        "hazard_active_seed_indices", "horizon_steps", "corridor_lateral_goal",
        "safe_uav_lateral_goal", "safe_ugv_lateral_goal", "task_action_limit",
        "operational_envelope_assumptions", "initial_operational_margin_min_m",
        "design_speed_tolerance_mps", "target_shadow_operational_violation_rate",
        "development_gates", "gate_name_aliases", "max_regime_rollouts",
        "wall_time_seconds_max", "output_bytes_max", "candidates",
    )
    drift = [key for key in invariant if manifest.get(key) != design.get(key)]
    if drift:
        raise CircaGazeboGZ0V7RunnerError(f"v7 frozen scientific payload drifted: {drift}")

    world = _safe_child(root, manifest["world_path"])
    if not world.is_file() or _sha256(world) != manifest["world_sha256"]:
        raise CircaGazeboGZ0V7RunnerError("Gazebo world source lock failed")
    for group in ("source_files", "protected_files"):
        values = manifest.get(group, {})
        if not values:
            raise CircaGazeboGZ0V7RunnerError(f"{group} must be nonempty")
        for relative, expected in values.items():
            path = _safe_child(root, relative)
            if not path.is_file() or _sha256(path) != expected:
                raise CircaGazeboGZ0V7RunnerError(f"{group} lock failed for {relative}")
    receipt = _safe_child(root, manifest["remote_preflight_receipt_path"])
    if not receipt.is_file() or _sha256(receipt) != manifest["remote_preflight_receipt_sha256"]:
        raise CircaGazeboGZ0V7RunnerError("remote preflight receipt lock failed")

    candidates = tuple(SceneCandidate.from_dict(item) for item in manifest["candidates"])
    if tuple(item.candidate_id for item in candidates) != tuple(
        item.candidate_id for item in design_candidates
    ):
        raise CircaGazeboGZ0V7RunnerError("candidate identity/order drifted")
    seeds = scenario_seeds(manifest)
    if len(seeds) != 60 or len(set(seeds)) != 60:
        raise CircaGazeboGZ0V7RunnerError("v7 seed block must contain 60 unique seeds")
    expected = len(candidates) * int(manifest["seeds_per_candidate"]) * len(DRIVERS) * len(CONTROLLERS)
    if expected != 360 or expected != int(manifest["max_regime_rollouts"]):
        raise CircaGazeboGZ0V7RunnerError("v7 factorial schedule size drifted")
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
            raise CircaGazeboGZ0V7RunnerError(f"runnable manifest {key} must equal {expected!r}")
    return world, candidates


def _finite_min(current: float | None, value: float) -> float:
    if not np.isfinite(value):
        return current if current is not None else 0.0
    return value if current is None else min(current, value)


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
        raise CircaGazeboGZ0V7RunnerError("unknown v7 factorial level")
    envelope = derive_operational_envelope(candidate, manifest["operational_envelope_assumptions"])
    design_speed = envelope.design_relative_speed_mps
    horizon = int(manifest["horizon_steps"])
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
    scenario = _scenario(candidate, manifest, hazard_active=hazard_active)
    env.reset(
        seed=seed,
        initial_state=scenario.initial_state,
        position_jitter_scale=candidate.position_jitter_scale,
    )
    task = scenario.task
    cbf = SandboxNominalCBFAdapter(task)
    safety_filter, _ = _backup_filter(candidate, manifest)
    backup_center, backup_gain = sandbox_backup_equilibrium(), sandbox_backup_gain()
    predictor = NominalRolloutHorizonPredictor(CompoundShift(), max_steps=40)

    states = [env.state.copy()]
    applied = [env._applied_action.copy()]
    initial_separation = float(np.linalg.norm(env.state[:2] - env.state[6:8]))
    min_sep, min_sep_step = initial_separation, 0
    initial_operational = initial_separation - envelope.operational_separation_m
    min_operational = initial_operational
    min_abs_operational = abs(initial_operational)
    max_relative_speed = float(np.linalg.norm(env.state[3:5] - env.state[8:10]))
    margins0 = constraint_margins(env.state, 0)
    min_uav, min_ugv, min_hard = margins0.uav_local, margins0.ugv_local, margins0.coupling
    min_abs_hard = abs(margins0.coupling)
    operational_step = hard_step = divergence_step = None
    applied_intervention = applied_intervention_steps = counterfactual_diverged = 0
    robust_triggered = robust_fail_closed = 0
    robust_reasons: Counter[str] = Counter()
    min_nominal_tube: float | None = None
    min_backup_tube: float | None = None

    for step in range(horizon):
        index = max(0, len(states) - 1 - candidate.observation_delay_steps)
        observed = states[index].copy()
        observed[[0, 1, 2, 6, 7]] += candidate.sensor_bias
        augmented = np.concatenate([observed, applied[index]])
        nominal = task.nominal_action(augmented)
        if controller_id == "shadow_no_override":
            action = nominal
            changed = False
        elif controller_id == "robust_backup_filter_v7":
            decision = safety_filter.decide(observed, applied[index], nominal)
            action = decision.action
            changed = not np.allclose(action, nominal)
            robust_triggered += int(decision.intervened)
            robust_fail_closed += int(decision.fail_closed)
            robust_reasons[decision.reason] += 1
            min_nominal_tube = _finite_min(
                min_nominal_tube, decision.nominal_plan.minimum_tightened_margin_m
            )
            min_backup_tube = _finite_min(
                min_backup_tube, decision.backup_plan.minimum_tightened_margin_m
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
            age = (
                candidate.observation_delay_steps + candidate.communication_delay_steps
            ) * env.dt + float(manifest["operational_envelope_assumptions"]["compute_dispatch_budget_s"])
            predicted = predictor.predict(
                observed,
                filtered,
                previous_applied_action=applied[index],
                step_index=step,
            )
            duration = max(0.0, predicted - 0.30 - age - 0.05)
            bundle = _assurance_bundle(
                filtered,
                step * env.dt,
                duration,
                f"circa-gz0-v7-{driver_id}-{candidate.candidate_id}-{seed}-{step}",
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
        operational_margin = separation - envelope.operational_separation_m
        if separation < min_sep:
            min_sep, min_sep_step = separation, step + 1
        min_operational = min(min_operational, operational_margin)
        min_abs_operational = min(min_abs_operational, abs(operational_margin))
        max_relative_speed = max(max_relative_speed, relative_speed)
        min_uav = min(min_uav, current.uav_local)
        min_ugv = min(min_ugv, current.ugv_local)
        min_hard = min(min_hard, current.coupling)
        min_abs_hard = min(min_abs_hard, abs(current.coupling))
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
        "role_id": scenario.role_id,
        "initial_radial_separation_rate_mps": scenario.initial_radial_separation_rate_mps,
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
        "robust_filter_triggered_steps": robust_triggered,
        "robust_filter_fail_closed_steps": robust_fail_closed,
        "robust_filter_reason_counts": dict(sorted(robust_reasons.items())),
        "minimum_nominal_tube_margin_m": min_nominal_tube,
        "minimum_backup_tube_margin_m": min_backup_tube,
        "speed_projection_count": int(getattr(env, "speed_projection_count", 0)),
        "minimum_planar_separation_m": min_sep,
        "minimum_planar_separation_step": min_sep_step,
        "initial_operational_margin_m": initial_operational,
        "minimum_operational_margin_m": min_operational,
        "minimum_absolute_operational_margin_m": min_abs_operational,
        "maximum_relative_speed_mps": max_relative_speed,
        "design_speed_envelope_exceeded": bool(
            max_relative_speed
            > design_speed + float(manifest["design_speed_tolerance_mps"])
        ),
        "minimum_uav_margin": min_uav,
        "minimum_ugv_margin": min_ugv,
        "minimum_hard_separation_margin": min_hard,
        "minimum_absolute_hard_margin": min_abs_hard,
        "final_relative_x_m": float(relative[0]),
        "final_relative_y_m": float(relative[1]),
        "completed_steps": env.step_index,
    }
    if set(row) != set(TRACE_FIELDS):
        raise CircaGazeboGZ0V7RunnerError("v7 trace schema drifted")
    return row


def _rate(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        raise CircaGazeboGZ0V7RunnerError("empty v7 factorial cell")
    return float(np.mean([row[field] for row in rows]))


def _diagnostic(
    rows: list[dict[str, Any]], families: tuple[str, ...], manifest: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    gates = manifest["development_gates"]
    manipulation: dict[str, Any] = {}
    for driver in DRIVERS:
        subset = [row for row in rows if row["driver_id"] == driver]
        manipulation[driver] = {
            "row_count": len(subset),
            "speed_envelope_exceedance_rate": _rate(
                subset, "design_speed_envelope_exceeded"
            ),
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
        candidate_ids = sorted(
            {row["candidate_id"] for row in rows if row["family_id"] == family}
        )
        for candidate_id in candidate_ids:
            candidate_rows = [row for row in rows if row["candidate_id"] == candidate_id]
            by_driver = {
                driver: [row for row in candidate_rows if row["driver_id"] == driver]
                for driver in DRIVERS
            }
            projected = by_driver["planar_speed_projected_v4"]
            unbounded = by_driver["command_persistent_unbounded_v3"]

            def cell(source: list[dict[str, Any]], controller: str) -> list[dict[str, Any]]:
                return [row for row in source if row["controller_id"] == controller]

            shadow = cell(projected, "shadow_no_override")
            robust = cell(projected, "robust_backup_filter_v7")
            cbf = cell(projected, "registered_one_step_cbf")
            unbounded_shadow = cell(unbounded, "shadow_no_override")
            unbounded_robust = cell(unbounded, "robust_backup_filter_v7")
            unbounded_cbf = cell(unbounded, "registered_one_step_cbf")
            item = {
                "candidate_id": candidate_id,
                "paired_seed_count": len(shadow),
                "unbounded_shadow_operational_violation_rate": _rate(
                    unbounded_shadow, "operational_first_violation"
                ),
                "unbounded_robust_backup_filter_operational_violation_rate": _rate(
                    unbounded_robust, "operational_first_violation"
                ),
                "unbounded_registered_cbf_operational_violation_rate": _rate(
                    unbounded_cbf, "operational_first_violation"
                ),
                "projected_shadow_operational_violation_rate": _rate(
                    shadow, "operational_first_violation"
                ),
                "projected_shadow_hard_violation_rate": _rate(shadow, "hard_first_violation"),
                "robust_backup_filter_operational_violation_rate": _rate(
                    robust, "operational_first_violation"
                ),
                "robust_backup_filter_hard_violation_rate": _rate(
                    robust, "hard_first_violation"
                ),
                "robust_backup_filter_intervention_rate": _rate(
                    robust, "applied_intervention"
                ),
                "robust_backup_filter_mean_fail_closed_steps": _rate(
                    robust, "robust_filter_fail_closed_steps"
                ),
                "registered_cbf_operational_violation_rate": _rate(
                    cbf, "operational_first_violation"
                ),
                "registered_cbf_hard_violation_rate": _rate(cbf, "hard_first_violation"),
                "projected_driver_speed_exceedance_rate": _rate(
                    projected, "design_speed_envelope_exceeded"
                ),
            }
            item["robust_backup_filter_paired_risk_reduction"] = (
                item["projected_shadow_operational_violation_rate"]
                - item["robust_backup_filter_operational_violation_rate"]
            )
            item["unbounded_robust_backup_filter_paired_risk_reduction"] = (
                item["unbounded_shadow_operational_violation_rate"]
                - item["unbounded_robust_backup_filter_operational_violation_rate"]
            )
            item["driver_by_robust_backup_filter_interaction"] = (
                item["robust_backup_filter_paired_risk_reduction"]
                - item["unbounded_robust_backup_filter_paired_risk_reduction"]
            )
            item["unbounded_registered_cbf_paired_risk_reduction"] = (
                item["unbounded_shadow_operational_violation_rate"]
                - item["unbounded_registered_cbf_operational_violation_rate"]
            )
            item["projected_registered_cbf_paired_risk_reduction"] = (
                item["projected_shadow_operational_violation_rate"]
                - item["registered_cbf_operational_violation_rate"]
            )
            item["robust_backup_filter_hard_risk_increase"] = (
                item["robust_backup_filter_hard_violation_rate"]
                - item["projected_shadow_hard_violation_rate"]
            )
            item["gate_pass"] = bool(
                gates["projected_shadow_operational_violation_rate_min"]
                <= item["projected_shadow_operational_violation_rate"]
                <= gates["projected_shadow_operational_violation_rate_max"]
                and item["robust_backup_filter_intervention_rate"]
                >= gates["predictive_shield_intervention_rate_min"]
                and item["robust_backup_filter_operational_violation_rate"]
                <= gates["predictive_shield_operational_violation_rate_max"]
                and item["robust_backup_filter_paired_risk_reduction"]
                >= gates["predictive_shield_paired_risk_reduction_min"]
                and item["robust_backup_filter_hard_risk_increase"]
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
                    abs(
                        item["projected_shadow_operational_violation_rate"]
                        - manifest["target_shadow_operational_violation_rate"]
                    ),
                    -item["robust_backup_filter_paired_risk_reduction"],
                    item["robust_backup_filter_operational_violation_rate"],
                    item["candidate_id"],
                ),
            )[0]
            if passing
            else None,
        }
    return manipulation, output


def run_gz0_v7(manifest_path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    if root not in manifest_file.parents or not manifest_file.is_file():
        raise CircaGazeboGZ0V7RunnerError("manifest path is unsafe or absent")
    manifest_raw = manifest_file.read_bytes()
    manifest = json.loads(manifest_raw)
    world, candidates = validate_runnable_manifest(manifest, root)
    output = _safe_child(root, manifest["output_path"])
    if output.exists():
        raise CircaGazeboGZ0V7RunnerError("exactly-once v7 output path already exists")
    hazard_indices = {int(value) for value in manifest["hazard_active_seed_indices"]}
    schedule = []
    seeds = iter(scenario_seeds(manifest))
    for candidate in candidates:
        for index in range(int(manifest["seeds_per_candidate"])):
            seed = next(seeds)
            for driver in DRIVERS:
                for controller in CONTROLLERS:
                    schedule.append(
                        (candidate, seed, index in hazard_indices, driver, controller)
                    )
    if len(schedule) != int(manifest["max_regime_rollouts"]):
        raise CircaGazeboGZ0V7RunnerError("generated v7 schedule does not match budget")
    random.Random(int(manifest["schedule_seed"])).shuffle(schedule)
    start = time.perf_counter()
    rows = []
    for candidate, seed, hazard_active, driver, controller in schedule:
        if time.perf_counter() - start > float(manifest["wall_time_seconds_max"]):
            raise CircaGazeboGZ0V7RunnerError("GZ0-v7 wall-time budget exceeded")
        rows.append(
            _run_regime(candidate, seed, hazard_active, driver, controller, world, manifest)
        )
    elapsed = time.perf_counter() - start
    manipulation, diagnostic = _diagnostic(rows, tuple(manifest["families"]), manifest)
    all_families = all(
        value["passing_candidate_count"] > 0 for value in diagnostic.values()
    )
    result = {
        "result_id": manifest["manifest_id"],
        "claim_eligible": False,
        "circa_gz1_run": False,
        "sealed_data_used": False,
        "formal_experiment_run": False,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "world_sha256": manifest["world_sha256"],
        "seed_namespace": SEED_NAMESPACE,
        "independent_seed_count": 60,
        "independent_unit": "candidate_by_scenario_seed; six regimes are paired within unit",
        "rows": rows,
        "factorial_manipulation": manipulation,
        "development_diagnostic": diagnostic,
        "all_families_have_passing_candidate": all_families,
        "route_gate_pass": bool(manipulation["pass"] and all_families),
        "elapsed_seconds": elapsed,
        "legacy_v2_fixture_disposition": manifest["legacy_v2_fixture_disposition"],
        "boundary": (
            "claim-ineligible GZ0-v7 Route A development only; no efficacy claim or "
            "GZ1 authorization"
        ),
    }
    body = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if len(body) > int(manifest["output_bytes_max"]):
        raise CircaGazeboGZ0V7RunnerError("GZ0-v7 output budget exceeded")
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_bytes(body)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    result = run_gz0_v7(args.manifest, args.repo_root)
    print(
        json.dumps(
            {
                "result_id": result["result_id"],
                "rows": len(result["rows"]),
                "elapsed_seconds": result["elapsed_seconds"],
                "factorial_manipulation_pass": result["factorial_manipulation"]["pass"],
                "all_families_have_passing_candidate": result[
                    "all_families_have_passing_candidate"
                ],
                "route_gate_pass": result["route_gate_pass"],
                "claim_eligible": False,
                "circa_gz1_run": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
