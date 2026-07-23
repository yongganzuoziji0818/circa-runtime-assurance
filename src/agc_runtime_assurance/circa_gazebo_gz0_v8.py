"""Non-runnable implementation audit kernel for frozen CIRCA-GZ0-v8.

The module validates the implementation-only manifest and deterministic fixtures.
It has no schedule generator, scientific runner, output writer, or launch path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .circa_gazebo_gz0 import SceneCandidate
from .circa_gazebo_gz0_v3 import derive_operational_envelope
from .gazebo_robust_backup_filter import (
    GazeboPlanarPlant,
    RobustBackupConfig,
    propagate_planar_state,
)
from .gazebo_role_valid_task import build_role_valid_scenario
from .gazebo_timestamp_aligned_set_filter import (
    TimestampAlignedSetBackupFilter,
    TimestampAlignmentConfig,
    align_async_state_history,
)


class CircaGazeboGZ0V8AuditError(RuntimeError):
    pass


DRIVERS = ("command_persistent_unbounded_v3", "planar_speed_projected_v4")
REGIMES = (
    "shadow_no_override",
    "registered_one_step_cbf",
    "robust_backup_filter_v7_stale_point",
    "timestamp_aligned_point_backup_v8_ablation",
    "timestamp_aligned_set_backup_v8",
)
FORBIDDEN_RUNTIME_FIELDS = (
    "master_seed",
    "schedule_seed",
    "output_path",
    "admission_path",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise CircaGazeboGZ0V8AuditError(f"unsafe path: {relative}")
    if not path.is_file():
        raise CircaGazeboGZ0V8AuditError(f"missing file: {relative}")
    return path


def _same_candidate_payload(current: dict[str, Any], prior: dict[str, Any]) -> bool:
    ignored = {"candidate_id"}
    return all(
        current.get(key) == value for key, value in prior.items() if key not in ignored
    ) and current["candidate_id"] == prior["candidate_id"].replace("V7", "V8")


def alignment_config(manifest: dict[str, Any]) -> TimestampAlignmentConfig:
    values = manifest["structured_uncertainty"]
    contract = manifest["timestamp_alignment_contract"]
    return TimestampAlignmentConfig(
        per_agent_position_error_bound_m=float(
            values["per_agent_position_error_bound_m_per_axis"]
        ),
        per_agent_velocity_error_bound_mps=float(
            values["per_agent_velocity_error_bound_mps_per_axis"]
        ),
        relative_acceleration_error_bound_mps2=float(
            values["relative_acceleration_error_bound_mps2"]
        ),
        common_mode_position_bias_bound_m=float(
            values["common_mode_position_bias_bound_m"]
        ),
        maximum_observation_age_steps=int(contract["maximum_observation_age_steps"]),
        maximum_additional_neighbor_communication_age_steps=int(
            contract["maximum_additional_neighbor_communication_age_steps"]
        ),
    )


def build_filter(
    candidate: SceneCandidate, manifest: dict[str, Any]
) -> TimestampAlignedSetBackupFilter:
    envelope = derive_operational_envelope(
        candidate, manifest["operational_envelope_assumptions"]
    )
    parameters = manifest["backup_filter"]
    plant = GazeboPlanarPlant(
        uav_mass=candidate.uav_mass,
        uav_drag=candidate.uav_drag,
        ugv_friction=candidate.ugv_friction,
        actuator_lag=candidate.actuator_lag,
        dt=float(parameters["sample_period_s"]),
        speed_limit_mps=0.5 * envelope.design_relative_speed_mps,
    )
    backup = RobustBackupConfig(
        operational_separation_m=envelope.operational_separation_m,
        action_limit=float(parameters["task_action_limit"]),
        horizon_steps=int(parameters["backup_horizon_steps"]),
        terminal_margin_m=float(parameters["terminal_margin_m"]),
        position_error_bound_m=float(
            manifest["structured_uncertainty"][
                "per_agent_position_error_bound_m_per_axis"
            ]
        ),
        velocity_error_bound_mps=float(
            manifest["structured_uncertainty"][
                "per_agent_velocity_error_bound_mps_per_axis"
            ]
        ),
        relative_acceleration_error_bound_mps2=float(
            manifest["structured_uncertainty"][
                "relative_acceleration_error_bound_mps2"
            ]
        ),
        barrier_retention=float(parameters["barrier_retention"]),
    )
    return TimestampAlignedSetBackupFilter(plant, backup, alignment_config(manifest))


def _scenario(
    candidate: SceneCandidate, manifest: dict[str, Any], *, hazard_active: bool
):
    envelope = derive_operational_envelope(
        candidate, manifest["operational_envelope_assumptions"]
    )
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


def deterministic_alignment_fixture(
    candidate: SceneCandidate, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Exercise timestamp alignment without producing an efficacy observation."""

    safety_filter = build_filter(candidate, manifest)
    scenario = _scenario(candidate, manifest, hazard_active=True)
    state = scenario.initial_state.copy()
    applied = np.zeros(5, dtype=float)
    states = [state.copy()]
    actions = [applied.copy()]
    for _ in range(6):
        command = safety_filter.backup_action(state)
        state, applied = propagate_planar_state(
            state, applied, command, safety_filter.plant
        )
        states.append(state.copy())
        actions.append(applied.copy())
    aligned = align_async_state_history(
        states,
        actions,
        candidate.observation_delay_steps,
        candidate.communication_delay_steps,
        safety_filter.plant,
        safety_filter.alignment,
        observed_common_mode_bias_m=candidate.sensor_bias,
    )
    # The registered sensor bias is common mode.  It belongs in each absolute
    # center but cancels in the relative state used by the separation
    # certificate.  Audit that actual certification object instead of asking
    # an absolute-position interval (which deliberately excludes common-mode
    # bias) to cover an unobservable global translation.
    relative_center = np.concatenate(
        [aligned.center[:2] - aligned.center[6:8], aligned.center[3:5] - aligned.center[8:10]]
    )
    relative_truth = np.concatenate(
        [states[-1][:2] - states[-1][6:8], states[-1][3:5] - states[-1][8:10]]
    )
    relative_radius = np.concatenate(
        [aligned.radius[:2] + aligned.radius[6:8], aligned.radius[3:5] + aligned.radius[8:10]]
    )
    error = np.abs(relative_center - relative_truth)
    enclosed = bool(np.all(error <= relative_radius + 1e-12))
    nominal = scenario.task.nominal_action(
        np.concatenate([aligned.center, actions[-1]])
    )
    full = safety_filter.decide(aligned, actions[-1], nominal)
    point = safety_filter.decide(
        aligned, actions[-1], nominal, point_ablation=True
    )
    return {
        "fixture_type": "deterministic_non_evidence_alignment_and_certificate",
        "candidate_id": candidate.candidate_id,
        "truth_enclosed": enclosed,
        "local_age_steps": aligned.local_age_steps,
        "neighbor_age_steps": aligned.neighbor_age_steps,
        "source_center_radius_shape": [10, 10],
        "audited_relative_state_dimension": 4,
        "provenance_hash": aligned.provenance_hash,
        "full_set_certificate_emitted": full.certificate_emitted,
        "point_ablation_certificate_emitted": point.certificate_emitted,
        "full_set_reason": full.reason,
        "point_ablation_reason": point.reason,
        "scientific_output_generated": False,
    }


def validate_implementation_manifest(
    manifest: dict[str, Any], root: Path
) -> tuple[Path, tuple[SceneCandidate, ...], dict[str, Any]]:
    expected = {
        "status": "implementation_frozen_nonrunnable",
        "route_authorized": True,
        "local_implementation_authorized": True,
        "remote_upload_authorized": False,
        "preflight_authorized": True,
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
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise CircaGazeboGZ0V8AuditError(
                f"implementation manifest {key} must equal {value!r}"
            )
    forbidden = [key for key in FORBIDDEN_RUNTIME_FIELDS if key in manifest]
    if forbidden:
        raise CircaGazeboGZ0V8AuditError(
            f"runtime seed/output material is forbidden: {forbidden}"
        )
    design_path = _safe_file(root, manifest["design_source_manifest_path"])
    if _sha256(design_path) != manifest["design_source_manifest_sha256"]:
        raise CircaGazeboGZ0V8AuditError("v8 frozen design hash mismatch")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if tuple(design.get("drivers", ())) != DRIVERS:
        raise CircaGazeboGZ0V8AuditError("v8 driver factors drifted")
    if tuple(design.get("regimes_per_driver", ())) != REGIMES:
        raise CircaGazeboGZ0V8AuditError("v8 method factors drifted")
    resolved = dict(design)
    resolved.update(manifest)

    v7_path = _safe_file(root, manifest["v7_source_manifest_path"])
    if _sha256(v7_path) != manifest["v7_source_manifest_sha256"]:
        raise CircaGazeboGZ0V8AuditError("v7 protected manifest hash mismatch")
    v7 = json.loads(v7_path.read_text(encoding="utf-8"))
    if len(design["candidates"]) != 12 or len(v7["candidates"]) != 12:
        raise CircaGazeboGZ0V8AuditError("candidate count must remain 12")
    if not all(
        _same_candidate_payload(current, prior)
        for current, prior in zip(design["candidates"], v7["candidates"])
    ):
        raise CircaGazeboGZ0V8AuditError("v8 candidate payload drifted from v7")

    for relative, expected_hash in manifest.get("source_files", {}).items():
        if _sha256(_safe_file(root, relative)) != expected_hash:
            raise CircaGazeboGZ0V8AuditError(f"source hash mismatch: {relative}")
    for relative, expected_hash in manifest.get("protected_files", {}).items():
        if _sha256(_safe_file(root, relative)) != expected_hash:
            raise CircaGazeboGZ0V8AuditError(
                f"protected evidence hash mismatch: {relative}"
            )
    if not manifest.get("source_files") or not manifest.get("protected_files"):
        raise CircaGazeboGZ0V8AuditError("source/protected locks must be nonempty")

    world = _safe_file(root, design["world_path"])
    if _sha256(world) != design["world_sha256"]:
        raise CircaGazeboGZ0V8AuditError("world hash mismatch")
    expected_rollouts = (
        len(design["candidates"])
        * int(design["seeds_per_candidate"])
        * len(DRIVERS)
        * len(REGIMES)
    )
    if expected_rollouts != 600 or expected_rollouts != int(
        design["max_regime_rollouts"]
    ):
        raise CircaGazeboGZ0V8AuditError("prospective v8 design size drifted")
    if (root / design["output_namespace_reserved"]).exists():
        raise CircaGazeboGZ0V8AuditError("v8 scientific output namespace must not exist")
    candidates = tuple(SceneCandidate.from_dict(value) for value in design["candidates"])
    return world, candidates, resolved


def audit_implementation(
    manifest_path: str | Path, repo_root: str | Path
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    path = Path(manifest_path).resolve()
    if root not in path.parents or not path.is_file():
        raise CircaGazeboGZ0V8AuditError("manifest path is unsafe or absent")
    raw = path.read_bytes()
    manifest = json.loads(raw)
    world, candidates, resolved = validate_implementation_manifest(manifest, root)
    fixtures = [deterministic_alignment_fixture(candidate, resolved) for candidate in candidates]
    if not all(item["truth_enclosed"] for item in fixtures):
        raise CircaGazeboGZ0V8AuditError("deterministic truth enclosure fixture failed")
    if not all(
        item["full_set_certificate_emitted"]
        and item["point_ablation_certificate_emitted"]
        for item in fixtures
    ):
        raise CircaGazeboGZ0V8AuditError("valid fixture failed to emit certificate")
    return {
        "audit_type": "local_non_evidence_v8_implementation_audit",
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "world_sha256": _sha256(world),
        "candidate_count": len(candidates),
        "prospective_rollout_count": 600,
        "deterministic_truth_enclosure_fixtures_passed": sum(
            int(item["truth_enclosed"]) for item in fixtures
        ),
        "full_set_certificate_fixtures_passed": sum(
            int(item["full_set_certificate_emitted"]) for item in fixtures
        ),
        "point_ablation_certificate_fixtures_passed": sum(
            int(item["point_ablation_certificate_emitted"]) for item in fixtures
        ),
        "seed_material_generated": False,
        "scientific_output_generated": False,
        "scientific_run_authorized": False,
        "circa_gz1_authorized": False,
        "fixtures": fixtures,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Audit the non-runnable CIRCA-GZ0-v8 implementation"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_implementation(args.manifest, args.repo_root),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
