"""Non-running scientific runner and lossless evidence codec for GZ0-v9.

The module contains the future exactly-once execution path, but the current
schema-construction manifest cannot activate it: it has no scientific seed
material, output path, capacity-audit authorization, or launch authorization.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np

from .circa_gazebo_gz0 import SceneCandidate
from .circa_gazebo_gz0_v3 import derive_operational_envelope
from .circa_gazebo_gz0_v8 import _scenario, build_filter
from .gazebo_second_system import constraint_margins
from .gazebo_second_system_v3 import GazeboAirGroundEnvV3
from .gazebo_second_system_v4 import GazeboAirGroundEnvV4
from .gazebo_timestamp_aligned_set_filter import align_async_state_history
from .sandbox_baselines import SandboxBaselineInfeasible, SandboxNominalCBFAdapter


class CircaGazeboGZ0V9RunnerError(RuntimeError):
    """Fail-closed runner/schema error."""


DRIVERS = ("command_persistent_unbounded_v3", "planar_speed_projected_v4")
METHODS = (
    "shadow_no_override",
    "registered_one_step_cbf",
    "robust_backup_filter_v7_stale_point",
    "timestamp_aligned_point_backup_v8_ablation",
    "timestamp_aligned_set_backup_v8",
)
TRACE_REQUIRED_FIELDS = (
    "source_timestamps",
    "decision_timestamp",
    "local_age_steps",
    "neighbor_age_steps",
    "applied_action_history_digest",
    "source_center_radius_set",
    "aligned_center_radius_set",
    "uncertainty_registry_hash",
    "nominal_tube_feasible",
    "backup_tube_feasible",
    "worst_set_margin",
    "terminal_reachability",
    "selected_action",
    "intervention_reason",
    "refusal_code",
    "certificate_validity_interval",
    "provenance_hash",
    "applied_planar_velocity",
    "design_speed_limit_mps",
    "design_speed_envelope_exceeded",
    "method_decision_reason",
    "backup_tube_infeasible_fail_closed",
    "completed_step_mask",
    "simulator_state_hash",
    "envelope_registry_hash",
    "decision_codebook_hash",
)
REASON_CODEBOOK = (
    "not_evaluated",
    "shadow_no_override",
    "registered_one_step_cbf",
    "registered_one_step_cbf_fallback",
    "robust_backup_filter_v7",
    "nominal_with_verified_aligned_set_backup_tube",
    "aligned_set_backup_filter_intervention",
    "aligned_set_backup_tube_infeasible_fail_closed",
    "alignment_evidence_refused",
    "nominal_with_verified_backup_tube",
    "backup_filter_intervention",
    "backup_tube_infeasible_fail_closed",
)
REFUSAL_CODEBOOK = ("none", "invalid_alignment_evidence", "missing_numeric_certificate")
SCHEMA_VERSION = "circa-gz0-v9-evidence-complete-summary-json-lossless-arrays-v1"
SEED_NAMESPACE = "circa-gz0-v9"


@dataclass(frozen=True)
class FactorialRun:
    candidate_index: int
    scenario_index: int
    driver_index: int
    method_index: int
    hazard_active: bool
    scenario_seed: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root not in path.parents:
        raise CircaGazeboGZ0V9RunnerError(f"unsafe path outside repository: {relative}")
    return path


def _binary_hash(value: str) -> np.ndarray:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise CircaGazeboGZ0V9RunnerError("invalid lowercase SHA-256 value")
    return np.frombuffer(bytes.fromhex(value), dtype=np.uint8)


def array_schema(rollouts: int, horizon: int) -> dict[str, tuple[np.dtype, tuple[int, ...]]]:
    if rollouts <= 0 or horizon <= 0:
        raise CircaGazeboGZ0V9RunnerError("rollouts and horizon must be positive")
    rh = (rollouts, horizon)
    return {
        "source_timestamps": (np.dtype("u1"), (12, horizon, 2)),
        "decision_timestamp": (np.dtype("u1"), (horizon,)),
        "local_age_steps": (np.dtype("u1"), (12, horizon)),
        "neighbor_age_steps": (np.dtype("u1"), (12, horizon)),
        "applied_action_history_digest": (np.dtype("u1"), rh + (32,)),
        "source_center_radius_set": (np.dtype("<f8"), rh + (2, 10)),
        "aligned_center_radius_set": (np.dtype("<f8"), rh + (2, 10)),
        "uncertainty_registry_hash": (np.dtype("u1"), (32,)),
        "nominal_tube_feasible": (np.dtype("i1"), rh),
        "backup_tube_feasible": (np.dtype("i1"), rh),
        "worst_set_margin": (np.dtype("<f8"), rh),
        "terminal_reachability": (np.dtype("i1"), rh),
        "selected_action": (np.dtype("<f8"), rh + (5,)),
        "intervention_reason": (np.dtype("u1"), rh),
        "refusal_code": (np.dtype("u1"), rh),
        "certificate_validity_interval": (np.dtype("i1"), (horizon, 2)),
        "provenance_hash": (np.dtype("u1"), rh + (32,)),
        "applied_planar_velocity": (np.dtype("<f8"), rh + (2, 2)),
        "design_speed_limit_mps": (np.dtype("<f8"), (rollouts,)),
        "design_speed_envelope_exceeded": (np.dtype("?"), rh),
        "method_decision_reason": (np.dtype("u1"), rh),
        "backup_tube_infeasible_fail_closed": (np.dtype("?"), rh),
        "completed_step_mask": (np.dtype("?"), rh),
        "simulator_state_hash": (np.dtype("u1"), rh + (32,)),
        "envelope_registry_hash": (np.dtype("u1"), (12, 32)),
        "decision_codebook_hash": (np.dtype("u1"), (32,)),
        "trace_valid": (np.dtype("?"), rh),
        "candidate_index": (np.dtype("u1"), (rollouts,)),
        "scenario_index": (np.dtype("u1"), (rollouts,)),
        "scenario_seed": (np.dtype("<i4"), (rollouts,)),
        "driver_index": (np.dtype("u1"), (rollouts,)),
        "method_index": (np.dtype("u1"), (rollouts,)),
        "hazard_active": (np.dtype("?"), (rollouts,)),
        "operational_first_violation": (np.dtype("?"), (rollouts,)),
        "hard_first_violation": (np.dtype("?"), (rollouts,)),
        "applied_intervention": (np.dtype("?"), (rollouts,)),
        "completed_steps": (np.dtype("u1"), (rollouts,)),
        "minimum_operational_margin_m": (np.dtype("<f8"), (rollouts,)),
        "minimum_hard_margin_m": (np.dtype("<f8"), (rollouts,)),
    }


def allocate_arrays(rollouts: int, horizon: int) -> dict[str, np.ndarray]:
    arrays = {
        name: np.zeros(shape, dtype=dtype)
        for name, (dtype, shape) in array_schema(rollouts, horizon).items()
    }
    for name in ("nominal_tube_feasible", "backup_tube_feasible", "terminal_reachability"):
        arrays[name].fill(-1)
    arrays["worst_set_margin"].fill(np.nan)
    arrays["certificate_validity_interval"].fill(-1)
    arrays["minimum_operational_margin_m"].fill(np.nan)
    arrays["minimum_hard_margin_m"].fill(np.nan)
    return arrays


def validate_arrays(arrays: Mapping[str, np.ndarray], rollouts: int, horizon: int) -> None:
    expected = array_schema(rollouts, horizon)
    if set(arrays) != set(expected):
        raise CircaGazeboGZ0V9RunnerError("lossless array member set drifted")
    for name, (dtype, shape) in expected.items():
        value = np.asarray(arrays[name])
        if value.dtype != dtype or value.shape != shape or value.dtype.hasobject:
            raise CircaGazeboGZ0V9RunnerError(f"array schema mismatch: {name}")
    completed = np.asarray(arrays["completed_step_mask"], dtype=bool)
    velocities = np.asarray(arrays["applied_planar_velocity"], dtype=float)
    limits = np.asarray(arrays["design_speed_limit_mps"], dtype=float)[:, None]
    relative_speed = np.linalg.norm(velocities[:, :, 0] - velocities[:, :, 1], axis=2)
    expected_exceeded = relative_speed > limits
    if np.any(
        np.asarray(arrays["design_speed_envelope_exceeded"])[completed]
        != expected_exceeded[completed]
    ):
        raise CircaGazeboGZ0V9RunnerError("speed-envelope evidence is not reproducible")
    reasons = np.asarray(arrays["method_decision_reason"])
    fail_closed = np.asarray(arrays["backup_tube_infeasible_fail_closed"])
    allowed_fail_closed = {
        REASON_CODEBOOK.index("backup_tube_infeasible_fail_closed"),
        REASON_CODEBOOK.index("aligned_set_backup_tube_infeasible_fail_closed"),
    }
    if np.any(fail_closed & ~np.isin(reasons, tuple(allowed_fail_closed))):
        raise CircaGazeboGZ0V9RunnerError("fail-closed evidence lacks a typed reason")


def encode_lossless_arrays(arrays: Mapping[str, np.ndarray]) -> bytes:
    members = {name: np.asarray(arrays[name]) for name in sorted(arrays)}
    stream = BytesIO()
    np.savez_compressed(stream, **members)
    return stream.getvalue()


def verify_lossless_arrays(payload: bytes, expected: Mapping[str, np.ndarray]) -> dict[str, Any]:
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise CircaGazeboGZ0V9RunnerError("archive member set drifted")
        for name, original in expected.items():
            restored = archive[name]
            if restored.dtype != original.dtype or restored.shape != original.shape:
                raise CircaGazeboGZ0V9RunnerError(f"archive schema drifted: {name}")
            if restored.tobytes() != original.tobytes():
                raise CircaGazeboGZ0V9RunnerError(f"archive round trip is not lossless: {name}")
    return {"members": len(expected), "bytes": len(payload), "lossless": True}


def _fixture_values(shape: tuple[int, ...], *, stress: bool, offset: int) -> np.ndarray:
    count = int(np.prod(shape))
    index = np.arange(count, dtype=np.uint64)
    if stress:
        mixed = index * np.uint64(6364136223846793005) + np.uint64(
            1442695040888963407 + offset
        )
        values = ((mixed >> np.uint64(11)).astype(np.float64) / float(2**53)) * 8.0 - 4.0
    else:
        values = ((index + np.uint64(offset)) % np.uint64(257)).astype(np.float64) / 32.0
    return values.reshape(shape)


def schema_fixture_arrays(
    rollouts: int, horizon: int, *, stress: bool
) -> dict[str, np.ndarray]:
    arrays = allocate_arrays(rollouts, horizon)
    for offset, name in enumerate(
        (
            "source_center_radius_set",
            "aligned_center_radius_set",
            "selected_action",
            "applied_planar_velocity",
            "design_speed_limit_mps",
        )
    ):
        arrays[name][...] = _fixture_values(arrays[name].shape, stress=stress, offset=offset + 1)
    arrays["worst_set_margin"][...] = _fixture_values(
        arrays["worst_set_margin"].shape, stress=stress, offset=7
    )
    for offset, name in enumerate(
        (
            "applied_action_history_digest",
            "provenance_hash",
            "uncertainty_registry_hash",
            "simulator_state_hash",
            "envelope_registry_hash",
            "decision_codebook_hash",
        )
    ):
        flat = np.arange(arrays[name].size, dtype=np.uint32)
        values = (flat * np.uint32(1664525) + np.uint32(1013904223 + offset)).astype(np.uint8)
        arrays[name][...] = values.reshape(arrays[name].shape)
    step = np.arange(horizon, dtype=np.uint8)
    arrays["decision_timestamp"][...] = step
    for candidate_index in range(12):
        local_delay = candidate_index % 4
        neighbor_delay = min(6, local_delay + candidate_index % 3)
        arrays["source_timestamps"][candidate_index, :, 0] = np.maximum(
            step.astype(np.int16) - local_delay, 0
        )
        arrays["source_timestamps"][candidate_index, :, 1] = np.maximum(
            step.astype(np.int16) - neighbor_delay, 0
        )
        arrays["local_age_steps"][candidate_index] = np.minimum(step, local_delay)
        arrays["neighbor_age_steps"][candidate_index] = np.minimum(step, neighbor_delay)
    arrays["nominal_tube_feasible"].fill(1)
    arrays["backup_tube_feasible"].fill(1)
    arrays["terminal_reachability"].fill(1)
    arrays["intervention_reason"].fill(5)
    arrays["method_decision_reason"].fill(5)
    arrays["completed_step_mask"].fill(True)
    arrays["certificate_validity_interval"][:, 0] = step
    arrays["certificate_validity_interval"][:, 1] = step
    arrays["trace_valid"].fill(True)
    row = np.arange(rollouts, dtype=np.int64)
    arrays["candidate_index"][...] = row % 12
    arrays["scenario_index"][...] = (row // 10) % 5
    arrays["scenario_seed"][...] = -1  # schema sentinel, never a generated scientific seed
    arrays["driver_index"][...] = (row // 5) % 2
    arrays["method_index"][...] = row % 5
    arrays["hazard_active"][...] = (row % 5) % 2 == 0
    arrays["completed_steps"].fill(horizon)
    relative_speed = np.linalg.norm(
        arrays["applied_planar_velocity"][:, :, 0]
        - arrays["applied_planar_velocity"][:, :, 1],
        axis=2,
    )
    arrays["design_speed_envelope_exceeded"][...] = (
        relative_speed > arrays["design_speed_limit_mps"][:, None]
    )
    validate_arrays(arrays, rollouts, horizon)
    return arrays


def materialize_required_fields(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Expand lossless templates to the frozen rollout-by-step logical trace."""

    rollouts, horizon = arrays["trace_valid"].shape
    candidates = np.asarray(arrays["candidate_index"], dtype=np.int64)
    if np.any(candidates >= 12):
        raise CircaGazeboGZ0V9RunnerError("candidate template index is invalid")
    output: dict[str, np.ndarray] = {}
    for name in TRACE_REQUIRED_FIELDS:
        if name in ("source_timestamps", "local_age_steps", "neighbor_age_steps"):
            output[name] = np.asarray(arrays[name])[candidates]
        elif name == "decision_timestamp":
            output[name] = np.broadcast_to(arrays[name], (rollouts, horizon)).copy()
        elif name == "uncertainty_registry_hash":
            output[name] = np.broadcast_to(arrays[name], (rollouts, 32)).copy()
        elif name == "certificate_validity_interval":
            template = np.broadcast_to(arrays[name], (rollouts, horizon, 2)).copy()
            template[arrays["nominal_tube_feasible"] < 0] = -1
            output[name] = template
        else:
            output[name] = np.asarray(arrays[name])
    return output


def summary_json_bytes(
    *, rollouts: int, horizon: int, array_bytes: int, schema_only: bool
) -> bytes:
    schema = array_schema(rollouts, horizon)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "claim_eligible": False,
        "schema_only": bool(schema_only),
        "scientific_run_executed": False if schema_only else None,
        "scientific_seed_material_generated": False if schema_only else None,
        "independent_unit": "candidate_x_scenario_seed",
        "within_unit_pairing": "all_five_methods_receive_identical_candidate_seed_realization",
        "drivers": list(DRIVERS),
        "methods": list(METHODS),
        "trace_required_fields": list(TRACE_REQUIRED_FIELDS),
        "reason_codebook": list(REASON_CODEBOOK),
        "refusal_codebook": list(REFUSAL_CODEBOOK),
        "rollouts": rollouts,
        "horizon": horizon,
        "array_archive_bytes": array_bytes,
        "arrays": {
            name: {"dtype": dtype.str, "shape": list(shape)}
            for name, (dtype, shape) in schema.items()
        },
        "results": None if schema_only else {},
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def schema_capacity_audit(
    *, rollouts: int, horizon: int, output_bytes_max: int
) -> dict[str, Any]:
    if output_bytes_max <= 0:
        raise CircaGazeboGZ0V9RunnerError("output budget must be positive")
    structured = schema_fixture_arrays(rollouts, horizon, stress=False)
    structured_payload = encode_lossless_arrays(structured)
    structured_check = verify_lossless_arrays(structured_payload, structured)
    stress = schema_fixture_arrays(rollouts, horizon, stress=True)
    stress_payload = encode_lossless_arrays(stress)
    stress_check = verify_lossless_arrays(stress_payload, stress)
    raw_bytes = sum(value.nbytes for value in stress.values())
    summary = summary_json_bytes(
        rollouts=rollouts,
        horizon=horizon,
        array_bytes=len(stress_payload),
        schema_only=True,
    )
    conservative_bytes = len(stress_payload) + len(summary)
    return {
        "schema_version": SCHEMA_VERSION,
        "rollouts": rollouts,
        "horizon": horizon,
        "trace_required_field_count": len(TRACE_REQUIRED_FIELDS),
        "archive_member_count": len(stress),
        "raw_array_bytes": raw_bytes,
        "structured_fixture_npz_bytes": len(structured_payload),
        "entropy_stress_npz_bytes": len(stress_payload),
        "summary_json_bytes": len(summary),
        "conservative_total_bytes": conservative_bytes,
        "output_bytes_max": output_bytes_max,
        "lossless_round_trip": bool(
            structured_check["lossless"] and stress_check["lossless"]
        ),
        "capacity_pass": conservative_bytes <= output_bytes_max,
        "capacity_rule": "entropy_stress_npz_plus_summary_json_must_not_exceed_budget",
        "scientific_seed_material_generated": False,
        "scientific_output_generated": False,
    }


def derive_scenario_seed(master_seed: int, candidate_id: str, index: int) -> int:
    if not isinstance(master_seed, int) or master_seed < 0 or not candidate_id or index < 0:
        raise CircaGazeboGZ0V9RunnerError("invalid future scientific seed input")
    digest = hashlib.sha256(
        f"{SEED_NAMESPACE}|{master_seed}|{candidate_id}|{index}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def compile_schedule(manifest: Mapping[str, Any]) -> list[FactorialRun]:
    if "master_seed" not in manifest:
        raise CircaGazeboGZ0V9RunnerError("scientific seed material is absent")
    hazards = {int(value) for value in manifest["hazard_active_seed_indices"]}
    schedule = []
    for candidate_index, candidate in enumerate(manifest["candidates"]):
        for scenario_index in range(int(manifest["seeds_per_candidate"])):
            seed = derive_scenario_seed(
                int(manifest["master_seed"]), candidate["candidate_id"], scenario_index
            )
            for driver_index in range(len(DRIVERS)):
                for method_index in range(len(METHODS)):
                    schedule.append(
                        FactorialRun(
                            candidate_index,
                            scenario_index,
                            driver_index,
                            method_index,
                            scenario_index in hazards,
                            seed,
                        )
                    )
    if len(schedule) != int(manifest["max_regime_rollouts"]):
        raise CircaGazeboGZ0V9RunnerError("factorial schedule size drifted")
    random.Random(int(manifest["schedule_seed"])).shuffle(schedule)
    return schedule


def validate_schema_only_manifest(manifest: Mapping[str, Any], root: Path) -> dict[str, Any]:
    expected = {
        "status": "RUNNER_IMPLEMENTED_NONRUNNABLE_SCHEMA_CONSTRUCTION_ONLY",
        "local_runner_implementation_authorized": True,
        "schema_capacity_audit_authorized": False,
        "scientific_run_authorized": False,
        "exactly_once_authorization": False,
        "scientific_seed_material_generated": False,
        "scientific_output_authorized": False,
        "remote_upload_authorized": False,
        "circa_gz1_authorized": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise CircaGazeboGZ0V9RunnerError(f"schema manifest {key} drifted")
    for forbidden in ("master_seed", "schedule_seed", "seed_namespace", "output_path"):
        if forbidden in manifest:
            raise CircaGazeboGZ0V9RunnerError(f"forbidden runtime field present: {forbidden}")
    base = _safe_child(root, manifest["design_source_manifest_path"])
    if _sha256(base) != manifest["design_source_manifest_sha256"]:
        raise CircaGazeboGZ0V9RunnerError("frozen design hash mismatch")
    frozen = json.loads(base.read_text(encoding="utf-8"))
    if frozen.get("status") != "SCIENTIFIC_ROUTE_FROZEN_DESIGN_ONLY_NONRUNNABLE":
        raise CircaGazeboGZ0V9RunnerError("v9 route is not frozen")
    payload = _safe_child(root, manifest["scientific_payload_source_path"])
    if _sha256(payload) != manifest["scientific_payload_source_sha256"]:
        raise CircaGazeboGZ0V9RunnerError("v8 carry-forward payload hash mismatch")
    design = json.loads(payload.read_text(encoding="utf-8"))
    if tuple(design["drivers"]) != DRIVERS or tuple(design["regimes_per_driver"]) != METHODS:
        raise CircaGazeboGZ0V9RunnerError("frozen factorial levels drifted")
    if tuple(manifest["trace_required_fields"]) != TRACE_REQUIRED_FIELDS:
        raise CircaGazeboGZ0V9RunnerError("v9 evidence fields drifted")
    expected_rollouts = (
        len(design["candidates"])
        * int(design["seeds_per_candidate"])
        * len(DRIVERS)
        * len(METHODS)
    )
    if expected_rollouts != 600 or expected_rollouts != int(design["max_regime_rollouts"]):
        raise CircaGazeboGZ0V9RunnerError("frozen factorial size drifted")
    if (root / manifest["reserved_scientific_output_path"]).exists():
        raise CircaGazeboGZ0V9RunnerError("reserved scientific output must remain absent")
    for group in ("source_files", "protected_files"):
        if not manifest.get(group):
            raise CircaGazeboGZ0V9RunnerError(f"{group} lock is empty")
        for relative, expected_hash in manifest[group].items():
            if _sha256(_safe_child(root, relative)) != expected_hash:
                raise CircaGazeboGZ0V9RunnerError(
                    f"{group} hash mismatch: {relative}"
                )
    return design


def _validate_runnable_manifest(
    manifest: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], tuple[SceneCandidate, ...], Path]:
    required = {
        "status": "AUTHORIZED_EXACTLY_ONCE",
        "scientific_run_authorized": True,
        "exactly_once_authorization": True,
        "scientific_seed_material_generated": True,
        "scientific_output_authorized": True,
        "scientific_claim_allowed": False,
        "circa_gz1_authorized": False,
        "retry_allowed": False,
        "seed_top_up_allowed": False,
        "seed_namespace": SEED_NAMESPACE,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise CircaGazeboGZ0V9RunnerError(f"runnable manifest {key} drifted")
    design_path = _safe_child(root, manifest["design_source_manifest_path"])
    if _sha256(design_path) != manifest["design_source_manifest_sha256"]:
        raise CircaGazeboGZ0V9RunnerError("runnable design lock failed")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    invariants = (
        "drivers",
        "regimes_per_driver",
        "role_validity",
        "timestamp_alignment_contract",
        "structured_uncertainty",
        "backup_filter",
        "families",
        "candidates_per_family",
        "seeds_per_candidate",
        "hazard_active_seed_indices",
        "horizon_steps",
        "operational_envelope_assumptions",
        "development_gates",
        "analysis_contract",
        "trace_required_fields",
        "max_regime_rollouts",
        "wall_time_seconds_max",
        "output_bytes_max",
        "candidates",
    )
    drift = [key for key in invariants if manifest.get(key) != design.get(key)]
    if drift:
        raise CircaGazeboGZ0V9RunnerError(f"runnable scientific payload drifted: {drift}")
    if tuple(manifest["drivers"]) != DRIVERS or tuple(manifest["regimes_per_driver"]) != METHODS:
        raise CircaGazeboGZ0V9RunnerError("runnable factorial levels drifted")
    if tuple(manifest["trace_required_fields"]) != TRACE_REQUIRED_FIELDS:
        raise CircaGazeboGZ0V9RunnerError("runnable trace fields drifted")
    for group in ("source_files", "protected_files"):
        if not manifest.get(group):
            raise CircaGazeboGZ0V9RunnerError(f"{group} lock is empty")
        for relative, expected_hash in manifest[group].items():
            if _sha256(_safe_child(root, relative)) != expected_hash:
                raise CircaGazeboGZ0V9RunnerError(f"{group} hash mismatch: {relative}")
    world = _safe_child(root, manifest["world_path"])
    if _sha256(world) != manifest["world_sha256"]:
        raise CircaGazeboGZ0V9RunnerError("world hash mismatch")
    output = _safe_child(root, manifest["output_path"])
    if output.exists():
        raise CircaGazeboGZ0V9RunnerError("exactly-once output already exists")
    candidates = tuple(SceneCandidate.from_dict(value) for value in manifest["candidates"])
    schedule = compile_schedule(manifest)
    seeds = {item.scenario_seed for item in schedule}
    if len(schedule) != 600 or len(seeds) != 60:
        raise CircaGazeboGZ0V9RunnerError("runnable paired schedule is invalid")
    return design, candidates, world


def _source_center_radius(
    states: list[np.ndarray], candidate: SceneCandidate, decision_step: int, design: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, int, int]:
    local = max(0, decision_step - candidate.observation_delay_steps)
    neighbor = max(
        0,
        decision_step
        - candidate.observation_delay_steps
        - candidate.communication_delay_steps,
    )
    center = states[local].copy()
    center[6:10] = states[neighbor][6:10]
    center[[0, 1, 6, 7]] += candidate.sensor_bias
    radius = np.zeros(10, dtype=float)
    uncertainty = design["structured_uncertainty"]
    radius[[0, 1, 6, 7]] = float(
        uncertainty["per_agent_position_error_bound_m_per_axis"]
    )
    radius[[3, 4, 8, 9]] = float(
        uncertainty["per_agent_velocity_error_bound_mps_per_axis"]
    )
    return center, radius, local, neighbor


def _reason_index(reason: str) -> int:
    if reason.startswith("alignment_evidence_refused"):
        return REASON_CODEBOOK.index("alignment_evidence_refused")
    try:
        return REASON_CODEBOOK.index(reason)
    except ValueError as error:
        raise CircaGazeboGZ0V9RunnerError(f"unregistered intervention reason: {reason}") from error


def _environment(
    driver: str,
    world: Path,
    candidate: SceneCandidate,
    design: Mapping[str, Any],
):
    envelope = derive_operational_envelope(
        candidate, design["operational_envelope_assumptions"]
    )
    kwargs = {"horizon": int(design["horizon_steps"])}
    if driver == "planar_speed_projected_v4":
        limit = 0.5 * envelope.design_relative_speed_mps
        return GazeboAirGroundEnvV4(
            world,
            candidate.shift(),
            uav_planar_speed_limit_mps=limit,
            ugv_planar_speed_limit_mps=limit,
            **kwargs,
        )
    if driver == "command_persistent_unbounded_v3":
        return GazeboAirGroundEnvV3(world, candidate.shift(), **kwargs)
    raise CircaGazeboGZ0V9RunnerError("unknown frozen driver")


def _write_step_trace(
    arrays: dict[str, np.ndarray],
    row: int,
    step: int,
    *,
    candidate_index: int,
    aligned: Any,
    source_center: np.ndarray,
    source_radius: np.ndarray,
    local_source: int,
    neighbor_source: int,
    action: np.ndarray,
    reason: str,
    fail_closed: bool,
    decision: Any | None,
) -> None:
    arrays["source_timestamps"][candidate_index, step] = (local_source, neighbor_source)
    arrays["decision_timestamp"][step] = step
    arrays["local_age_steps"][candidate_index, step] = aligned.local_age_steps
    arrays["neighbor_age_steps"][candidate_index, step] = aligned.neighbor_age_steps
    arrays["applied_action_history_digest"][row, step] = _binary_hash(
        aligned.applied_action_history_digest
    )
    arrays["source_center_radius_set"][row, step, 0] = source_center
    arrays["source_center_radius_set"][row, step, 1] = source_radius
    arrays["aligned_center_radius_set"][row, step, 0] = aligned.center
    arrays["aligned_center_radius_set"][row, step, 1] = aligned.radius
    arrays["uncertainty_registry_hash"][:] = _binary_hash(
        aligned.uncertainty_registry_hash
    )
    arrays["selected_action"][row, step] = action
    arrays["intervention_reason"][row, step] = _reason_index(reason)
    arrays["method_decision_reason"][row, step] = _reason_index(reason)
    arrays["backup_tube_infeasible_fail_closed"][row, step] = bool(fail_closed)
    arrays["provenance_hash"][row, step] = _binary_hash(aligned.provenance_hash)
    arrays["trace_valid"][row, step] = True
    if decision is None:
        return
    arrays["refusal_code"][row, step] = (
        0 if decision.certificate_emitted else REFUSAL_CODEBOOK.index("invalid_alignment_evidence")
    )
    if not decision.certificate_emitted:
        return
    nominal = decision.nominal_plan
    backup = decision.backup_plan
    arrays["nominal_tube_feasible"][row, step] = int(nominal.feasible)
    arrays["backup_tube_feasible"][row, step] = int(backup.feasible)
    selected = backup if decision.intervened else nominal
    arrays["worst_set_margin"][row, step] = selected.minimum_tightened_margin_m
    arrays["terminal_reachability"][row, step] = int(selected.terminal_reached)
    arrays["certificate_validity_interval"][step] = decision.certificate[
        "validity_interval_steps"
    ]


def _run_one(
    spec: FactorialRun,
    row: int,
    arrays: dict[str, np.ndarray],
    candidates: tuple[SceneCandidate, ...],
    world: Path,
    design: Mapping[str, Any],
) -> None:
    candidate = candidates[spec.candidate_index]
    driver = DRIVERS[spec.driver_index]
    method = METHODS[spec.method_index]
    env = _environment(driver, world, candidate, design)
    scenario = _scenario(candidate, dict(design), hazard_active=spec.hazard_active)
    env.reset(
        seed=spec.scenario_seed,
        initial_state=scenario.initial_state,
        position_jitter_scale=candidate.position_jitter_scale,
    )
    set_filter = build_filter(candidate, dict(design))
    point_filter = set_filter._point_filter
    cbf = SandboxNominalCBFAdapter(scenario.task)
    states = [env.state.copy()]
    applied = [env._applied_action.copy()]
    envelope = derive_operational_envelope(
        candidate, design["operational_envelope_assumptions"]
    )
    design_speed_limit = float(envelope.design_relative_speed_mps) + float(
        design["design_speed_tolerance_mps"]
    )
    arrays["design_speed_limit_mps"][row] = design_speed_limit
    envelope_payload = json.dumps(
        {
            "candidate_id": candidate.candidate_id,
            "operational_envelope_assumptions": design["operational_envelope_assumptions"],
            "design_speed_tolerance_mps": design["design_speed_tolerance_mps"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    arrays["envelope_registry_hash"][spec.candidate_index] = np.frombuffer(
        hashlib.sha256(envelope_payload).digest(), dtype=np.uint8
    )
    arrays["decision_codebook_hash"][:] = np.frombuffer(
        hashlib.sha256(
            json.dumps(REASON_CODEBOOK, separators=(",", ":")).encode("utf-8")
        ).digest(),
        dtype=np.uint8,
    )
    minimum_operational = float(
        np.linalg.norm(env.state[:2] - env.state[6:8])
        - envelope.operational_separation_m
    )
    minimum_hard = constraint_margins(env.state, 0).coupling
    operational_violation = minimum_operational < 0.0
    hard_violation = minimum_hard < 0.0
    intervened_any = False

    for step in range(int(design["horizon_steps"])):
        aligned = align_async_state_history(
            states,
            applied,
            candidate.observation_delay_steps,
            candidate.communication_delay_steps,
            set_filter.plant,
            set_filter.alignment,
            observed_common_mode_bias_m=candidate.sensor_bias,
        )
        source_center, source_radius, local_source, neighbor_source = _source_center_radius(
            states, candidate, step, design
        )
        stale_observed = source_center.copy()
        stale_applied = applied[local_source]
        stale_nominal = scenario.task.nominal_action(
            np.concatenate([stale_observed, stale_applied])
        )
        aligned_nominal = scenario.task.nominal_action(
            np.concatenate([aligned.center, applied[-1]])
        )
        decision = None
        changed = False
        fail_closed = False
        if method == "shadow_no_override":
            action = stale_nominal
            reason = "shadow_no_override"
        elif method == "registered_one_step_cbf":
            fallback = set_filter.backup_action(stale_observed, stale_nominal)
            try:
                action = cbf.decide(
                    np.concatenate([stale_observed, stale_applied]),
                    fallback_action=fallback,
                ).action
                reason = "registered_one_step_cbf"
            except (SandboxBaselineInfeasible, ValueError):
                action = fallback
                reason = "registered_one_step_cbf_fallback"
            changed = not np.allclose(action, stale_nominal)
        elif method == "robust_backup_filter_v7_stale_point":
            stale_decision = point_filter.decide(
                stale_observed, stale_applied, stale_nominal
            )
            action = stale_decision.action
            reason = stale_decision.reason
            changed = stale_decision.intervened
            fail_closed = stale_decision.fail_closed
        else:
            decision = set_filter.decide(
                aligned,
                applied[-1],
                aligned_nominal,
                point_ablation=(method == "timestamp_aligned_point_backup_v8_ablation"),
            )
            action = decision.action
            reason = decision.reason
            changed = decision.intervened
            fail_closed = decision.fail_closed
        intervened_any |= bool(changed)
        _write_step_trace(
            arrays,
            row,
            step,
            candidate_index=spec.candidate_index,
            aligned=aligned,
            source_center=source_center,
            source_radius=source_radius,
            local_source=local_source,
            neighbor_source=neighbor_source,
            action=np.asarray(action, dtype=float),
            reason=reason,
            fail_closed=fail_closed,
            decision=decision,
        )
        _, _, _, truncated, info = env.step(action)
        states.append(env.state.copy())
        applied.append(env._applied_action.copy())
        velocities = np.stack((env.state[3:5], env.state[8:10]))
        arrays["applied_planar_velocity"][row, step] = velocities
        relative_speed = float(np.linalg.norm(velocities[0] - velocities[1]))
        arrays["design_speed_envelope_exceeded"][row, step] = bool(
            relative_speed > design_speed_limit
        )
        arrays["completed_step_mask"][row, step] = True
        state_bytes = np.asarray(env.state, dtype="<f8").tobytes()
        arrays["simulator_state_hash"][row, step] = np.frombuffer(
            hashlib.sha256(state_bytes).digest(), dtype=np.uint8
        )
        operational = float(
            np.linalg.norm(env.state[:2] - env.state[6:8])
            - envelope.operational_separation_m
        )
        minimum_operational = min(minimum_operational, operational)
        minimum_hard = min(minimum_hard, info["margins"].coupling)
        operational_violation |= operational < 0.0
        hard_violation |= info["margins"].coupling < 0.0
        if truncated:
            break
    arrays["candidate_index"][row] = spec.candidate_index
    arrays["scenario_index"][row] = spec.scenario_index
    arrays["scenario_seed"][row] = spec.scenario_seed
    arrays["driver_index"][row] = spec.driver_index
    arrays["method_index"][row] = spec.method_index
    arrays["hazard_active"][row] = spec.hazard_active
    arrays["operational_first_violation"][row] = operational_violation
    arrays["hard_first_violation"][row] = hard_violation
    arrays["applied_intervention"][row] = intervened_any
    arrays["completed_steps"][row] = env.step_index
    arrays["minimum_operational_margin_m"][row] = minimum_operational
    arrays["minimum_hard_margin_m"][row] = minimum_hard


def _result_summary(
    manifest_sha256: str,
    arrays: Mapping[str, np.ndarray],
    elapsed_seconds: float,
    archive_bytes: int,
) -> bytes:
    cells = []
    for driver_index, driver in enumerate(DRIVERS):
        for method_index, method in enumerate(METHODS):
            mask = (arrays["driver_index"] == driver_index) & (
                arrays["method_index"] == method_index
            )
            cells.append(
                {
                    "driver": driver,
                    "method": method,
                    "n": int(np.sum(mask)),
                    "operational_first_violation_rate": float(
                        np.mean(arrays["operational_first_violation"][mask])
                    ),
                    "hard_first_violation_rate": float(
                        np.mean(arrays["hard_first_violation"][mask])
                    ),
                    "intervention_rate": float(
                        np.mean(arrays["applied_intervention"][mask])
                    ),
                }
            )
    payload = json.loads(
        summary_json_bytes(
            rollouts=int(arrays["candidate_index"].size),
            horizon=int(arrays["trace_valid"].shape[1]),
            array_bytes=archive_bytes,
            schema_only=False,
        )
    )
    payload.update(
        {
            "manifest_sha256": manifest_sha256,
            "elapsed_seconds": elapsed_seconds,
            "cells": cells,
            "scientific_run_executed": True,
            "scientific_seed_material_generated": True,
            "claim_ceiling": "claim_ineligible_gz0_mechanism_screen_only",
        }
    )
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def run_gz0_v9(manifest_path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    path = Path(manifest_path).resolve()
    if root not in path.parents or not path.is_file():
        raise CircaGazeboGZ0V9RunnerError("unsafe or absent runnable manifest")
    raw = path.read_bytes()
    manifest = json.loads(raw)
    design, candidates, world = _validate_runnable_manifest(manifest, root)
    schedule = compile_schedule(manifest)
    arrays = allocate_arrays(len(schedule), int(design["horizon_steps"]))
    start = time.perf_counter()
    for row, spec in enumerate(schedule):
        if time.perf_counter() - start > float(design["wall_time_seconds_max"]):
            raise CircaGazeboGZ0V9RunnerError("v9 wall-time budget exceeded")
        _run_one(spec, row, arrays, candidates, world, design)
    elapsed = time.perf_counter() - start
    validate_arrays(arrays, len(schedule), int(design["horizon_steps"]))
    archive = encode_lossless_arrays(arrays)
    verify_lossless_arrays(archive, arrays)
    summary = _result_summary(hashlib.sha256(raw).hexdigest(), arrays, elapsed, len(archive))
    total = len(archive) + len(summary)
    if total > int(design["output_bytes_max"]):
        raise CircaGazeboGZ0V9RunnerError("v9 lossless evidence exceeds output budget")
    output = _safe_child(root, manifest["output_path"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_bytes(summary)
    (output / "trace_arrays.npz").write_bytes(archive)
    return {
        "status": "COMPLETE_CLAIM_INELIGIBLE_GZ0_V9",
        "rollouts": len(schedule),
        "elapsed_seconds": elapsed,
        "output_bytes": total,
        "claim_eligible": False,
        "circa_gz1_run": False,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--validate-schema", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.validate_schema == args.run:
        raise SystemExit("select exactly one of --validate-schema or --run")
    if args.run:
        print(json.dumps(run_gz0_v9(args.manifest, args.repo_root), sort_keys=True))
        return
    root = Path(args.repo_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    if root not in manifest_path.parents or not manifest_path.is_file():
        raise SystemExit("unsafe or absent schema manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    design = validate_schema_only_manifest(manifest, root)
    arrays = schema_fixture_arrays(2, 3, stress=False)
    validate_arrays(arrays, 2, 3)
    print(
        json.dumps(
            {
                "status": "PASS_NONRUNNING_SCHEMA_CONSTRUCTION",
                "trace_required_field_count": len(TRACE_REQUIRED_FIELDS),
                "archive_member_count": len(arrays),
                "scientific_seed_material_generated": False,
                "scientific_output_generated": False,
                "schema_capacity_audit_performed": False,
                "prospective_rollouts": int(design["max_regime_rollouts"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
