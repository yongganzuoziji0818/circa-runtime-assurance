"""Exactly-once scientific runner for the frozen CIRCA-RESS-V8 diversity route.

This module is inert unless it receives a runnable manifest carrying all
source, queue, capacity, seed, target-absence, and authorization locks.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np

from .circa_ress_v8_sim_diversity_adapter import (
    DiversityRuntimeAdapter,
    load_frozen_design,
    scenario_from_design,
    world_patch_for,
)
from .circa_ress_v8_sim_diversity_schema import (
    CANDIDATE_CODEBOOK,
    DRIVER_CODEBOOK,
    FAMILY_CODEBOOK,
    FAULT_CODEBOOK,
    METHOD_CODEBOOK,
    REFUSAL_CODEBOOK,
    SPLIT_CODEBOOK,
    array_schema,
    codebook_hash,
)
from .contracts import ActionEnvelope
from .environment import CompoundShift
from .gazebo_robust_backup_filter import (
    GazeboPlanarPlant,
    RobustBackupConfig,
    RobustBackupSafetyFilter,
)
from .gazebo_second_system_v3 import GazeboAirGroundEnvV3
from .gazebo_second_system_v4 import GazeboAirGroundEnvV4
from .gazebo_timestamp_aligned_set_filter import (
    TimestampAlignmentConfig,
    TimestampAlignmentError,
    align_async_state_history,
)
from .sandbox_baselines import SandboxBaselineInfeasible, SandboxNominalCBFAdapter
from .sandbox_task import SandboxComparisonTask


ROUTE_ID = "CIRCA-RESS-V8-SIM-DIVERSITY-R1"
SEED_NAMESPACE = "circa-ress-v8-sim-diversity-r1"
DESIGN_SHA256 = "f84325ab6d901d2f03f37f2e6b34ebba570d513c8f8b8f29a4581bf9d363aaa9"
OPERATIONALIZATION_SHA256 = (
    "ab765f8ea28982ace2930d84f96f92cbd8986bc3191a7866a7468ace00c57c0f"
)
WORLD_REGISTRY_SHA256 = (
    "c3cf8c4cf5fa812560f1f70b030521bfd6d33cbfc252e408f515c8de0955eb72"
)
CAPACITY_RECEIPT_SHA256 = (
    "54e7cf3c754e3a8471fe19078359873d5f6e41cb15c13c42db51747cdef609bb"
)
OUTPUT_CAPACITY_BYTES = 268_435_456
HORIZON = 80
ROLLOUTS = 6400
VALIDATION_METHODS = (
    "shadow_no_override",
    "timestamp_aligned_set_backup_v8",
)
REASON_CODEBOOK = (
    "not_evaluated",
    "shadow_no_override",
    "registered_one_step_cbf",
    "registered_one_step_cbf_fallback",
    "nominal_with_verified_backup_tube",
    "backup_filter_intervention",
    "backup_tube_infeasible_fail_closed",
    "nominal_with_verified_aligned_set_backup_tube",
    "aligned_set_backup_filter_intervention",
    "aligned_set_backup_tube_infeasible_fail_closed",
    "typed_refusal_invalid_alignment_evidence",
    "typed_refusal_invalid_provenance",
    "typed_refusal_invalid_monotonic_time",
    "typed_refusal_expired_action",
    "typed_refusal_invalid_action_contract",
)


class DiversityRunnerError(RuntimeError):
    """Fail-closed runner error."""


@dataclass(frozen=True)
class RunSpec:
    family_index: int
    candidate_index: int
    split_index: int
    seed_index: int
    future_seed: int
    driver_index: int
    method_index: int
    pair_id: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _hash_bytes(value: str) -> np.ndarray:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise DiversityRunnerError("invalid lowercase SHA-256")
    return np.frombuffer(bytes.fromhex(value), dtype=np.uint8)


def _safe_child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise DiversityRunnerError(f"path escapes repository: {relative}")
    return path


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def derive_future_seed(
    master_seed: int,
    split: str,
    family: str,
    candidate: str,
    index: int,
) -> int:
    if (
        not isinstance(master_seed, int)
        or master_seed < 0
        or split not in SPLIT_CODEBOOK
        or family not in FAMILY_CODEBOOK
        or candidate not in CANDIDATE_CODEBOOK
        or not isinstance(index, int)
        or index < 0
    ):
        raise DiversityRunnerError("invalid future seed derivation input")
    payload = f"{SEED_NAMESPACE}|{master_seed}|{split}|{family}|{candidate}|{index}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (
        2**31 - 1
    )


def compile_schedule(manifest: Mapping[str, Any]) -> list[RunSpec]:
    master_seed = manifest.get("master_seed")
    schedule_seed = manifest.get("schedule_seed")
    if not isinstance(master_seed, int) or not isinstance(schedule_seed, int):
        raise DiversityRunnerError("scientific seed material is absent")
    schedule: list[RunSpec] = []
    pair_id = 0
    seen: set[tuple[str, str, str, int]] = set()
    for split_index, split in enumerate(SPLIT_CODEBOOK):
        count = 32 if split == "validation" else 64
        methods = VALIDATION_METHODS if split == "validation" else METHOD_CODEBOOK
        for family_index, family in enumerate(FAMILY_CODEBOOK):
            for candidate_index, candidate in enumerate(CANDIDATE_CODEBOOK):
                for seed_index in range(count):
                    seed = derive_future_seed(
                        master_seed, split, family, candidate, seed_index
                    )
                    key = (split, family, candidate, seed)
                    if key in seen:
                        raise DiversityRunnerError("future seed collision")
                    seen.add(key)
                    for driver_index in range(len(DRIVER_CODEBOOK)):
                        for method in methods:
                            schedule.append(
                                RunSpec(
                                    family_index=family_index,
                                    candidate_index=candidate_index,
                                    split_index=split_index,
                                    seed_index=seed_index,
                                    future_seed=seed,
                                    driver_index=driver_index,
                                    method_index=METHOD_CODEBOOK.index(method),
                                    pair_id=pair_id,
                                )
                            )
                    pair_id += 1
    if len(schedule) != ROLLOUTS or pair_id != 960 or len(seen) != 960:
        raise DiversityRunnerError("frozen paired schedule dimensions drifted")
    random.Random(schedule_seed).shuffle(schedule)
    return schedule


def allocate_arrays() -> dict[str, np.ndarray]:
    arrays = {
        name: np.zeros(shape, dtype=dtype)
        for name, (dtype, shape) in array_schema(ROLLOUTS, HORIZON).items()
    }
    for name in (
        "nominal_tube_feasible",
        "backup_tube_feasible",
        "terminal_reachability",
    ):
        arrays[name].fill(-1)
    for name in (
        "operational_margin_m",
        "hard_margin_m",
        "minimum_operational_margin_m",
        "minimum_hard_margin_m",
    ):
        arrays[name].fill(np.nan)
    return arrays


def _validate_scientific_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    expected = array_schema(ROLLOUTS, HORIZON)
    if set(arrays) != set(expected):
        raise DiversityRunnerError("array member set drifted")
    for name, (dtype, shape) in expected.items():
        value = np.asarray(arrays[name])
        if value.dtype != dtype or value.shape != shape or value.dtype.hasobject:
            raise DiversityRunnerError(f"array schema drifted: {name}")
    if np.any(arrays["future_seed_sentinel"] < 0):
        raise DiversityRunnerError("scientific seed field is incomplete")
    if np.any(arrays["completed_steps"] != HORIZON):
        raise DiversityRunnerError("one or more rollouts are incomplete")
    if not np.all(arrays["completed_step_mask"]):
        raise DiversityRunnerError("step evidence is incomplete")
    typed = np.broadcast_to(arrays["typed_refusal"][:, None], (ROLLOUTS, HORIZON))
    if np.any(arrays["fail_closed"] & ~typed):
        raise DiversityRunnerError("fail-closed evidence lacks typed refusal")
    if np.any(arrays["family_index"] >= len(FAMILY_CODEBOOK)):
        raise DiversityRunnerError("family index exceeds codebook")
    if np.any(arrays["candidate_index"] >= len(CANDIDATE_CODEBOOK)):
        raise DiversityRunnerError("candidate index exceeds codebook")
    if np.any(arrays["split_index"] >= len(SPLIT_CODEBOOK)):
        raise DiversityRunnerError("split index exceeds codebook")
    if np.any(arrays["driver_index"] >= len(DRIVER_CODEBOOK)):
        raise DiversityRunnerError("driver index exceeds codebook")
    if np.any(arrays["method_index"] >= len(METHOD_CODEBOOK)):
        raise DiversityRunnerError("method index exceeds codebook")


def validate_runnable_manifest(
    manifest: Mapping[str, Any], repo_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path]:
    required = {
        "route_id": ROUTE_ID,
        "status": "AUTHORIZED_EXACTLY_ONCE_SCIENTIFIC_RUN",
        "scientific_run_authorized": True,
        "exactly_once_authorization": True,
        "scientific_seed_materialized": True,
        "scientific_output_authorized": True,
        "scientific_claim_allowed_before_analysis": False,
        "retry_allowed": False,
        "seed_top_up_allowed": False,
        "sample_top_up_allowed": False,
        "maximum_workers": 1,
        "seed_namespace": SEED_NAMESPACE,
        "rollouts": ROLLOUTS,
        "horizon_steps": HORIZON,
        "output_capacity_bytes": OUTPUT_CAPACITY_BYTES,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise DiversityRunnerError(f"runnable manifest drifted: {key}")
    locks = (
        ("design_path", "design_sha256", DESIGN_SHA256),
        (
            "operationalization_path",
            "operationalization_sha256",
            OPERATIONALIZATION_SHA256,
        ),
        (
            "world_registry_path",
            "world_registry_sha256",
            WORLD_REGISTRY_SHA256,
        ),
        (
            "capacity_receipt_path",
            "capacity_receipt_sha256",
            CAPACITY_RECEIPT_SHA256,
        ),
        ("seed_receipt_path", "seed_receipt_sha256", None),
        ("queue_snapshot_path", "queue_snapshot_sha256", None),
        ("preflight_receipt_path", "preflight_receipt_sha256", None),
    )
    loaded: dict[str, dict[str, Any]] = {}
    for path_key, hash_key, fixed_hash in locks:
        path = _safe_child(repo_root, str(manifest.get(path_key, "")))
        expected_hash = fixed_hash or str(manifest.get(hash_key, ""))
        if not path.is_file() or sha256(path) != expected_hash:
            raise DiversityRunnerError(f"manifest lock failed: {path_key}")
        loaded[path_key] = json.loads(path.read_text(encoding="utf-8"))
    if loaded["preflight_receipt_path"].get("status") != (
        "PASS_NONSCIENTIFIC_REMOTE_PREFLIGHT"
    ):
        raise DiversityRunnerError("non-scientific preflight did not pass")
    if loaded["seed_receipt_path"].get("status") != "FROZEN_FUTURE_SEEDS":
        raise DiversityRunnerError("future seed receipt did not freeze")
    if loaded["queue_snapshot_path"].get("p4_admission_status") != (
        "ADMITTED_CPU_SHARED_EXACTLY_ONCE"
    ):
        raise DiversityRunnerError("queue snapshot does not admit this route")
    if manifest["master_seed"] != loaded["seed_receipt_path"].get("master_seed"):
        raise DiversityRunnerError("master seed does not match seed receipt")
    if manifest["schedule_seed"] != loaded["seed_receipt_path"].get("schedule_seed"):
        raise DiversityRunnerError("schedule seed does not match seed receipt")
    for relative, expected_hash in manifest.get("source_hashes", {}).items():
        path = _safe_child(repo_root, relative)
        if not path.is_file() or sha256(path) != expected_hash:
            raise DiversityRunnerError(f"source hash lock failed: {relative}")
    design = loaded["design_path"]
    operationalization = loaded["operationalization_path"]
    registry = loaded["world_registry_path"]
    load_frozen_design(_safe_child(repo_root, manifest["design_path"]))
    if registry.get("variant_count") != 10:
        raise DiversityRunnerError("world registry does not contain ten variants")
    output = _safe_child(repo_root, str(manifest.get("output_path", "")))
    state = _safe_child(repo_root, str(manifest.get("runtime_state_path", "")))
    if output.exists():
        raise DiversityRunnerError("exactly-once scientific target already exists")
    if state.exists():
        raise DiversityRunnerError("runtime state target already exists")
    compile_schedule(manifest)
    return design, operationalization, registry, output, state


def _counter_uniform(seed: int, family: str, candidate: str, step: int, tag: str) -> float:
    body = f"{SEED_NAMESPACE}|{seed}|{family}|{candidate}|{step}|{tag}".encode()
    integer = int.from_bytes(hashlib.sha256(body).digest()[:8], "big")
    return integer / float(2**64)


def _stress_plan(
    seed: int, family: str, candidate: str, patch: Mapping[str, Any]
) -> tuple[list[int], list[int], list[bool]]:
    packet_loss = float(patch["packet_loss"])
    jitter_max = int(patch["timestamp_jitter_steps"])
    losses: list[bool] = []
    jitters: list[int] = []
    streak = 0
    streaks: list[int] = []
    for step in range(HORIZON):
        loss = _counter_uniform(seed, family, candidate, step, "loss") < packet_loss
        streak = streak + 1 if loss else 0
        losses.append(loss)
        streaks.append(streak)
        if jitter_max:
            jitter = int(
                math.floor(
                    _counter_uniform(seed, family, candidate, step, "jitter")
                    * (jitter_max + 1)
                )
            )
        else:
            jitter = 0
        jitters.append(min(jitter, jitter_max))
    return jitters, streaks, losses


def _shift_and_task(
    operationalization: Mapping[str, Any], patch: Mapping[str, Any]
) -> tuple[CompoundShift, SandboxComparisonTask, float]:
    task = operationalization["task"]
    dynamics = operationalization["dynamics_and_observation"]
    parameters = patch["scenario_parameters"]
    transition = parameters.get("friction_transition")
    friction = float(transition[1]) if transition is not None else float(
        dynamics["base_ugv_friction"]
    )
    shift = CompoundShift(
        uav_mass=float(dynamics["base_uav_mass"]),
        uav_drag=float(dynamics["base_uav_drag"]),
        ugv_friction=friction,
        actuator_lag=0.0,
        sensor_bias=0.0,
    )
    comparison = SandboxComparisonTask(
        shift=shift,
        uav_goal=tuple(task["uav_goal"]),
        ugv_goal=tuple(task["ugv_goal"]),
        uav_position_gain=float(task["uav_position_gain"]),
        uav_velocity_gain=float(task["uav_velocity_gain"]),
        ugv_position_gain=float(task["ugv_position_gain"]),
        ugv_velocity_gain=float(task["ugv_velocity_gain"]),
        minimum_separation=float(task["hard_separation_m"]),
        action_limit=float(task["action_limit"]),
    )
    return shift, comparison, friction


def _filters(
    operationalization: Mapping[str, Any], friction: float
) -> tuple[DiversityRuntimeAdapter, RobustBackupSafetyFilter]:
    method = operationalization["method_binding"]
    dynamics = operationalization["dynamics_and_observation"]
    task = operationalization["task"]
    plant = GazeboPlanarPlant(
        uav_mass=float(dynamics["base_uav_mass"]),
        uav_drag=float(dynamics["base_uav_drag"]),
        ugv_friction=friction,
        actuator_lag=0.0,
        dt=0.1,
        speed_limit_mps=float(task["action_limit"]),
    )
    backup = RobustBackupConfig(
        operational_separation_m=float(task["hard_separation_m"]),
        action_limit=float(task["action_limit"]),
        horizon_steps=int(method["backup_horizon_steps"]),
        terminal_margin_m=float(method["terminal_margin_m"]),
        position_error_bound_m=float(method["position_error_bound_m"]),
        velocity_error_bound_mps=float(method["velocity_error_bound_mps"]),
        relative_acceleration_error_bound_mps2=float(
            method["relative_acceleration_error_bound_mps2"]
        ),
    )
    return (
        DiversityRuntimeAdapter(actuator_lag=0.0, friction=friction),
        RobustBackupSafetyFilter(plant, backup),
    )


def _environment(
    driver: str,
    world: Path,
    shift: CompoundShift,
    action_limit: float,
):
    if driver == "command_persistent_unbounded_v3":
        return GazeboAirGroundEnvV3(world, shift, horizon=HORIZON)
    if driver == "planar_speed_projected_v4":
        return GazeboAirGroundEnvV4(
            world,
            shift,
            horizon=HORIZON,
            uav_planar_speed_limit_mps=action_limit,
            ugv_planar_speed_limit_mps=action_limit,
        )
    raise DiversityRunnerError("unknown driver")


def _scenario_entry(
    registry: Mapping[str, Any], family: str, candidate: str
) -> Mapping[str, Any]:
    matches = [
        value
        for value in registry["variants"]
        if value["family"] == family and value["candidate"] == candidate
    ]
    if len(matches) != 1:
        raise DiversityRunnerError("world registry scenario is not unique")
    return matches[0]


def _source_set(
    states: list[np.ndarray],
    local_age: int,
    communication_age: int,
    sensor_bias: float,
    alignment: TimestampAlignmentConfig,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    step = len(states) - 1
    local_source = max(0, step - local_age)
    neighbor_source = max(0, step - local_age - communication_age)
    center = states[local_source].copy()
    center[6:10] = states[neighbor_source][6:10]
    center[[0, 1, 6, 7]] += sensor_bias
    radius = np.zeros(10, dtype=float)
    radius[[0, 1, 6, 7]] = alignment.per_agent_position_error_bound_m
    radius[[3, 4, 8, 9]] = alignment.per_agent_velocity_error_bound_mps
    return center, radius, local_source, neighbor_source


def _operational_margin(
    state: np.ndarray,
    family: str,
    parameters: Mapping[str, Any],
    operationalization: Mapping[str, Any],
) -> tuple[float, float]:
    task = operationalization["task"]
    separation = float(np.linalg.norm(state[:2] - state[6:8]))
    hard = separation - float(task["hard_separation_m"])
    margins = [hard]
    if family in ("SDF1", "SDF4"):
        aisle = float(parameters["aisle_width_m"])
        free_half = aisle / 2.0 - float(task["wall_half_thickness_m"])
        margins.append(
            free_half - float(task["uav_planar_radius_m"]) - abs(float(state[1]))
        )
        margins.append(
            free_half
            - float(task["ugv_planar_half_width_m"])
            - abs(float(state[7]))
        )
    elif family == "SDF2":
        free_half = 1.0
        margins.append(
            free_half - float(task["uav_planar_radius_m"]) - abs(float(state[1]))
        )
        margins.append(
            free_half
            - float(task["ugv_planar_half_width_m"])
            - abs(float(state[7]))
        )
    return min(margins), hard


def _reason_index(reason: str, refusal: str) -> int:
    if refusal != "none":
        typed = f"typed_refusal_{refusal}"
        if typed in REASON_CODEBOOK:
            return REASON_CODEBOOK.index(typed)
        return REASON_CODEBOOK.index("typed_refusal_invalid_alignment_evidence")
    if reason in REASON_CODEBOOK:
        return REASON_CODEBOOK.index(reason)
    raise DiversityRunnerError(f"unregistered decision reason: {reason}")


def _fault_index(family: str, faults: list[str], loss: bool) -> int:
    if family == "SDF5":
        text = " ".join(faults)
        if "provenance" in text:
            return FAULT_CODEBOOK.index("evidence_provenance_hash_mismatch")
        if "monotonic" in text:
            return FAULT_CODEBOOK.index("monotonic_time_reversal")
        if "expired" in text:
            return FAULT_CODEBOOK.index("action_expired")
    if loss:
        return FAULT_CODEBOOK.index("packet_omission")
    if family in ("SDF2", "SDF4"):
        return FAULT_CODEBOOK.index("compound_ood")
    return FAULT_CODEBOOK.index("none")


def _write_hash(array: np.ndarray, value: str) -> None:
    array[...] = _hash_bytes(value)


def _run_one(
    spec: RunSpec,
    row: int,
    arrays: dict[str, np.ndarray],
    repo_root: Path,
    design: Mapping[str, Any],
    operationalization: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> None:
    family = FAMILY_CODEBOOK[spec.family_index]
    candidate = CANDIDATE_CODEBOOK[spec.candidate_index]
    split = SPLIT_CODEBOOK[spec.split_index]
    driver = DRIVER_CODEBOOK[spec.driver_index]
    method = METHOD_CODEBOOK[spec.method_index]
    entry = _scenario_entry(registry, family, candidate)
    world = _safe_child(repo_root, entry["path"])
    if sha256(world) != entry["sha256"]:
        raise DiversityRunnerError("world variant hash mismatch")
    patch = entry["world_patch"]
    parameters = entry["scenario_parameters"]
    shift, task, friction = _shift_and_task(operationalization, entry)
    independent, point_filter = _filters(operationalization, friction)
    cbf = SandboxNominalCBFAdapter(task)
    action_limit = float(operationalization["task"]["action_limit"])
    env = _environment(driver, world, shift, action_limit)
    env.reset(
        seed=spec.future_seed,
        initial_state=np.asarray(
            operationalization["task"]["initial_state"], dtype=float
        ),
        position_jitter_scale=float(
            operationalization["task"]["position_jitter_standard_deviation_m"]
        ),
    )
    states = [env.state.copy()]
    applied = [env._applied_action.copy()]
    jitters, loss_streaks, losses = _stress_plan(
        spec.future_seed, family, candidate, patch
    )
    lag_steps = int(patch["actuation_lag_steps"])
    lag_queue = [np.zeros(5, dtype=float) for _ in range(lag_steps)]
    alignment = TimestampAlignmentConfig()
    initial_operational, initial_hard = _operational_margin(
        env.state, family, parameters, operationalization
    )
    if family != "SDF5" and initial_operational < (
        float(design["common_contract"]["minimum_initial_operational_margin_m"])
        - 1e-9
    ):
        raise DiversityRunnerError("initial operational margin violates frozen floor")
    minimum_operational = initial_operational
    minimum_hard = initial_hard
    operational_violation = initial_operational < 0.0
    hard_violation = initial_hard < 0.0
    intervened_any = False
    typed_refusal_any = False
    positive_certificate_before_violation = False
    scenario_digest = _canonical_hash(
        {
            "entry": entry,
            "split": split,
            "seed": spec.future_seed,
            "stress_plan": {
                "timestamp_jitter": jitters,
                "packet_loss": losses,
                "loss_streak": loss_streaks,
            },
        }
    )

    for step in range(HORIZON):
        local_age = int(patch["observation_delay_steps"]) + jitters[step]
        communication_age = int(patch["communication_delay_steps"]) + loss_streaks[step]
        sensor_bias = float(patch["sensor_bias_m"])
        source_center, source_radius, local_source, neighbor_source = _source_set(
            states,
            local_age,
            communication_age,
            sensor_bias,
            alignment,
        )
        stale_applied = applied[local_source]
        nominal = task.nominal_action(
            np.concatenate([source_center, stale_applied])
        )
        refusal = "none"
        fail_closed = False
        certificate_emitted = False
        nominal_feasible = -1
        backup_feasible = -1
        terminal_reached = -1
        aligned_center = source_center.copy()
        aligned_radius = source_radius.copy()
        provenance = _canonical_hash(
            {
                "scenario": scenario_digest,
                "step": step,
                "local_source": local_source,
                "neighbor_source": neighbor_source,
                "source_center": source_center.tolist(),
                "source_radius": source_radius.tolist(),
            }
        )
        try:
            aligned = align_async_state_history(
                states,
                applied,
                local_age,
                communication_age,
                independent._filter.plant,
                independent._filter.alignment,
                observed_common_mode_bias_m=sensor_bias,
            )
            aligned_center = aligned.center
            aligned_radius = aligned.radius
            action_history_digest = aligned.applied_action_history_digest
            uncertainty_hash = aligned.uncertainty_registry_hash
            provenance = aligned.provenance_hash
        except (TimestampAlignmentError, ValueError):
            action_history_digest = _canonical_hash(
                [value.tolist() for value in applied]
            )
            uncertainty_hash = _canonical_hash(alignment.registry_payload())

        if method == "shadow_no_override":
            selected = nominal
            reason = "shadow_no_override"
            intervened = False
        elif method == "registered_one_step_cbf":
            fallback = point_filter.backup_action(source_center, nominal)
            try:
                selected = cbf.decide(
                    np.concatenate([source_center, stale_applied]),
                    fallback_action=fallback,
                ).action
                reason = "registered_one_step_cbf"
            except (SandboxBaselineInfeasible, ValueError):
                selected = fallback
                reason = "registered_one_step_cbf_fallback"
            intervened = not np.allclose(selected, nominal)
        elif method == "robust_backup_filter_v7_stale_point":
            decision = point_filter.decide(
                source_center, stale_applied, nominal
            )
            selected = decision.action
            reason = decision.reason
            intervened = decision.intervened
            fail_closed = decision.fail_closed
            nominal_feasible = int(decision.nominal_plan.feasible)
            backup_feasible = int(decision.backup_plan.feasible)
            terminal_reached = int(
                (
                    decision.backup_plan
                    if decision.intervened
                    else decision.nominal_plan
                ).terminal_reached
            )
        elif method == "timestamp_aligned_set_backup_v8":
            issued_at = step * 0.1
            valid_until = issued_at + 0.1
            now = issued_at
            provenance_valid = True
            monotonic_valid = True
            faults = list(patch["registered_faults"])
            if family == "SDF5" and candidate == "A":
                provenance_valid = False
            elif family == "SDF5" and candidate == "B":
                if step % 2 == 0:
                    monotonic_valid = False
                else:
                    valid_until = max(1e-9, issued_at)
                    issued_at = max(0.0, valid_until - 0.1)
                    now = valid_until + 0.1
            envelope = ActionEnvelope(
                action=np.asarray(nominal, dtype=float),
                issued_at=issued_at,
                valid_until=valid_until,
                source="circa_ress_v8_shared_nominal",
            )
            decision = independent.decide(
                state_history=states,
                applied_action_history=applied,
                nominal_envelope=envelope,
                now=now,
                observation_delay_steps=local_age,
                communication_delay_steps=communication_age,
                provenance_valid=provenance_valid,
                monotonic_time_valid=monotonic_valid,
                observed_common_mode_bias_m=sensor_bias,
            )
            selected = decision.envelope.action
            reason = decision.reason
            intervened = decision.intervened
            fail_closed = decision.fail_closed
            refusal = decision.refusal_code
            certificate_emitted = decision.certificate_emitted
            if refusal == "none":
                if reason == "nominal_with_verified_aligned_set_backup_tube":
                    nominal_feasible, backup_feasible, terminal_reached = 1, 1, 1
                elif reason == "aligned_set_backup_filter_intervention":
                    nominal_feasible, backup_feasible, terminal_reached = 0, 1, 1
                elif reason == "aligned_set_backup_tube_infeasible_fail_closed":
                    nominal_feasible, backup_feasible, terminal_reached = 0, 0, 0
        else:
            raise DiversityRunnerError("unknown method")

        selected = np.clip(np.asarray(selected, dtype=float), -action_limit, action_limit)
        requested = selected
        if lag_queue:
            command = lag_queue.pop(0)
            lag_queue.append(requested.copy())
        else:
            command = requested.copy()
        disturbance = float(patch["lateral_disturbance_mps2"])
        exogenous = np.zeros(5, dtype=float)
        exogenous[1] = 0.5 * disturbance
        exogenous[4] = -0.5 * disturbance
        actual_command = command + exogenous

        arrays["physics_time_s"][row, step] = (step + 1) * 0.1
        arrays["source_timestamps"][row, step] = (
            local_source * 0.1,
            neighbor_source * 0.1,
        )
        arrays["decision_timestamp"][row, step] = step * 0.1
        arrays["local_age_steps"][row, step] = local_age
        arrays["neighbor_age_steps"][row, step] = communication_age
        arrays["source_center_radius_set"][row, step, 0] = source_center
        arrays["source_center_radius_set"][row, step, 1] = source_radius
        arrays["aligned_center_radius_set"][row, step, 0] = aligned_center
        arrays["aligned_center_radius_set"][row, step, 1] = aligned_radius
        _write_hash(
            arrays["applied_action_history_digest"][row, step],
            action_history_digest,
        )
        _write_hash(arrays["provenance_hash"][row, step], provenance)
        _write_hash(
            arrays["uncertainty_registry_hash"][row], uncertainty_hash
        )
        arrays["selected_action"][row, step] = selected
        arrays["nominal_tube_feasible"][row, step] = nominal_feasible
        arrays["backup_tube_feasible"][row, step] = backup_feasible
        arrays["terminal_reachability"][row, step] = terminal_reached
        arrays["decision_reason_code"][row, step] = _reason_index(reason, refusal)
        arrays["refusal_code"][row, step] = REFUSAL_CODEBOOK.index(refusal)
        arrays["fault_code"][row, step] = _fault_index(
            family, list(patch["registered_faults"]), losses[step]
        )
        arrays["fail_closed"][row, step] = fail_closed
        arrays["intervened"][row, step] = intervened
        intervened_any |= bool(intervened)
        typed_refusal_any |= bool(fail_closed or refusal != "none")

        _, _, _, truncated, _info = env.step(actual_command)
        states.append(env.state.copy())
        applied.append(env._applied_action.copy())
        arrays["applied_planar_velocity"][row, step] = np.stack(
            (env.state[3:5], env.state[8:10])
        )
        speed_limit = action_limit + float(
            design["common_contract"]["design_speed_tolerance_mps"]
        )
        arrays["design_speed_envelope_exceeded"][row, step] = bool(
            np.any(
                np.linalg.norm(
                    arrays["applied_planar_velocity"][row, step], axis=1
                )
                > speed_limit
            )
        )
        arrays["completed_step_mask"][row, step] = True
        operational, hard = _operational_margin(
            env.state, family, parameters, operationalization
        )
        arrays["operational_margin_m"][row, step] = operational
        arrays["hard_margin_m"][row, step] = hard
        _write_hash(
            arrays["world_state_hash"][row, step],
            hashlib.sha256(np.asarray(env.state, dtype="<f8").tobytes()).hexdigest(),
        )
        if certificate_emitted and operational < 0.0:
            positive_certificate_before_violation = True
        minimum_operational = min(minimum_operational, operational)
        minimum_hard = min(minimum_hard, hard)
        operational_violation |= operational < 0.0
        hard_violation |= hard < 0.0
        if truncated and step + 1 != HORIZON:
            raise DiversityRunnerError("simulator truncated before frozen horizon")

    if family != "SDF5" and typed_refusal_any:
        operational_violation = True
    arrays["family_index"][row] = spec.family_index
    arrays["candidate_index"][row] = spec.candidate_index
    arrays["split_index"][row] = spec.split_index
    arrays["future_seed_sentinel"][row] = spec.future_seed
    arrays["driver_index"][row] = spec.driver_index
    arrays["method_index"][row] = spec.method_index
    arrays["pair_id"][row] = spec.pair_id
    arrays["completed_steps"][row] = HORIZON
    arrays["operational_first_violation"][row] = operational_violation
    arrays["hard_first_violation"][row] = hard_violation
    arrays["applied_intervention"][row] = intervened_any
    arrays["typed_refusal"][row] = typed_refusal_any
    arrays["minimum_operational_margin_m"][row] = minimum_operational
    arrays["minimum_hard_margin_m"][row] = minimum_hard
    arrays["design_speed_limit_mps"][row] = action_limit + float(
        design["common_contract"]["design_speed_tolerance_mps"]
    )
    _write_hash(arrays["scenario_registry_hash"][row], scenario_digest)
    if positive_certificate_before_violation:
        arrays["decision_reason_code"][row, -1] = arrays[
            "decision_reason_code"
        ][row, -1]
    del env
    gc.collect()


def _summary(
    arrays: Mapping[str, np.ndarray],
    manifest_sha256: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    rows = []
    for split_index, split in enumerate(SPLIT_CODEBOOK):
        for family_index, family in enumerate(FAMILY_CODEBOOK):
            for driver_index, driver in enumerate(DRIVER_CODEBOOK):
                for method_index, method in enumerate(METHOD_CODEBOOK):
                    mask = (
                        (arrays["split_index"] == split_index)
                        & (arrays["family_index"] == family_index)
                        & (arrays["driver_index"] == driver_index)
                        & (arrays["method_index"] == method_index)
                    )
                    if not np.any(mask):
                        continue
                    rows.append(
                        {
                            "split": split,
                            "family": family,
                            "driver": driver,
                            "method": method,
                            "rollouts": int(np.sum(mask)),
                            "operational_first_violation_rate": float(
                                np.mean(arrays["operational_first_violation"][mask])
                            ),
                            "hard_first_violation_rate": float(
                                np.mean(arrays["hard_first_violation"][mask])
                            ),
                            "typed_refusal_rate": float(
                                np.mean(arrays["typed_refusal"][mask])
                            ),
                            "speed_exceedance_rate": float(
                                np.mean(
                                    np.any(
                                        arrays["design_speed_envelope_exceeded"][mask],
                                        axis=1,
                                    )
                                )
                            ),
                        }
                    )
    return {
        "schema_version": "1.0",
        "result_id": "circa-ress-v8-sim-diversity-r1-scientific-result",
        "route_id": ROUTE_ID,
        "status": "SCIENTIFIC_RUN_COMPLETE_PENDING_FROZEN_ANALYSIS",
        "manifest_sha256": manifest_sha256,
        "rollouts": ROLLOUTS,
        "horizon_steps": HORIZON,
        "independent_units": 960,
        "scientific_attempts_consumed": 1,
        "retry_allowed": False,
        "runtime_seconds": runtime_seconds,
        "descriptive_cells": rows,
        "claim_eligible": False,
        "analysis_pending": True,
    }


def run(manifest_path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest_sha = sha256(manifest_file)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    design, operationalization, registry, output, state_path = (
        validate_runnable_manifest(manifest, repo)
    )
    schedule = compile_schedule(manifest)
    output.mkdir(parents=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    started_at_ns = time.time_ns()
    attempt = {
        "schema_version": "1.0",
        "route_id": ROUTE_ID,
        "status": "SCIENTIFIC_ATTEMPT_STARTED",
        "manifest_sha256": manifest_sha,
        "started_at_ns": started_at_ns,
        "scientific_attempts_consumed": 1,
        "retry_allowed": False,
    }
    _atomic_json(output / "ATTEMPT_STARTED.json", attempt)
    _atomic_json(
        state_path,
        {
            **attempt,
            "status": "RUNNING_SCIENTIFIC_EXACTLY_ONCE",
            "completed_rollouts": 0,
            "total_rollouts": ROLLOUTS,
        },
    )
    arrays = allocate_arrays()
    _write_hash(arrays["world_sha256"], WORLD_REGISTRY_SHA256)
    _write_hash(
        arrays["adapter_source_sha256"],
        manifest["source_hashes"][
            "src/agc_runtime_assurance/circa_ress_v8_sim_diversity_adapter.py"
        ],
    )
    _write_hash(
        arrays["schema_source_sha256"],
        manifest["source_hashes"][
            "src/agc_runtime_assurance/circa_ress_v8_sim_diversity_schema.py"
        ],
    )
    _write_hash(arrays["codebook_hash"], codebook_hash())
    started = time.perf_counter()
    try:
        for row, spec in enumerate(schedule):
            _run_one(
                spec,
                row,
                arrays,
                repo,
                design,
                operationalization,
                registry,
            )
            if (row + 1) % 25 == 0 or row + 1 == ROLLOUTS:
                _atomic_json(
                    state_path,
                    {
                        **attempt,
                        "status": "RUNNING_SCIENTIFIC_EXACTLY_ONCE",
                        "completed_rollouts": row + 1,
                        "total_rollouts": ROLLOUTS,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )
        _validate_scientific_arrays(arrays)
        runtime_seconds = time.perf_counter() - started
        summary = _summary(arrays, manifest_sha, runtime_seconds)
        archive_stream = BytesIO()
        np.savez_compressed(
            archive_stream,
            **{name: arrays[name] for name in sorted(arrays)},
        )
        archive = archive_stream.getvalue()
        summary_bytes = (
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        attempt_bytes = (output / "ATTEMPT_STARTED.json").stat().st_size
        total_bytes = len(archive) + len(summary_bytes) + attempt_bytes
        if total_bytes > OUTPUT_CAPACITY_BYTES:
            raise DiversityRunnerError(
                f"lossless output capacity exceeded: {total_bytes}"
            )
        trace_tmp = output / "trace_arrays.npz.tmp"
        trace_tmp.write_bytes(archive)
        trace_tmp.replace(output / "trace_arrays.npz")
        result_tmp = output / "result.json.tmp"
        result_tmp.write_bytes(summary_bytes)
        result_tmp.replace(output / "result.json")
        terminal = {
            **attempt,
            "status": "PASS_SCIENTIFIC_RUN_COMPLETE_PENDING_FROZEN_ANALYSIS",
            "completed_rollouts": ROLLOUTS,
            "total_rollouts": ROLLOUTS,
            "runtime_seconds": runtime_seconds,
            "output_bytes": total_bytes,
            "output_capacity_bytes": OUTPUT_CAPACITY_BYTES,
            "trace_arrays_sha256": sha256(output / "trace_arrays.npz"),
            "result_sha256": sha256(output / "result.json"),
        }
        _atomic_json(state_path, terminal)
        return terminal
    except BaseException as error:
        _atomic_json(
            state_path,
            {
                **attempt,
                "status": "TERMINAL_FAIL_SCIENTIFIC_NO_RETRY",
                "completed_rollouts": int(
                    np.count_nonzero(arrays["completed_steps"] == HORIZON)
                ),
                "total_rollouts": ROLLOUTS,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


__all__ = [
    "DiversityRunnerError",
    "RunSpec",
    "compile_schedule",
    "derive_future_seed",
    "run",
    "validate_runnable_manifest",
]
