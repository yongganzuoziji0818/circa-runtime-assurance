"""Non-running lossless schema for CIRCA-RESS-V8-SIM-DIVERSITY-R1.

This module defines array shapes and in-memory sentinels only.  It has no seed
derivation, simulator bridge, archive writer, capacity audit, or scientific
entry point.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "circa-ress-v8-sim-diversity-r1-lossless-arrays-v1"
DEFAULT_ROLLOUTS = 6400
DEFAULT_HORIZON = 80
FAMILY_CODEBOOK = ("SDF1", "SDF2", "SDF3", "SDF4", "SDF5")
CANDIDATE_CODEBOOK = ("A", "B")
SPLIT_CODEBOOK = ("validation", "evaluation")
DRIVER_CODEBOOK = (
    "command_persistent_unbounded_v3",
    "planar_speed_projected_v4",
)
METHOD_CODEBOOK = (
    "shadow_no_override",
    "registered_one_step_cbf",
    "robust_backup_filter_v7_stale_point",
    "timestamp_aligned_set_backup_v8",
)
REFUSAL_CODEBOOK = (
    "none",
    "invalid_alignment_evidence",
    "invalid_provenance",
    "invalid_monotonic_time",
    "expired_action",
    "invalid_action_contract",
    "incomplete_horizon",
    "simulator_or_transport_failure",
)
FAULT_CODEBOOK = (
    "none",
    "transport_delay",
    "packet_omission",
    "out_of_order_or_stale_state",
    "evidence_provenance_hash_mismatch",
    "monotonic_time_reversal",
    "action_expired",
    "compound_ood",
)


class DiversitySchemaError(ValueError):
    """Raised when a non-running schema object violates the frozen contract."""


def _positive_dimensions(rollouts: int, horizon: int) -> None:
    if not isinstance(rollouts, int) or not isinstance(horizon, int):
        raise DiversitySchemaError("schema dimensions must be integers")
    if rollouts <= 0 or horizon <= 0:
        raise DiversitySchemaError("schema dimensions must be positive")


def array_schema(
    rollouts: int = DEFAULT_ROLLOUTS,
    horizon: int = DEFAULT_HORIZON,
) -> dict[str, tuple[np.dtype, tuple[int, ...]]]:
    _positive_dimensions(rollouts, horizon)
    rh = (rollouts, horizon)
    return {
        "physics_time_s": (np.dtype("<f8"), rh),
        "source_timestamps": (np.dtype("<f8"), rh + (2,)),
        "decision_timestamp": (np.dtype("<f8"), rh),
        "local_age_steps": (np.dtype("u1"), rh),
        "neighbor_age_steps": (np.dtype("u1"), rh),
        "source_center_radius_set": (np.dtype("<f8"), rh + (2, 10)),
        "aligned_center_radius_set": (np.dtype("<f8"), rh + (2, 10)),
        "applied_action_history_digest": (np.dtype("u1"), rh + (32,)),
        "uncertainty_registry_hash": (np.dtype("u1"), (rollouts, 32)),
        "provenance_hash": (np.dtype("u1"), rh + (32,)),
        "world_state_hash": (np.dtype("u1"), rh + (32,)),
        "selected_action": (np.dtype("<f8"), rh + (5,)),
        "applied_planar_velocity": (np.dtype("<f8"), rh + (2, 2)),
        "operational_margin_m": (np.dtype("<f8"), rh),
        "hard_margin_m": (np.dtype("<f8"), rh),
        "nominal_tube_feasible": (np.dtype("i1"), rh),
        "backup_tube_feasible": (np.dtype("i1"), rh),
        "terminal_reachability": (np.dtype("i1"), rh),
        "decision_reason_code": (np.dtype("u1"), rh),
        "refusal_code": (np.dtype("u1"), rh),
        "fault_code": (np.dtype("u1"), rh),
        "fail_closed": (np.dtype("?"), rh),
        "intervened": (np.dtype("?"), rh),
        "completed_step_mask": (np.dtype("?"), rh),
        "design_speed_envelope_exceeded": (np.dtype("?"), rh),
        "family_index": (np.dtype("u1"), (rollouts,)),
        "candidate_index": (np.dtype("u1"), (rollouts,)),
        "split_index": (np.dtype("u1"), (rollouts,)),
        "future_seed_sentinel": (np.dtype("<i8"), (rollouts,)),
        "driver_index": (np.dtype("u1"), (rollouts,)),
        "method_index": (np.dtype("u1"), (rollouts,)),
        "pair_id": (np.dtype("<i4"), (rollouts,)),
        "completed_steps": (np.dtype("u1"), (rollouts,)),
        "operational_first_violation": (np.dtype("?"), (rollouts,)),
        "hard_first_violation": (np.dtype("?"), (rollouts,)),
        "applied_intervention": (np.dtype("?"), (rollouts,)),
        "typed_refusal": (np.dtype("?"), (rollouts,)),
        "minimum_operational_margin_m": (np.dtype("<f8"), (rollouts,)),
        "minimum_hard_margin_m": (np.dtype("<f8"), (rollouts,)),
        "design_speed_limit_mps": (np.dtype("<f8"), (rollouts,)),
        "scenario_registry_hash": (np.dtype("u1"), (rollouts, 32)),
        "world_sha256": (np.dtype("u1"), (32,)),
        "adapter_source_sha256": (np.dtype("u1"), (32,)),
        "schema_source_sha256": (np.dtype("u1"), (32,)),
        "codebook_hash": (np.dtype("u1"), (32,)),
    }


def allocate_schema_sentinel(
    rollouts: int,
    horizon: int,
) -> dict[str, np.ndarray]:
    arrays = {
        name: np.zeros(shape, dtype=dtype)
        for name, (dtype, shape) in array_schema(rollouts, horizon).items()
    }
    arrays["future_seed_sentinel"].fill(-1)
    arrays["nominal_tube_feasible"].fill(-1)
    arrays["backup_tube_feasible"].fill(-1)
    arrays["terminal_reachability"].fill(-1)
    arrays["minimum_operational_margin_m"].fill(np.nan)
    arrays["minimum_hard_margin_m"].fill(np.nan)
    arrays["operational_margin_m"].fill(np.nan)
    arrays["hard_margin_m"].fill(np.nan)
    return arrays


def validate_schema_arrays(
    arrays: Mapping[str, np.ndarray],
    rollouts: int,
    horizon: int,
) -> None:
    expected = array_schema(rollouts, horizon)
    if set(arrays) != set(expected):
        raise DiversitySchemaError("schema member set drifted")
    for name, (dtype, shape) in expected.items():
        value = np.asarray(arrays[name])
        if value.dtype != dtype or value.shape != shape or value.dtype.hasobject:
            raise DiversitySchemaError(f"schema mismatch: {name}")
    if np.any(np.asarray(arrays["future_seed_sentinel"]) != -1):
        raise DiversitySchemaError("non-running schema must not contain seed material")
    completed = np.asarray(arrays["completed_step_mask"], dtype=bool)
    typed_refusal = np.broadcast_to(
        np.asarray(arrays["typed_refusal"], dtype=bool)[:, None],
        completed.shape,
    )
    if np.any(
        np.asarray(arrays["fail_closed"])[completed] & ~typed_refusal[completed]
    ):
        raise DiversitySchemaError("completed fail-closed steps require typed refusal")
    if np.any(np.asarray(arrays["completed_steps"]) > horizon):
        raise DiversitySchemaError("completed step count exceeds horizon")
    for name, size in (
        ("family_index", len(FAMILY_CODEBOOK)),
        ("candidate_index", len(CANDIDATE_CODEBOOK)),
        ("split_index", len(SPLIT_CODEBOOK)),
        ("driver_index", len(DRIVER_CODEBOOK)),
        ("method_index", len(METHOD_CODEBOOK)),
    ):
        if np.any(np.asarray(arrays[name]) >= size):
            raise DiversitySchemaError(f"{name} exceeds its codebook")


def codebook_hash() -> str:
    payload = {
        "families": FAMILY_CODEBOOK,
        "candidates": CANDIDATE_CODEBOOK,
        "splits": SPLIT_CODEBOOK,
        "drivers": DRIVER_CODEBOOK,
        "methods": METHOD_CODEBOOK,
        "refusals": REFUSAL_CODEBOOK,
        "faults": FAULT_CODEBOOK,
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def schema_metadata(
    rollouts: int = DEFAULT_ROLLOUTS,
    horizon: int = DEFAULT_HORIZON,
) -> dict[str, Any]:
    schema = array_schema(rollouts, horizon)
    return {
        "schema_version": SCHEMA_VERSION,
        "schema_only": True,
        "runnable": False,
        "scientific_seed_material_generated": False,
        "scientific_output_generated": False,
        "capacity_audit_executed": False,
        "independent_unit": "family_x_candidate_x_future_seed",
        "rollouts": rollouts,
        "horizon": horizon,
        "members": {
            name: {"dtype": dtype.str, "shape": list(shape)}
            for name, (dtype, shape) in schema.items()
        },
        "codebook_hash": codebook_hash(),
    }


__all__ = [
    "CANDIDATE_CODEBOOK",
    "DEFAULT_HORIZON",
    "DEFAULT_ROLLOUTS",
    "DRIVER_CODEBOOK",
    "DiversitySchemaError",
    "FAMILY_CODEBOOK",
    "FAULT_CODEBOOK",
    "METHOD_CODEBOOK",
    "REFUSAL_CODEBOOK",
    "SCHEMA_VERSION",
    "SPLIT_CODEBOOK",
    "allocate_schema_sentinel",
    "array_schema",
    "codebook_hash",
    "schema_metadata",
    "validate_schema_arrays",
]
