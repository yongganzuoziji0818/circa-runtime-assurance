"""Non-runnable lossless evidence schema for the proposed CIRCA-GZ1-v10 route.

This module intentionally has no simulator import, scientific schedule builder,
output writer, or experiment entry point.  It can only describe and validate the
future evidence arrays and compute a conservative storage-capacity proposal.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np


SCHEMA_VERSION = "circa-gz1-v10-isaac-confirmatory-lossless-arrays-design-v1"
DEFAULT_ROLLOUTS = 15_360
DEFAULT_HORIZON = 80
DEFAULT_CAP_BYTES = 1_536 * 2**20
SCIENTIFIC_SEED_MATERIAL_GENERATED = False
SCIENTIFIC_RUN_AUTHORIZED = False
SCIENTIFIC_OUTPUT_AUTHORIZED = False


class CircaIsaacGZ1V10SchemaError(ValueError):
    """Raised when the non-running schema contract is internally inconsistent."""


def array_schema(rollouts: int, horizon: int) -> dict[str, tuple[np.dtype, tuple[int, ...]]]:
    """Return the frozen logical array schema without allocating full-size arrays."""

    if rollouts <= 0 or horizon <= 0:
        raise CircaIsaacGZ1V10SchemaError("rollouts and horizon must be positive")
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
        "scenario_index": (np.dtype("<u2"), (rollouts,)),
        "scenario_seed": (np.dtype("<i8"), (rollouts,)),
        "driver_index": (np.dtype("u1"), (rollouts,)),
        "method_index": (np.dtype("u1"), (rollouts,)),
        "hazard_active": (np.dtype("?"), (rollouts,)),
        "operational_first_violation": (np.dtype("?"), (rollouts,)),
        "hard_first_violation": (np.dtype("?"), (rollouts,)),
        "applied_intervention": (np.dtype("?"), (rollouts,)),
        "completed_steps": (np.dtype("u1"), (rollouts,)),
        "minimum_operational_margin_m": (np.dtype("<f8"), (rollouts,)),
        "minimum_hard_margin_m": (np.dtype("<f8"), (rollouts,)),
        # Isaac/PhysX cross-simulator evidence fields.
        "physics_time_s": (np.dtype("<f8"), rh),
        "isaac_body_state": (np.dtype("<f8"), rh + (2, 13)),
        "isaac_contact_impulse": (np.dtype("<f8"), rh + (2, 3)),
        "solver_substeps": (np.dtype("u1"), rh),
        "physics_backend_hash": (np.dtype("u1"), (32,)),
        "stage_hash": (np.dtype("u1"), (32,)),
        # Prespecified design/blocking fields.  Values remain absent until a
        # separately authorized scientific schedule is materialized.
        "stress_block_index": (np.dtype("u1"), (rollouts,)),
        "sensor_bias_xy_m": (np.dtype("<f8"), (rollouts, 2, 2)),
        "observation_delay_steps_by_agent": (np.dtype("u1"), (rollouts, 2)),
        "communication_delay_steps_by_direction": (np.dtype("u1"), (rollouts, 2)),
        "communication_dropout_rate": (np.dtype("<f8"), (rollouts,)),
        "actuation_lag_steps_by_agent": (np.dtype("u1"), (rollouts, 2)),
        "mass_scale_by_agent": (np.dtype("<f8"), (rollouts, 2)),
        "friction_scale_by_agent": (np.dtype("<f8"), (rollouts, 2)),
        "drag_scale_by_agent": (np.dtype("<f8"), (rollouts, 2)),
        "wind_xy_mps": (np.dtype("<f8"), (rollouts, 2)),
        "witness_radius_scale": (np.dtype("<f8"), (rollouts,)),
        "randomization_block_index": (np.dtype("<u2"), (rollouts,)),
        "pair_id": (np.dtype("<u4"), (rollouts,)),
    }


def allocate_tiny_fixture(rollouts: int = 2, horizon: int = 3) -> dict[str, np.ndarray]:
    """Allocate a tiny schema fixture containing no scientific seed material."""

    arrays = {
        name: np.zeros(shape, dtype=dtype)
        for name, (dtype, shape) in array_schema(rollouts, horizon).items()
    }
    arrays["scenario_seed"].fill(-1)
    for name in ("nominal_tube_feasible", "backup_tube_feasible", "terminal_reachability"):
        arrays[name].fill(-1)
    for name in ("worst_set_margin", "minimum_operational_margin_m", "minimum_hard_margin_m"):
        arrays[name].fill(np.nan)
    arrays["certificate_validity_interval"].fill(-1)
    return arrays


def validate_tiny_fixture(arrays: dict[str, np.ndarray], rollouts: int = 2, horizon: int = 3) -> None:
    expected = array_schema(rollouts, horizon)
    if set(arrays) != set(expected):
        raise CircaIsaacGZ1V10SchemaError("fixture field set does not match schema")
    for name, (dtype, shape) in expected.items():
        value = arrays[name]
        if value.dtype != dtype or value.shape != shape or value.dtype.hasobject:
            raise CircaIsaacGZ1V10SchemaError(f"schema mismatch for {name}")
    if not np.all(arrays["scenario_seed"] == -1):
        raise CircaIsaacGZ1V10SchemaError("schema fixture contains non-sentinel seed material")


def capacity_projection(
    rollouts: int = DEFAULT_ROLLOUTS,
    horizon: int = DEFAULT_HORIZON,
    cap_bytes: int = DEFAULT_CAP_BYTES,
    *,
    safety_factor: float = 1.10,
    fixed_overhead_bytes: int = 4 * 2**20,
) -> dict[str, Any]:
    """Return a conservative proposal, not a capacity-audit PASS receipt.

    The projection treats every logical array byte as incompressible, adds ten
    percent for archive/codec variability, and adds a fixed 4 MiB allowance for
    JSON summaries and archive headers.  A later high-entropy schema-only audit
    remains mandatory before any scientific authorization.
    """

    if cap_bytes <= 0 or safety_factor < 1.0 or fixed_overhead_bytes < 0:
        raise CircaIsaacGZ1V10SchemaError("invalid capacity projection arguments")
    schema = array_schema(rollouts, horizon)
    raw_array_bytes = sum(
        dtype.itemsize * math.prod(shape) for dtype, shape in schema.values()
    )
    conservative_bytes = math.ceil(raw_array_bytes * safety_factor + fixed_overhead_bytes)
    largest = sorted(
        (
            {
                "field": name,
                "bytes": dtype.itemsize * math.prod(shape),
                "dtype": dtype.str,
                "shape": list(shape),
            }
            for name, (dtype, shape) in schema.items()
        ),
        key=lambda item: item["bytes"],
        reverse=True,
    )[:10]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "CAPACITY_PROPOSAL_ONLY_NOT_AN_AUDIT_PASS",
        "scientific_seed_material_generated": False,
        "scientific_run_executed": False,
        "scientific_output_generated": False,
        "rollouts": rollouts,
        "horizon_steps": horizon,
        "field_count": len(schema),
        "object_dtype_present": any(dtype.hasobject for dtype, _ in schema.values()),
        "raw_array_bytes": raw_array_bytes,
        "safety_factor": safety_factor,
        "fixed_overhead_bytes": fixed_overhead_bytes,
        "conservative_projected_bytes": conservative_bytes,
        "proposed_cap_bytes": cap_bytes,
        "proposal_headroom_bytes": cap_bytes - conservative_bytes,
        "projection_within_proposed_cap": conservative_bytes <= cap_bytes,
        "largest_fields": largest,
        "next_capacity_gate": "separately_authorized_high_entropy_lossless_schema_only_audit",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=int, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--cap-mib", type=int, default=1_536)
    args = parser.parse_args()
    fixture = allocate_tiny_fixture()
    validate_tiny_fixture(fixture)
    result = capacity_projection(args.rollouts, args.horizon, args.cap_mib * 2**20)
    result["tiny_fixture_validation"] = "PASS_NO_SCIENTIFIC_SEEDS"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
