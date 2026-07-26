"""Non-running lossless schema for CIRCA-RESS-V9.

The member layout intentionally reuses the audited V8 primitive layout while
freezing the V9 dimensions and schema identity.  This module does not derive
seeds, write archives, invoke Gazebo, or expose a scientific entry point.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .circa_ress_v8_sim_diversity_schema import (
    CANDIDATE_CODEBOOK,
    DRIVER_CODEBOOK,
    DiversitySchemaError,
    FAMILY_CODEBOOK,
    FAULT_CODEBOOK,
    METHOD_CODEBOOK,
    REFUSAL_CODEBOOK,
    SPLIT_CODEBOOK,
    allocate_schema_sentinel as _allocate,
    array_schema as _array_schema,
    codebook_hash,
    validate_schema_arrays as _validate,
)


SCHEMA_VERSION = "circa-ress-v9-feasible-initial-domain-r1-lossless-arrays-v1"
DEFAULT_ROLLOUTS = 7424
DEFAULT_HORIZON = 80


def array_schema(
    rollouts: int = DEFAULT_ROLLOUTS,
    horizon: int = DEFAULT_HORIZON,
) -> dict[str, tuple[np.dtype, tuple[int, ...]]]:
    return _array_schema(rollouts, horizon)


def allocate_schema_sentinel(
    rollouts: int = DEFAULT_ROLLOUTS,
    horizon: int = DEFAULT_HORIZON,
) -> dict[str, np.ndarray]:
    return _allocate(rollouts, horizon)


def validate_schema_arrays(
    arrays: Mapping[str, np.ndarray],
    rollouts: int = DEFAULT_ROLLOUTS,
    horizon: int = DEFAULT_HORIZON,
) -> None:
    _validate(arrays, rollouts, horizon)


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
