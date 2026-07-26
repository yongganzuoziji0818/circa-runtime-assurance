"""Outcome-blind feasible initial-state domain for CIRCA-RESS-V9.

This module is deterministic and simulator-free.  It defines support bounds,
maps a future seed to one state, and proves the complete support satisfies the
frozen 0.30 m initial operational-margin floor for every positive candidate.
It contains no seed material and has no scientific entry point.
"""

from __future__ import annotations

import hashlib
import math
from typing import Mapping

import numpy as np


ROUTE_ID = "CIRCA-RESS-V9-FEASIBLE-INITIAL-DOMAIN-R1"
SEED_NAMESPACE = "circa-ress-v9-feasible-initial-domain-r1"
INITIAL_MARGIN_FLOOR_M = 0.30
NOMINAL_STATE = np.asarray(
    [-2.5, 0.0, 2.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0],
    dtype=float,
)
DELTA_LOW = np.asarray([-0.03, -0.03, -0.03, 0.0], dtype=float)
DELTA_HIGH = np.asarray([0.03, 0.03, 0.03, 0.0], dtype=float)
STATE_INDICES = np.asarray([0, 1, 6, 7], dtype=int)

AISLE_WIDTHS_M: Mapping[str, float] = {
    "SDF1_A": 1.60,
    "SDF1_B": 1.30,
    "SDF4_A": 1.50,
    "SDF4_B": 1.30,
}
UAV_RADIUS_M = 0.18
UGV_HALF_WIDTH_M = 0.25
WALL_HALF_THICKNESS_M = 0.10
HARD_SEPARATION_M = 1.0


class V9InitialDomainError(ValueError):
    """Raised when the frozen V9 initial-state domain drifts."""


def _counter_uniform(future_seed: int, coordinate: str) -> float:
    if not isinstance(future_seed, int) or future_seed < 0:
        raise V9InitialDomainError("future seed must be a non-negative integer")
    body = f"{SEED_NAMESPACE}|{future_seed}|initial-domain|{coordinate}".encode()
    integer = int.from_bytes(hashlib.sha256(body).digest()[:8], "big")
    return integer / float(2**64)


def initial_state_from_future_seed(future_seed: int) -> np.ndarray:
    """Map a future seed once into the frozen bounded support."""

    state = NOMINAL_STATE.copy()
    coordinates = ("uav_x", "uav_y", "ugv_x", "ugv_y")
    unit = np.asarray(
        [_counter_uniform(future_seed, name) for name in coordinates],
        dtype=float,
    )
    delta = DELTA_LOW + unit * (DELTA_HIGH - DELTA_LOW)
    state[STATE_INDICES] += delta
    validate_realized_state(state)
    return state


def validate_realized_state(state: np.ndarray) -> None:
    value = np.asarray(state, dtype=float)
    if value.shape != (10,) or not np.all(np.isfinite(value)):
        raise V9InitialDomainError("initial state must be finite with shape (10,)")
    delta = value[STATE_INDICES] - NOMINAL_STATE[STATE_INDICES]
    if np.any(delta < DELTA_LOW) or np.any(delta > DELTA_HIGH):
        raise V9InitialDomainError("initial state lies outside frozen support")
    if value[7] != 0.0:
        raise V9InitialDomainError("UGV lateral position must remain exactly zero")
    separation = float(np.linalg.norm(value[:2] - value[6:8]))
    if separation - HARD_SEPARATION_M < INITIAL_MARGIN_FLOOR_M:
        raise V9InitialDomainError("initial hard-separation margin below floor")


def analytic_support_margins() -> dict[str, float]:
    """Return conservative lower margins over the complete coordinate support."""

    worst_dx = (
        NOMINAL_STATE[6]
        + DELTA_LOW[2]
        - NOMINAL_STATE[0]
        - DELTA_HIGH[0]
    )
    worst_dy = max(
        0.0,
        abs(NOMINAL_STATE[1]) - abs(DELTA_HIGH[1]),
    )
    hard = math.hypot(worst_dx, worst_dy) - HARD_SEPARATION_M
    margins = {"inter_agent_hard": hard}
    for candidate, aisle in AISLE_WIDTHS_M.items():
        free_half = aisle / 2.0 - WALL_HALF_THICKNESS_M
        margins[f"{candidate}_uav_corridor"] = (
            free_half - UAV_RADIUS_M - DELTA_HIGH[1]
        )
        margins[f"{candidate}_ugv_corridor"] = free_half - UGV_HALF_WIDTH_M
    margins["SDF2_uav_ramp"] = 1.0 - UAV_RADIUS_M - DELTA_HIGH[1]
    margins["SDF2_ugv_ramp"] = 1.0 - UGV_HALF_WIDTH_M
    return margins


def validate_complete_support() -> dict[str, float]:
    margins = analytic_support_margins()
    failed = {
        name: value
        for name, value in margins.items()
        if value < INITIAL_MARGIN_FLOOR_M - 1e-12
    }
    if failed:
        raise V9InitialDomainError(f"frozen support is infeasible: {failed}")
    return margins


__all__ = [
    "DELTA_HIGH",
    "DELTA_LOW",
    "INITIAL_MARGIN_FLOOR_M",
    "NOMINAL_STATE",
    "ROUTE_ID",
    "SEED_NAMESPACE",
    "V9InitialDomainError",
    "analytic_support_margins",
    "initial_state_from_future_seed",
    "validate_complete_support",
    "validate_realized_state",
]
