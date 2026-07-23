"""Role-valid 1U1G task construction for the CIRCA-GZ0-v7 instrument.

The v1--v6 negative-control task still exchanged the agents' longitudinal
positions.  This module makes the role difference explicit: hazard cases keep
the crossing task and closing initial velocities; negative controls retain the
initial ordering and start with separating velocities.  It contains no runner
and creates no experimental output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circa_gazebo_gz0 import SceneCandidate
from .sandbox_task import SandboxComparisonTask


@dataclass(frozen=True)
class RoleValidScenario:
    role_id: str
    hazard_active: bool
    initial_state: np.ndarray
    task: SandboxComparisonTask
    initial_radial_separation_rate_mps: float


def _radial_separation_rate(state: np.ndarray) -> float:
    vector = np.asarray(state, dtype=float).reshape(-1)
    if vector.shape != (10,) or not np.all(np.isfinite(vector)):
        raise ValueError("state must be finite with shape (10,)")
    position = vector[:2] - vector[6:8]
    distance = float(np.linalg.norm(position))
    if distance <= 1e-12:
        raise ValueError("relative direction is undefined at zero separation")
    velocity = vector[3:5] - vector[8:10]
    return float(position @ velocity / distance)


def role_conditioned_initial_state(
    candidate: SceneCandidate, *, hazard_active: bool
) -> np.ndarray:
    """Return a role-conditioned state without changing speed magnitudes.

    Hazard cases retain the registered closing state.  Negative controls reverse
    only the longitudinal velocity directions, so the initial ordering is
    preserved and separation begins nondecreasing.
    """

    state = candidate.initial_state().copy()
    if not hazard_active:
        state[3] = -abs(float(state[3]))
        state[8] = abs(float(state[8]))
    return state


def build_role_valid_scenario(
    candidate: SceneCandidate,
    *,
    hazard_active: bool,
    operational_separation_m: float,
    action_limit: float,
    corridor_lateral_goal: float,
    control_uav_lateral_goal: float,
    control_ugv_lateral_goal: float,
    longitudinal_goal_magnitude: float = 4.0,
) -> RoleValidScenario:
    values = np.asarray(
        [
            operational_separation_m,
            action_limit,
            corridor_lateral_goal,
            control_uav_lateral_goal,
            control_ugv_lateral_goal,
            longitudinal_goal_magnitude,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("role-valid task parameters must be finite")
    if operational_separation_m <= 0.0 or action_limit <= 0.0:
        raise ValueError("separation and action limit must be positive")
    if longitudinal_goal_magnitude <= candidate.half_gap:
        raise ValueError("longitudinal goal must lie beyond both initial positions")

    state = role_conditioned_initial_state(candidate, hazard_active=hazard_active)
    magnitude = float(longitudinal_goal_magnitude)
    if hazard_active:
        task = SandboxComparisonTask(
            uav_goal=(magnitude, float(corridor_lateral_goal), 2.0),
            ugv_goal=(-magnitude, float(corridor_lateral_goal)),
            minimum_separation=float(operational_separation_m),
            action_limit=float(action_limit),
        )
        role_id = "crossing_hazard"
    else:
        task = SandboxComparisonTask(
            uav_goal=(-magnitude, float(control_uav_lateral_goal), 2.0),
            ugv_goal=(magnitude, float(control_ugv_lateral_goal)),
            minimum_separation=float(operational_separation_m),
            action_limit=float(action_limit),
        )
        role_id = "ordering_preserving_negative_control"

    radial_rate = _radial_separation_rate(state)
    if hazard_active and radial_rate >= 0.0:
        raise ValueError("hazard role must begin closing")
    if not hazard_active and radial_rate < 0.0:
        raise ValueError("negative-control role must begin nonclosing")
    return RoleValidScenario(role_id, hazard_active, state, task, radial_rate)
