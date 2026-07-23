"""Command-persistent deterministic Gazebo actuation adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from .gazebo_second_system import CONTROLLER_DT_SECONDS, PHYSICS_ITERATIONS_PER_STEP
from .gazebo_second_system_v2 import GazeboAirGroundEnvV2


class GazeboAirGroundEnvV3(GazeboAirGroundEnvV2):
    """Integrate registered accelerations and refresh velocity every physics tick.

    Gazebo's Python velocity command is transient in this TestFixture path.  The
    adapter therefore treats ``action_to_world_forces`` outputs as deterministic
    accelerations, integrates them at the frozen physics step, and refreshes the
    resulting world velocity command on every pre-update callback.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        self._commanded_uav_velocity = np.zeros(3, dtype=float)
        self._commanded_ugv_velocity = np.zeros(3, dtype=float)
        super().__init__(*args, **kwargs)

    def _set_commanded_velocity(self, ecm: Any) -> None:
        self._uav_link.set_linear_velocity(
            ecm, self._math.Vector3d(*map(float, self._commanded_uav_velocity))
        )
        self._ugv_link.set_linear_velocity(
            ecm, self._math.Vector3d(*map(float, self._commanded_ugv_velocity))
        )
        self._uav_link.set_angular_velocity(ecm, self._math.Vector3d(0.0, 0.0, 0.0))
        self._ugv_link.set_angular_velocity(ecm, self._math.Vector3d(0.0, 0.0, 0.0))

    def _on_pre_update(self, _info: Any, ecm: Any) -> None:
        try:
            self._ensure_entities(ecm)
            import time

            now = time.perf_counter_ns()
            if self._step_wall_start_ns and not self._first_pre_update_ns:
                self._first_pre_update_ns = now
            if self._reset_pose_pending:
                u = self._requested_state
                self._uav_model.set_world_pose_cmd(
                    ecm, self._math.Pose3d(float(u[0]), float(u[1]), float(u[2]), 0.0, 0.0, 0.0)
                )
                self._ugv_model.set_world_pose_cmd(
                    ecm, self._math.Pose3d(float(u[6]), float(u[7]), 0.2, 0.0, 0.0, 0.0)
                )
                self._reset_pose_pending = False
            if self._reset_velocity_hold > 0:
                u = self._requested_state
                self._commanded_uav_velocity = np.asarray(u[3:6], dtype=float).copy()
                self._commanded_ugv_velocity = np.array([u[8], u[9], 0.0], dtype=float)
                self._reset_velocity_hold -= 1
            else:
                physics_dt = CONTROLLER_DT_SECONDS / PHYSICS_ITERATIONS_PER_STEP
                self._commanded_uav_velocity += self._uav_force * physics_dt
                self._commanded_ugv_velocity += self._ugv_force * physics_dt
            self._set_commanded_velocity(ecm)
        except Exception as exc:  # pragma: no cover - remote integration path
            self._callback_error = exc
