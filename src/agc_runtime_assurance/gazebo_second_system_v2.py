"""Versioned Gazebo driver repair for reset-velocity command lifetime."""

from __future__ import annotations

from typing import Any

import numpy as np

from .gazebo_second_system import GazeboAirGroundEnv, GazeboSecondSystemError, constraint_margins


class GazeboAirGroundEnvV2(GazeboAirGroundEnv):
    """Hold reset velocity across two pre-update callbacks.

    The v1 driver issued pose and velocity commands in the same first callback,
    after which the measured velocity collapsed.  This version keeps the pose
    command one-shot but refreshes the requested linear velocity in both reset
    iterations before normal force actuation begins.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        self._reset_pose_pending = False
        self._reset_velocity_hold = 0
        super().__init__(*args, **kwargs)

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
                self._uav_link.set_linear_velocity(
                    ecm, self._math.Vector3d(float(u[3]), float(u[4]), float(u[5]))
                )
                self._ugv_link.set_linear_velocity(
                    ecm, self._math.Vector3d(float(u[8]), float(u[9]), 0.0)
                )
                self._uav_link.set_angular_velocity(ecm, self._math.Vector3d(0.0, 0.0, 0.0))
                self._ugv_link.set_angular_velocity(ecm, self._math.Vector3d(0.0, 0.0, 0.0))
                self._reset_velocity_hold -= 1
            else:
                self._uav_link.add_world_force(ecm, self._math.Vector3d(*map(float, self._uav_force)))
                self._ugv_link.add_world_force(ecm, self._math.Vector3d(*map(float, self._ugv_force)))
        except Exception as exc:  # pragma: no cover - remote integration path
            self._callback_error = exc

    def reset(
        self,
        *,
        seed: int,
        interaction_close: bool = False,
        initial_state: np.ndarray | None = None,
        position_jitter_scale: float = 0.03,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not np.isfinite(position_jitter_scale) or position_jitter_scale < 0.0:
            raise GazeboSecondSystemError("position_jitter_scale must be finite and non-negative")
        rng = np.random.default_rng(int(seed))
        jitter = rng.normal(0.0, position_jitter_scale, 4)
        state = (
            np.array([-4.0, -2.0, 2.0, 0.0, 0.0, 0.0, 4.0, 2.0, 0.0, 0.0])
            if initial_state is None
            else np.asarray(initial_state, dtype=float).reshape(-1).copy()
        )
        if state.shape != (10,) or not np.all(np.isfinite(state)):
            raise GazeboSecondSystemError("initial_state must be finite with shape (10,)")
        state[[0, 1, 6, 7]] += jitter
        if interaction_close and initial_state is None:
            close_jitter = np.random.default_rng(int(seed)).normal(0.0, 0.08, 2)
            state[:2] = np.array([-1.15, 0.0]) + close_jitter
            state[6:8] = np.array([1.15, 0.0]) - close_jitter
        self._requested_state = state
        self._applied_action.fill(0.0)
        self._uav_force.fill(0.0)
        self._ugv_force.fill(0.0)
        self._reset_pending = False
        self._reset_pose_pending = True
        self._reset_velocity_hold = 2
        self.step_index = 0
        self._run_iterations(2)
        margins = constraint_margins(self.state, self.step_index)
        return self._observation(), {"development_only": True, "margins": margins}

