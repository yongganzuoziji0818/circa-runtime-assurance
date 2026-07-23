"""Gazebo Sim 8 development-only second-system adapter.

The adapter is deliberately optional: importing the rest of the package does
not require Gazebo.  It exposes the same 10-state / 5-action contract as the
frozen sandbox while delegating integration to Gazebo rigid-body physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np

from .environment import CompoundShift
from .risk import ConstraintMargins


CONTROLLER_DT_SECONDS = 0.1
PHYSICS_ITERATIONS_PER_STEP = 10
UAV_QUADRATIC_DRAG = 0.015
UGV_QUADRATIC_DRAG = 0.025


class GazeboSecondSystemError(RuntimeError):
    """Fail-closed error raised by the optional Gazebo bridge."""


@dataclass(frozen=True)
class GazeboStepReceipt:
    wall_start_ns: int
    first_pre_update_ns: int
    post_update_ns: int
    wall_end_ns: int
    physics_iterations: int
    deterministic_compute_bound_available: bool = False

    def validate(self) -> None:
        stamps = (
            self.wall_start_ns,
            self.first_pre_update_ns,
            self.post_update_ns,
            self.wall_end_ns,
        )
        if any(not isinstance(value, int) or value <= 0 for value in stamps):
            raise GazeboSecondSystemError("latency receipt timestamps must be positive integers")
        if tuple(sorted(stamps)) != stamps:
            raise GazeboSecondSystemError("latency receipt timestamps are not monotonic")
        if self.physics_iterations != PHYSICS_ITERATIONS_PER_STEP:
            raise GazeboSecondSystemError("unexpected physics iteration count")
        if self.deterministic_compute_bound_available:
            raise GazeboSecondSystemError("generic Linux/Gazebo cannot assert a deterministic compute bound")

    def as_seconds(self) -> dict[str, float | bool]:
        self.validate()
        return {
            "dispatch_to_first_pre_update": (self.first_pre_update_ns - self.wall_start_ns) * 1e-9,
            "first_pre_to_last_post_update": (self.post_update_ns - self.first_pre_update_ns) * 1e-9,
            "post_update_to_return": (self.wall_end_ns - self.post_update_ns) * 1e-9,
            "total_wall": (self.wall_end_ns - self.wall_start_ns) * 1e-9,
            "simulated_actuation_period": CONTROLLER_DT_SECONDS,
            "deterministic_compute_bound_available": False,
        }


def constraint_margins(state: np.ndarray, step_index: int) -> ConstraintMargins:
    vector = np.asarray(state, dtype=float).reshape(-1)
    if vector.shape != (10,) or not np.all(np.isfinite(vector)):
        raise GazeboSecondSystemError("state must be finite with shape (10,)")
    if not isinstance(step_index, int) or not 0 <= step_index <= 200:
        raise GazeboSecondSystemError("step_index must lie in [0, 200]")
    uav_box = min(10.0 - abs(vector[0]), 10.0 - abs(vector[1]), vector[2] - 0.5, 5.0 - vector[2])
    uav_speed = 3.0 - float(np.linalg.norm(vector[3:6]))
    ugv_box = min(10.0 - abs(vector[6]), 10.0 - abs(vector[7]))
    ugv_speed = 2.5 - float(np.linalg.norm(vector[8:10]))
    separation = float(np.linalg.norm(vector[:2] - vector[6:8])) - 1.0
    return ConstraintMargins(min(uav_box, uav_speed), min(ugv_box, ugv_speed), separation, 200.0 - step_index)


def action_to_world_forces(
    state: np.ndarray,
    applied_action: np.ndarray,
    shift: CompoundShift,
) -> tuple[np.ndarray, np.ndarray]:
    vector = np.asarray(state, dtype=float).reshape(-1)
    action = np.asarray(applied_action, dtype=float).reshape(-1)
    shift.validate()
    if vector.shape != (10,) or action.shape != (5,):
        raise GazeboSecondSystemError("state/action dimensions must be (10,)/(5,)")
    if not np.all(np.isfinite(vector)) or not np.all(np.isfinite(action)):
        raise GazeboSecondSystemError("state and action must be finite")
    clipped = np.clip(action, -2.0, 2.0)
    uav_velocity = vector[3:6]
    ugv_velocity = vector[8:10]
    uav_force = (
        clipped[:3]
        - shift.uav_drag * uav_velocity
        - UAV_QUADRATIC_DRAG * np.linalg.norm(uav_velocity) * uav_velocity
    ) / shift.uav_mass
    ugv_force_xy = (
        clipped[3:] / (1.0 + shift.ugv_friction)
        - UGV_QUADRATIC_DRAG * np.linalg.norm(ugv_velocity) * ugv_velocity
    )
    return uav_force, np.array([ugv_force_xy[0], ugv_force_xy[1], 0.0], dtype=float)


class GazeboAirGroundEnv:
    """Synchronous, headless Gazebo environment for one rollout."""

    dt = CONTROLLER_DT_SECONDS

    def __init__(self, world_path: str | Path, shift: CompoundShift | None = None, horizon: int = 80):
        self.world_path = Path(world_path).resolve()
        if not self.world_path.is_file():
            raise GazeboSecondSystemError("Gazebo world file is missing")
        body = self.world_path.read_text(encoding="utf-8").lower()
        if any(token in body for token in ("http://", "https://", "fuel.gazebosim", "model://")):
            raise GazeboSecondSystemError("Gazebo world must be self-contained")
        self.shift = shift or CompoundShift()
        self.shift.validate()
        self.horizon = int(horizon)
        if self.horizon <= 0 or self.horizon > 200:
            raise GazeboSecondSystemError("horizon must lie in [1, 200]")

        try:
            import gz.math7 as gz_math
            import gz.sim8 as gz_sim
        except ImportError as exc:  # pragma: no cover - exercised on remote G0
            raise GazeboSecondSystemError("Gazebo Sim 8 Python bindings are unavailable") from exc
        self._math = gz_math
        self._sim = gz_sim
        self._fixture = gz_sim.TestFixture(str(self.world_path))
        self._fixture.on_pre_update(self._on_pre_update)
        self._fixture.on_post_update(self._on_post_update)
        self._fixture.finalize()
        self._server = self._fixture.server()
        if self._server is None:
            raise GazeboSecondSystemError("Gazebo TestFixture failed to finalize")
        self._uav_model: Any | None = None
        self._ugv_model: Any | None = None
        self._uav_link: Any | None = None
        self._ugv_link: Any | None = None
        self._requested_state = np.array([-4.0, -2.0, 2.0, 0.0, 0.0, 0.0, 4.0, 2.0, 0.0, 0.0])
        self.state = self._requested_state.copy()
        self._applied_action = np.zeros(5, dtype=float)
        self._uav_force = np.zeros(3, dtype=float)
        self._ugv_force = np.zeros(3, dtype=float)
        self._reset_pending = True
        self._callback_error: Exception | None = None
        self._step_wall_start_ns = 0
        self._first_pre_update_ns = 0
        self._last_post_update_ns = 0
        self.step_index = 0

    def _ensure_entities(self, ecm: Any) -> None:
        if self._uav_link is not None and self._ugv_link is not None:
            return
        world = self._sim.World(self._sim.world_entity(ecm))
        uav_id = world.model_by_name(ecm, "uav")
        ugv_id = world.model_by_name(ecm, "ugv")
        if uav_id == self._sim.K_NULL_ENTITY or ugv_id == self._sim.K_NULL_ENTITY:
            raise GazeboSecondSystemError("required uav/ugv model entity is missing")
        self._uav_model = self._sim.Model(uav_id)
        self._ugv_model = self._sim.Model(ugv_id)
        uav_link_id = self._uav_model.link_by_name(ecm, "body")
        ugv_link_id = self._ugv_model.link_by_name(ecm, "body")
        if uav_link_id == self._sim.K_NULL_ENTITY or ugv_link_id == self._sim.K_NULL_ENTITY:
            raise GazeboSecondSystemError("required body link entity is missing")
        self._uav_link = self._sim.Link(uav_link_id)
        self._ugv_link = self._sim.Link(ugv_link_id)
        self._uav_link.enable_velocity_checks(ecm)
        self._ugv_link.enable_velocity_checks(ecm)

    def _on_pre_update(self, _info: Any, ecm: Any) -> None:
        try:
            self._ensure_entities(ecm)
            now = time.perf_counter_ns()
            if self._step_wall_start_ns and not self._first_pre_update_ns:
                self._first_pre_update_ns = now
            if self._reset_pending:
                u = self._requested_state
                self._uav_model.set_world_pose_cmd(ecm, self._math.Pose3d(float(u[0]), float(u[1]), float(u[2]), 0.0, 0.0, 0.0))
                self._ugv_model.set_world_pose_cmd(ecm, self._math.Pose3d(float(u[6]), float(u[7]), 0.2, 0.0, 0.0, 0.0))
                self._uav_link.set_linear_velocity(ecm, self._math.Vector3d(float(u[3]), float(u[4]), float(u[5])))
                self._ugv_link.set_linear_velocity(ecm, self._math.Vector3d(float(u[8]), float(u[9]), 0.0))
                self._uav_link.set_angular_velocity(ecm, self._math.Vector3d(0.0, 0.0, 0.0))
                self._ugv_link.set_angular_velocity(ecm, self._math.Vector3d(0.0, 0.0, 0.0))
                self._reset_pending = False
            else:
                self._uav_link.add_world_force(ecm, self._math.Vector3d(*map(float, self._uav_force)))
                self._ugv_link.add_world_force(ecm, self._math.Vector3d(*map(float, self._ugv_force)))
        except Exception as exc:  # pragma: no cover - remote integration path
            self._callback_error = exc

    @staticmethod
    def _xyz(value: Any) -> np.ndarray:
        return np.array([float(value.x()), float(value.y()), float(value.z())], dtype=float)

    def _on_post_update(self, _info: Any, ecm: Any) -> None:
        try:
            self._ensure_entities(ecm)
            uav_pose = self._uav_link.world_pose(ecm)
            ugv_pose = self._ugv_link.world_pose(ecm)
            uav_velocity = self._uav_link.world_linear_velocity(ecm)
            ugv_velocity = self._ugv_link.world_linear_velocity(ecm)
            if any(value is None for value in (uav_pose, ugv_pose, uav_velocity, ugv_velocity)):
                raise GazeboSecondSystemError("Gazebo state component is unavailable")
            uav_position = self._xyz(uav_pose.pos())
            ugv_position = self._xyz(ugv_pose.pos())
            uav_v = self._xyz(uav_velocity)
            ugv_v = self._xyz(ugv_velocity)
            self.state = np.concatenate([uav_position, uav_v, ugv_position[:2], ugv_v[:2]])
            if not np.all(np.isfinite(self.state)):
                raise GazeboSecondSystemError("Gazebo returned a non-finite state")
            if self._step_wall_start_ns:
                self._last_post_update_ns = time.perf_counter_ns()
        except Exception as exc:  # pragma: no cover - remote integration path
            self._callback_error = exc

    def _run_iterations(self, iterations: int) -> None:
        self._callback_error = None
        result = self._server.run(True, int(iterations), False)
        if result is False:
            raise GazeboSecondSystemError("Gazebo server run failed")
        if self._callback_error is not None:
            raise GazeboSecondSystemError("Gazebo callback failed") from self._callback_error

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
        self._reset_pending = True
        self.step_index = 0
        self._run_iterations(2)
        margins = constraint_margins(self.state, self.step_index)
        return self._observation(), {"development_only": True, "margins": margins}

    def _observation(self) -> np.ndarray:
        observation = self.state.copy()
        observation[[0, 1, 2, 6, 7]] += self.shift.sensor_bias
        return observation.astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        control = np.asarray(action, dtype=float).reshape(-1)
        if control.shape != (5,) or not np.all(np.isfinite(control)):
            raise GazeboSecondSystemError("joint action must be finite with shape (5,)")
        control = np.clip(control, -2.0, 2.0)
        lag = self.shift.actuator_lag
        self._applied_action = lag * self._applied_action + (1.0 - lag) * control
        self._uav_force, self._ugv_force = action_to_world_forces(self.state, self._applied_action, self.shift)
        self._step_wall_start_ns = time.perf_counter_ns()
        self._first_pre_update_ns = 0
        self._last_post_update_ns = 0
        self._run_iterations(PHYSICS_ITERATIONS_PER_STEP)
        wall_end_ns = time.perf_counter_ns()
        if not self._first_pre_update_ns or not self._last_post_update_ns:
            raise GazeboSecondSystemError("Gazebo latency callbacks were not observed")
        receipt = GazeboStepReceipt(
            self._step_wall_start_ns,
            self._first_pre_update_ns,
            self._last_post_update_ns,
            wall_end_ns,
            PHYSICS_ITERATIONS_PER_STEP,
        )
        receipt.validate()
        self._step_wall_start_ns = 0
        self.step_index += 1
        margins = constraint_margins(self.state, self.step_index)
        violated = bool(np.any(margins.as_array() < 0.0))
        uav_goal = np.array([4.0, 2.0, 2.0])
        ugv_goal = np.array([-4.0, -2.0])
        distance = np.linalg.norm(self.state[:3] - uav_goal) + np.linalg.norm(self.state[6:8] - ugv_goal)
        reward = -0.05 * float(distance) - 0.005 * float(np.dot(control, control))
        return self._observation(), reward, violated, self.step_index >= self.horizon, {
            "development_only": True,
            "constraint_violation": violated,
            "margins": margins,
            "applied_action": self._applied_action.copy(),
            "latency_receipt": receipt,
        }
