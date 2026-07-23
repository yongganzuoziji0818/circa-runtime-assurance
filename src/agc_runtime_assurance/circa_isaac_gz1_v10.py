"""Isaac Sim/PhysX adapter for the frozen CIRCA-GZ1-v10 route.

The module contains no top-level Isaac imports.  ``SimulationApp`` must be
created by the authorized launcher before an environment instance is built.
The state/action contract intentionally matches the immutable GZ0-v9 adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .environment import CompoundShift
from .gazebo_second_system import (
    CONTROLLER_DT_SECONDS,
    PHYSICS_ITERATIONS_PER_STEP,
    action_to_world_forces,
    constraint_margins,
)


class CircaIsaacGZ1V10Error(RuntimeError):
    """Fail-closed Isaac adapter error."""


class IsaacAirGroundEnvV10:
    """One reusable, headless 1-UAV/1-UGV PhysX environment."""

    dt = CONTROLLER_DT_SECONDS
    physics_dt = CONTROLLER_DT_SECONDS / PHYSICS_ITERATIONS_PER_STEP

    def __init__(self, simulation_app: Any, world_path: str | Path, *, horizon: int = 80):
        path = Path(world_path).resolve()
        if not path.is_file() or path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
            raise CircaIsaacGZ1V10Error("frozen Isaac world is absent or invalid")
        body = path.read_text(encoding="utf-8") if path.suffix.lower() == ".usda" else ""
        if any(token in body.lower() for token in ("http://", "https://", "omniverse://")):
            raise CircaIsaacGZ1V10Error("Isaac world must be self-contained")
        self.simulation_app = simulation_app
        self.world_path = path
        self.horizon = int(horizon)
        if self.horizon != 80:
            raise CircaIsaacGZ1V10Error("frozen horizon must equal 80")

        # Isaac imports are intentionally delayed until SimulationApp exists.
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import RigidPrim
        from isaacsim.core.experimental.utils.stage import is_stage_loading

        context = omni.usd.get_context()
        if not context.open_stage(str(path)):
            raise CircaIsaacGZ1V10Error("Isaac failed to open the frozen USD stage")
        self.simulation_app.update()
        self.simulation_app.update()
        while is_stage_loading():
            self.simulation_app.update()

        try:
            World.clear_instance()
        except Exception:
            pass
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.physics_dt,
            rendering_dt=self.physics_dt,
            backend="numpy",
        )
        self.uav = RigidPrim(
            prim_paths_expr="/World/UAV",
            name="circa_gz1_v10_uav",
            contact_filter_prim_paths_expr=["/World/UGV"],
            max_contact_count=16,
        )
        self.ugv = RigidPrim(
            prim_paths_expr="/World/UGV",
            name="circa_gz1_v10_ugv",
            contact_filter_prim_paths_expr=["/World/UAV"],
            max_contact_count=16,
        )
        self.world.scene.add(self.uav)
        self.world.scene.add(self.ugv)
        self.world.reset(soft=False)
        self.world._physics_context.set_gravity(0.0)

        self.shift = CompoundShift()
        self.driver = "command_persistent_unbounded_v3"
        self.planar_speed_limit_mps: float | None = None
        self.state = np.array([-4.0, -2.0, 2.0, 0.0, 0.0, 0.0, 4.0, 2.0, 0.0, 0.0])
        self._applied_action = np.zeros(5, dtype=float)
        self._commanded_uav_velocity = np.zeros(3, dtype=float)
        self._commanded_ugv_velocity = np.zeros(3, dtype=float)
        self.step_index = 0
        self.body_state_history: list[np.ndarray] = []
        self.contact_impulse_history: list[np.ndarray] = []

    def configure(self, *, driver: str, shift: CompoundShift, planar_speed_limit_mps: float | None) -> None:
        if driver not in {"command_persistent_unbounded_v3", "planar_speed_projected_v4"}:
            raise CircaIsaacGZ1V10Error("unknown frozen driver")
        shift.validate()
        if driver == "planar_speed_projected_v4":
            if planar_speed_limit_mps is None or not np.isfinite(planar_speed_limit_mps) or planar_speed_limit_mps <= 0:
                raise CircaIsaacGZ1V10Error("projected driver requires a positive speed limit")
        self.driver = driver
        self.shift = shift
        self.planar_speed_limit_mps = planar_speed_limit_mps

    @staticmethod
    def _pose_arrays(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        uav_position = np.asarray(state[:3], dtype=float).reshape(1, 3)
        ugv_position = np.array([[state[6], state[7], 0.2]], dtype=float)
        identity = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=float)
        return uav_position, ugv_position, identity, identity.copy()

    @staticmethod
    def _velocity_six(linear: np.ndarray) -> np.ndarray:
        return np.concatenate((np.asarray(linear, dtype=float), np.zeros(3))).reshape(1, 6)

    def _set_pose_velocity(self, state: np.ndarray) -> None:
        up, gp, uq, gq = self._pose_arrays(state)
        self.uav.set_world_poses(positions=up, orientations=uq)
        self.ugv.set_world_poses(positions=gp, orientations=gq)
        self.uav.set_velocities(self._velocity_six(state[3:6]))
        self.ugv.set_velocities(self._velocity_six(np.array([state[8], state[9], 0.0])))

    @staticmethod
    def _first_row(value: Any, width: int) -> np.ndarray:
        result = np.asarray(value, dtype=float)
        if result.ndim == 1:
            result = result.reshape(1, -1)
        if result.shape[0] < 1 or result.shape[1] < width:
            raise CircaIsaacGZ1V10Error("Isaac rigid-body state has an unexpected shape")
        return result[0, :width].copy()

    def _read_state(self) -> tuple[np.ndarray, np.ndarray]:
        up, uq = self.uav.get_world_poses()
        gp, gq = self.ugv.get_world_poses()
        uv = self._first_row(self.uav.get_velocities(), 6)
        gv = self._first_row(self.ugv.get_velocities(), 6)
        up0 = self._first_row(up, 3)
        gp0 = self._first_row(gp, 3)
        uq0 = self._first_row(uq, 4)
        gq0 = self._first_row(gq, 4)
        state = np.concatenate((up0, uv[:3], gp0[:2], gv[:2]))
        body_state = np.stack(
            (
                np.concatenate((up0, uq0, uv[:3], uv[3:6])),
                np.concatenate((gp0, gq0, gv[:3], gv[3:6])),
            )
        )
        if state.shape != (10,) or body_state.shape != (2, 13) or not np.all(np.isfinite(body_state)):
            raise CircaIsaacGZ1V10Error("Isaac returned invalid rigid-body state")
        return state, body_state

    def _net_contact_impulse(self) -> np.ndarray:
        impulses = np.zeros((2, 3), dtype=float)
        for index, prim in enumerate((self.uav, self.ugv)):
            try:
                force = np.asarray(prim.get_net_contact_forces(dt=self.physics_dt), dtype=float)
            except TypeError:
                force = np.asarray(prim.get_net_contact_forces(), dtype=float)
            if force.size:
                impulses[index] = force.reshape(-1, 3).sum(axis=0) * self.physics_dt
        if not np.all(np.isfinite(impulses)):
            raise CircaIsaacGZ1V10Error("Isaac returned invalid contact impulse")
        return impulses

    def reset(
        self,
        *,
        seed: int,
        interaction_close: bool = False,
        initial_state: np.ndarray | None = None,
        position_jitter_scale: float = 0.03,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if interaction_close:
            raise CircaIsaacGZ1V10Error("interaction_close shortcut is not part of frozen GZ1")
        state = (
            np.array([-4.0, -2.0, 2.0, 0.0, 0.0, 0.0, 4.0, 2.0, 0.0, 0.0])
            if initial_state is None
            else np.asarray(initial_state, dtype=float).reshape(-1).copy()
        )
        if state.shape != (10,) or not np.all(np.isfinite(state)):
            raise CircaIsaacGZ1V10Error("initial state must be finite with shape (10,)")
        if position_jitter_scale:
            rng = np.random.default_rng(int(seed))
            state[[0, 1, 6, 7]] += rng.normal(0.0, float(position_jitter_scale), 4)
        self._applied_action.fill(0.0)
        self._commanded_uav_velocity = state[3:6].copy()
        self._commanded_ugv_velocity = np.array([state[8], state[9], 0.0], dtype=float)
        self._set_pose_velocity(state)
        self.world.step(render=False)
        self.state, _ = self._read_state()
        self.step_index = 0
        self.body_state_history = []
        self.contact_impulse_history = []
        return self._observation(), {"development_only": False, "margins": constraint_margins(self.state, 0)}

    def _observation(self) -> np.ndarray:
        observation = self.state.copy()
        observation[[0, 1, 2, 6, 7]] += self.shift.sensor_bias
        return observation.astype(np.float32)

    @staticmethod
    def _project_planar(velocity: np.ndarray, limit: float) -> np.ndarray:
        result = np.asarray(velocity, dtype=float).copy()
        norm = float(np.linalg.norm(result[:2]))
        if norm > limit:
            result[:2] *= limit / norm
        return result

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        control = np.asarray(action, dtype=float).reshape(-1)
        if control.shape != (5,) or not np.all(np.isfinite(control)):
            raise CircaIsaacGZ1V10Error("action must be finite with shape (5,)")
        control = np.clip(control, -2.0, 2.0)
        lag = float(self.shift.actuator_lag)
        self._applied_action = lag * self._applied_action + (1.0 - lag) * control
        uav_accel, ugv_accel = action_to_world_forces(self.state, self._applied_action, self.shift)
        contact_sum = np.zeros((2, 3), dtype=float)
        for _ in range(PHYSICS_ITERATIONS_PER_STEP):
            self._commanded_uav_velocity += uav_accel * self.physics_dt
            self._commanded_ugv_velocity += ugv_accel * self.physics_dt
            if self.driver == "planar_speed_projected_v4":
                assert self.planar_speed_limit_mps is not None
                self._commanded_uav_velocity = self._project_planar(
                    self._commanded_uav_velocity, self.planar_speed_limit_mps
                )
                self._commanded_ugv_velocity = self._project_planar(
                    self._commanded_ugv_velocity, self.planar_speed_limit_mps
                )
            self.uav.set_velocities(self._velocity_six(self._commanded_uav_velocity))
            self.ugv.set_velocities(self._velocity_six(self._commanded_ugv_velocity))
            self.world.step(render=False)
            contact_sum += self._net_contact_impulse()
        self.state, body_state = self._read_state()
        self.body_state_history.append(body_state)
        self.contact_impulse_history.append(contact_sum)
        self.step_index += 1
        margins = constraint_margins(self.state, self.step_index)
        violated = bool(np.any(margins.as_array() < 0.0))
        uav_goal = np.array([4.0, 2.0, 2.0])
        ugv_goal = np.array([-4.0, -2.0])
        distance = np.linalg.norm(self.state[:3] - uav_goal) + np.linalg.norm(self.state[6:8] - ugv_goal)
        reward = -0.05 * float(distance) - 0.005 * float(np.dot(control, control))
        return self._observation(), reward, violated, self.step_index >= self.horizon, {
            "margins": margins,
            "applied_action": self._applied_action.copy(),
            "solver_substeps": PHYSICS_ITERATIONS_PER_STEP,
        }

    def close(self) -> None:
        try:
            self.world.stop()
        except Exception:
            pass
