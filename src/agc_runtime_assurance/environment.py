"""Small deterministic 1-UAV + 1-UGV development environment.

This is a G0 contract and fault-injection sandbox.  It is not a formal or sealed
experiment environment and its outputs cannot support paper claims.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .risk import ConstraintMargins


@dataclass(frozen=True)
class CompoundShift:
    uav_mass: float = 1.0
    uav_drag: float = 0.10
    ugv_friction: float = 0.10
    actuator_lag: float = 0.0
    sensor_bias: float = 0.0

    def validate(self) -> None:
        if self.uav_mass <= 0 or self.uav_drag < 0 or self.ugv_friction < 0:
            raise ValueError("invalid dynamics shift")
        if not 0.0 <= self.actuator_lag < 1.0:
            raise ValueError("actuator_lag must lie in [0, 1)")


class AirGroundRuntimeEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, shift: CompoundShift | None = None, horizon: int = 200):
        super().__init__()
        self.shift = shift or CompoundShift()
        self.shift.validate()
        self.horizon = int(horizon)
        self.dt = 0.1
        self.action_space = spaces.Box(-2.0, 2.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32)
        self.state = np.zeros(10, dtype=float)
        self._applied_action = np.zeros(5, dtype=float)
        self.step_index = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        jitter = self.np_random.normal(0.0, 0.03, size=4)
        self.state = np.array([-4.0, -2.0, 2.0, 0.0, 0.0, 0.0, 4.0, 2.0, 0.0, 0.0])
        self.state[[0, 1, 6, 7]] += jitter
        self._applied_action.fill(0.0)
        self.step_index = 0
        return self._observation(), {"development_only": True, "margins": self.constraint_margins()}

    def _observation(self) -> np.ndarray:
        obs = self.state.copy()
        obs[[0, 1, 2, 6, 7]] += self.shift.sensor_bias
        return obs.astype(np.float32)

    def constraint_margins(self) -> ConstraintMargins:
        uav_pos, uav_vel = self.state[:3], self.state[3:6]
        ugv_pos, ugv_vel = self.state[6:8], self.state[8:10]
        uav_box = min(10.0 - abs(uav_pos[0]), 10.0 - abs(uav_pos[1]), uav_pos[2] - 0.5, 5.0 - uav_pos[2])
        uav_speed = 3.0 - float(np.linalg.norm(uav_vel))
        ugv_box = min(10.0 - abs(ugv_pos[0]), 10.0 - abs(ugv_pos[1]))
        ugv_speed = 2.5 - float(np.linalg.norm(ugv_vel))
        separation = float(np.linalg.norm(uav_pos[:2] - ugv_pos)) - 1.0
        mission = 200.0 - self.step_index
        return ConstraintMargins(min(uav_box, uav_speed), min(ugv_box, ugv_speed), separation, mission)

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=float), -2.0, 2.0)
        if action.shape != (5,):
            raise ValueError("joint action must have shape (5,)")
        lag = self.shift.actuator_lag
        self._applied_action = lag * self._applied_action + (1.0 - lag) * action

        uav_acc = (self._applied_action[:3] - self.shift.uav_drag * self.state[3:6]) / self.shift.uav_mass
        ugv_acc = self._applied_action[3:] / (1.0 + self.shift.ugv_friction)
        self.state[3:6] += self.dt * uav_acc
        self.state[:3] += self.dt * self.state[3:6]
        self.state[8:10] += self.dt * ugv_acc
        self.state[6:8] += self.dt * self.state[8:10]
        self.step_index += 1

        margins = self.constraint_margins()
        violated = bool(np.any(margins.as_array() < 0.0))
        uav_goal = np.array([4.0, 2.0, 2.0])
        ugv_goal = np.array([-4.0, -2.0])
        distance = np.linalg.norm(self.state[:3] - uav_goal) + np.linalg.norm(self.state[6:8] - ugv_goal)
        reward = -0.05 * float(distance) - 0.005 * float(np.dot(action, action))
        terminated = violated
        truncated = self.step_index >= self.horizon
        info = {
            "development_only": True,
            "constraint_violation": violated,
            "margins": margins,
            "applied_action": self._applied_action.copy(),
        }
        return self._observation(), reward, terminated, truncated, info
