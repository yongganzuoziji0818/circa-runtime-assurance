"""Planar-speed-projected command-persistent Gazebo driver."""

from __future__ import annotations

from typing import Any

import numpy as np

from .gazebo_second_system import GazeboSecondSystemError
from .gazebo_second_system_v3 import GazeboAirGroundEnvV3


class GazeboAirGroundEnvV4(GazeboAirGroundEnvV3):
    """Project each planar command onto a prospectively registered speed ball."""

    def __init__(
        self,
        *args: Any,
        uav_planar_speed_limit_mps: float,
        ugv_planar_speed_limit_mps: float,
        **kwargs: Any,
    ):
        limits = np.asarray(
            [uav_planar_speed_limit_mps, ugv_planar_speed_limit_mps], dtype=float
        )
        if not np.all(np.isfinite(limits)) or np.any(limits <= 0.0):
            raise GazeboSecondSystemError("planar speed limits must be finite and positive")
        self.uav_planar_speed_limit_mps = float(limits[0])
        self.ugv_planar_speed_limit_mps = float(limits[1])
        self.speed_projection_count = 0
        super().__init__(*args, **kwargs)

    @staticmethod
    def project_planar(velocity: np.ndarray, limit: float) -> tuple[np.ndarray, bool]:
        vector = np.asarray(velocity, dtype=float).reshape(-1).copy()
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise GazeboSecondSystemError("velocity must be finite with shape (3,)")
        if not np.isfinite(limit) or limit <= 0.0:
            raise GazeboSecondSystemError("speed limit must be finite and positive")
        norm = float(np.linalg.norm(vector[:2]))
        projected = norm > limit
        if projected:
            vector[:2] *= limit / norm
        return vector, projected

    def _set_commanded_velocity(self, ecm: Any) -> None:
        self._commanded_uav_velocity, uav_projected = self.project_planar(
            self._commanded_uav_velocity, self.uav_planar_speed_limit_mps
        )
        self._commanded_ugv_velocity, ugv_projected = self.project_planar(
            self._commanded_ugv_velocity, self.ugv_planar_speed_limit_mps
        )
        self.speed_projection_count += int(uav_projected or ugv_projected)
        super()._set_commanded_velocity(ecm)
