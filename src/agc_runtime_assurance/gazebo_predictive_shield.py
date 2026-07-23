"""Transparent actuation-consistent separation shield for GZ0-v5 controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PredictiveShieldDecision:
    action: np.ndarray
    intervened: bool
    separation_m: float
    closing_speed_mps: float
    trigger_distance_m: float


class PredictiveSeparationShield:
    """Apply opposing outward actions inside a physics-derived stopping guard."""

    def __init__(self, *, minimum_relative_deceleration_mps2: float, action_limit: float):
        values = np.asarray([minimum_relative_deceleration_mps2, action_limit], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("shield deceleration and action limit must be finite and positive")
        self.minimum_relative_deceleration_mps2 = float(values[0])
        self.action_limit = float(values[1])

    def decide(
        self,
        state: np.ndarray,
        nominal_action: np.ndarray,
        *,
        operational_separation_m: float,
        reaction_time_s: float,
    ) -> PredictiveShieldDecision:
        vector = np.asarray(state, dtype=float).reshape(-1)
        nominal = np.asarray(nominal_action, dtype=float).reshape(-1)
        if vector.shape not in {(10,), (15,)} or not np.all(np.isfinite(vector)):
            raise ValueError("state must be finite with shape (10,) or (15,)")
        if nominal.shape != (5,) or not np.all(np.isfinite(nominal)):
            raise ValueError("nominal action must be finite with shape (5,)")
        parameters = np.asarray([operational_separation_m, reaction_time_s], dtype=float)
        if not np.all(np.isfinite(parameters)) or operational_separation_m <= 0.0 or reaction_time_s < 0.0:
            raise ValueError("invalid separation or reaction time")
        relative_position = vector[:2] - vector[6:8]
        separation = float(np.linalg.norm(relative_position))
        if separation <= 1e-12:
            raise ValueError("relative direction is undefined at zero separation")
        direction = relative_position / separation
        relative_velocity = vector[3:5] - vector[8:10]
        closing_speed = max(0.0, -float(relative_velocity @ direction))
        trigger_distance = float(
            operational_separation_m
            + closing_speed * reaction_time_s
            + closing_speed * closing_speed / (2.0 * self.minimum_relative_deceleration_mps2)
        )
        intervene = bool(closing_speed > 0.0 and separation <= trigger_distance)
        action = np.clip(nominal, -self.action_limit, self.action_limit)
        if intervene:
            action = action.copy()
            action[:2] = self.action_limit * direction
            action[3:5] = -self.action_limit * direction
        return PredictiveShieldDecision(
            action, intervene, separation, closing_speed, trigger_distance
        )
