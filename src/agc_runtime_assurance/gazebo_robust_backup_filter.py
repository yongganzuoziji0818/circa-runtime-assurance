"""Auditable sampled-data robust-backup filter for CIRCA-GZ0-v7.

This is a validation instrument, not the CIRCA contribution.  It accepts a
nominal first action only when that action followed by a fixed outward backup
reaches a terminal nonclosing set while every tightened sampled state remains
outside the operational-separation boundary.  Otherwise it issues the backup
action and records whether even the backup tube was infeasible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gazebo_second_system import UAV_QUADRATIC_DRAG, UGV_QUADRATIC_DRAG


@dataclass(frozen=True)
class GazeboPlanarPlant:
    uav_mass: float
    uav_drag: float
    ugv_friction: float
    actuator_lag: float
    dt: float = 0.1
    speed_limit_mps: float | None = None

    def validate(self) -> None:
        values = np.asarray(
            [self.uav_mass, self.uav_drag, self.ugv_friction, self.actuator_lag, self.dt],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("plant parameters must be finite")
        if self.uav_mass <= 0.0 or self.uav_drag < 0.0 or self.ugv_friction < 0.0:
            raise ValueError("invalid mass, drag, or friction")
        if not 0.0 <= self.actuator_lag < 1.0 or self.dt <= 0.0:
            raise ValueError("invalid actuator lag or sample period")
        if self.speed_limit_mps is not None and (
            not np.isfinite(self.speed_limit_mps) or self.speed_limit_mps <= 0.0
        ):
            raise ValueError("speed limit must be positive when present")


@dataclass(frozen=True)
class RobustBackupConfig:
    operational_separation_m: float
    action_limit: float
    horizon_steps: int = 40
    terminal_margin_m: float = 0.10
    position_error_bound_m: float = 0.02
    velocity_error_bound_mps: float = 0.02
    relative_acceleration_error_bound_mps2: float = 0.05
    # gamma=1 in h[k+1] - h[k] + gamma*h[k] >= 0, hence retention=0.
    # Recursive safety is supplied by the verified backup suffix and terminal
    # nonclosing set rather than by a post-hoc stricter decay coefficient.
    barrier_retention: float = 0.0
    tolerance: float = 1e-9

    def validate(self) -> None:
        values = np.asarray(
            [
                self.operational_separation_m,
                self.action_limit,
                self.terminal_margin_m,
                self.position_error_bound_m,
                self.velocity_error_bound_mps,
                self.relative_acceleration_error_bound_mps2,
                self.barrier_retention,
                self.tolerance,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("backup-filter parameters must be finite")
        if self.operational_separation_m <= 0.0 or self.action_limit <= 0.0:
            raise ValueError("separation and action limit must be positive")
        if not isinstance(self.horizon_steps, int) or not 1 <= self.horizon_steps <= 200:
            raise ValueError("horizon_steps must lie in [1, 200]")
        if np.any(values[2:6] < 0.0):
            raise ValueError("terminal and uncertainty bounds must be non-negative")
        if not 0.0 <= self.barrier_retention <= 1.0 or self.tolerance < 0.0:
            raise ValueError("invalid barrier retention or tolerance")


@dataclass(frozen=True)
class BackupPlanEvaluation:
    feasible: bool
    terminal_reached: bool
    minimum_tightened_margin_m: float
    minimum_barrier_residual_m: float
    terminal_step: int | None
    predicted_steps: int
    reason: str


@dataclass(frozen=True)
class RobustBackupDecision:
    action: np.ndarray
    intervened: bool
    fail_closed: bool
    reason: str
    nominal_plan: BackupPlanEvaluation
    backup_plan: BackupPlanEvaluation


def _project_planar(vector: np.ndarray, limit: float | None) -> np.ndarray:
    output = np.asarray(vector, dtype=float).reshape(2).copy()
    if limit is not None:
        norm = float(np.linalg.norm(output))
        if norm > limit:
            output *= limit / norm
    return output


def propagate_planar_state(
    state: np.ndarray,
    applied_action: np.ndarray,
    command: np.ndarray,
    plant: GazeboPlanarPlant,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate the registered semi-implicit planar reference model one step."""

    plant.validate()
    vector = np.asarray(state, dtype=float).reshape(-1)
    previous = np.asarray(applied_action, dtype=float).reshape(-1)
    requested = np.asarray(command, dtype=float).reshape(-1)
    if vector.shape != (10,) or previous.shape != (5,) or requested.shape != (5,):
        raise ValueError("state/action dimensions must be (10,)/(5,)")
    if not all(np.all(np.isfinite(item)) for item in (vector, previous, requested)):
        raise ValueError("state and actions must be finite")

    requested = requested.copy()
    applied = plant.actuator_lag * previous + (1.0 - plant.actuator_lag) * requested
    uav_velocity = vector[3:5]
    ugv_velocity = vector[8:10]
    uav_acceleration = (
        applied[:2]
        - plant.uav_drag * uav_velocity
        - UAV_QUADRATIC_DRAG * np.linalg.norm(uav_velocity) * uav_velocity
    ) / plant.uav_mass
    ugv_acceleration = (
        applied[3:5] / (1.0 + plant.ugv_friction)
        - UGV_QUADRATIC_DRAG * np.linalg.norm(ugv_velocity) * ugv_velocity
    )
    next_state = vector.copy()
    next_state[3:5] = _project_planar(
        uav_velocity + plant.dt * uav_acceleration, plant.speed_limit_mps
    )
    next_state[8:10] = _project_planar(
        ugv_velocity + plant.dt * ugv_acceleration, plant.speed_limit_mps
    )
    next_state[:2] += plant.dt * next_state[3:5]
    next_state[6:8] += plant.dt * next_state[8:10]
    return next_state, applied


class RobustBackupSafetyFilter:
    """Binary minimal-intervention predictive filter with fail-closed recovery."""

    def __init__(self, plant: GazeboPlanarPlant, config: RobustBackupConfig):
        plant.validate()
        config.validate()
        self.plant = plant
        self.config = config

    def backup_action(self, state: np.ndarray, nominal_action: np.ndarray | None = None) -> np.ndarray:
        vector = np.asarray(state, dtype=float).reshape(-1)
        if vector.shape != (10,) or not np.all(np.isfinite(vector)):
            raise ValueError("state must be finite with shape (10,)")
        relative = vector[:2] - vector[6:8]
        separation = float(np.linalg.norm(relative))
        if separation <= 1e-12:
            direction = np.array([1.0, 0.0])
        else:
            direction = relative / separation
        action = np.zeros(5, dtype=float)
        if nominal_action is not None:
            nominal = np.asarray(nominal_action, dtype=float).reshape(-1)
            if nominal.shape != (5,) or not np.all(np.isfinite(nominal)):
                raise ValueError("nominal action must be finite with shape (5,)")
            action[2] = np.clip(nominal[2], -self.config.action_limit, self.config.action_limit)
        action[:2] = self.config.action_limit * direction
        action[3:5] = -self.config.action_limit * direction
        return action

    def _margin(self, state: np.ndarray, position_radius: float) -> float:
        separation = float(np.linalg.norm(state[:2] - state[6:8]))
        return separation - self.config.operational_separation_m - position_radius

    def _terminal(self, state: np.ndarray, position_radius: float, velocity_radius: float) -> bool:
        relative_position = state[:2] - state[6:8]
        separation = float(np.linalg.norm(relative_position))
        if separation <= 1e-12:
            return False
        relative_velocity = state[3:5] - state[8:10]
        radial_rate = float(relative_position @ relative_velocity / separation)
        margin = self._margin(state, position_radius)
        return bool(
            margin >= self.config.terminal_margin_m
            and radial_rate - velocity_radius >= 0.0
        )

    def evaluate_plan(
        self,
        state: np.ndarray,
        applied_action: np.ndarray,
        first_action: np.ndarray,
    ) -> BackupPlanEvaluation:
        vector = np.asarray(state, dtype=float).reshape(-1).copy()
        applied = np.asarray(applied_action, dtype=float).reshape(-1).copy()
        first = np.asarray(first_action, dtype=float).reshape(-1).copy()
        if vector.shape != (10,) or applied.shape != (5,) or first.shape != (5,):
            raise ValueError("plan state/action dimensions must be (10,)/(5,)")
        if not all(np.all(np.isfinite(item)) for item in (vector, applied, first)):
            raise ValueError("plan state and actions must be finite")
        first = np.clip(first, -self.config.action_limit, self.config.action_limit)

        position_radius = self.config.position_error_bound_m
        velocity_radius = self.config.velocity_error_bound_mps
        previous_margin = self._margin(vector, position_radius)
        minimum_margin = previous_margin
        minimum_residual = float("inf")
        if previous_margin < -self.config.tolerance:
            return BackupPlanEvaluation(
                False, False, previous_margin, float("-inf"), None, 0,
                "initial_tightened_operational_set_violated",
            )

        for step in range(self.config.horizon_steps):
            command = first if step == 0 else self.backup_action(vector)
            vector, applied = propagate_planar_state(vector, applied, command, self.plant)
            position_radius += (
                self.plant.dt * velocity_radius
                + 0.5 * self.plant.dt**2 * self.config.relative_acceleration_error_bound_mps2
            )
            velocity_radius += self.plant.dt * self.config.relative_acceleration_error_bound_mps2
            margin = self._margin(vector, position_radius)
            residual = margin - self.config.barrier_retention * previous_margin
            minimum_margin = min(minimum_margin, margin)
            minimum_residual = min(minimum_residual, residual)
            if margin < -self.config.tolerance:
                return BackupPlanEvaluation(
                    False, False, minimum_margin, minimum_residual, None, step + 1,
                    f"tightened_operational_set_failed_at_step_{step + 1}",
                )
            if residual < -self.config.tolerance:
                return BackupPlanEvaluation(
                    False, False, minimum_margin, minimum_residual, None, step + 1,
                    f"discrete_barrier_failed_at_step_{step + 1}",
                )
            if self._terminal(vector, position_radius, velocity_radius):
                return BackupPlanEvaluation(
                    True, True, minimum_margin, minimum_residual, step + 1, step + 1,
                    "terminal_robust_nonclosing_set_reached",
                )
            previous_margin = margin
        return BackupPlanEvaluation(
            False, False, minimum_margin, minimum_residual, None,
            self.config.horizon_steps, "terminal_set_not_reached_within_horizon",
        )

    def decide(
        self,
        state: np.ndarray,
        applied_action: np.ndarray,
        nominal_action: np.ndarray,
    ) -> RobustBackupDecision:
        nominal = np.asarray(nominal_action, dtype=float).reshape(-1)
        if nominal.shape != (5,) or not np.all(np.isfinite(nominal)):
            raise ValueError("nominal action must be finite with shape (5,)")
        nominal = np.clip(nominal, -self.config.action_limit, self.config.action_limit)
        backup = self.backup_action(state, nominal)
        nominal_plan = self.evaluate_plan(state, applied_action, nominal)
        backup_plan = self.evaluate_plan(state, applied_action, backup)
        if nominal_plan.feasible:
            return RobustBackupDecision(
                nominal, False, False, "nominal_with_verified_backup_tube", nominal_plan, backup_plan
            )
        if backup_plan.feasible:
            return RobustBackupDecision(
                backup, True, False, "backup_filter_intervention", nominal_plan, backup_plan
            )
        return RobustBackupDecision(
            backup, True, True, "backup_tube_infeasible_fail_closed", nominal_plan, backup_plan
        )
