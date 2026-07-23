"""Frozen shared 1U1G task adapter for baseline-comparison G0.

The adapter supplies one nominal policy and one conservative next-state safety
contract to every method.  It is development-only and is not platform evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .environment import CompoundShift
from .risk import ConstraintMargins
from .sandbox_model import (
    air_ground_augmented_matrices,
    sandbox_axis_aligned_state_constraints,
)


@dataclass(frozen=True)
class AffineConstraintBundle:
    A: np.ndarray
    b: np.ndarray
    separation_reference: np.ndarray
    contract_fingerprint: str


class SandboxComparisonTask:
    """Deterministic nominal policy and one-step affine safety constraints."""

    def __init__(
        self,
        *,
        shift: CompoundShift | None = None,
        uav_goal: tuple[float, float, float] = (4.0, 2.0, 2.0),
        ugv_goal: tuple[float, float] = (-4.0, -2.0),
        uav_position_gain: float = 0.8,
        uav_velocity_gain: float = 1.1,
        ugv_position_gain: float = 0.6,
        ugv_velocity_gain: float = 1.0,
        minimum_separation: float = 1.0,
        action_limit: float = 2.0,
    ):
        self.shift = shift or CompoundShift()
        self.shift.validate()
        self.uav_goal = np.asarray(uav_goal, dtype=float)
        self.ugv_goal = np.asarray(ugv_goal, dtype=float)
        gains = np.asarray(
            [uav_position_gain, uav_velocity_gain,
             ugv_position_gain, ugv_velocity_gain], dtype=float,
        )
        if self.uav_goal.shape != (3,) or self.ugv_goal.shape != (2,):
            raise ValueError("goals must have shapes (3,) and (2,)")
        if not np.all(np.isfinite(self.uav_goal)) or not np.all(np.isfinite(self.ugv_goal)):
            raise ValueError("goals must be finite")
        if not np.all(np.isfinite(gains)) or np.any(gains <= 0.0):
            raise ValueError("policy gains must be finite and positive")
        if not np.isfinite(minimum_separation) or minimum_separation <= 0.0:
            raise ValueError("minimum separation must be finite and positive")
        if not np.isfinite(action_limit) or action_limit <= 0.0:
            raise ValueError("action limit must be finite and positive")
        self.uav_position_gain, self.uav_velocity_gain = map(float, gains[:2])
        self.ugv_position_gain, self.ugv_velocity_gain = map(float, gains[2:])
        self.minimum_separation = float(minimum_separation)
        self.action_lower = -float(action_limit) * np.ones(5)
        self.action_upper = float(action_limit) * np.ones(5)
        self.A_model, self.B_model = air_ground_augmented_matrices(self.shift)
        self.state_lower, self.state_upper = sandbox_axis_aligned_state_constraints()
        self._relative_position_map = np.zeros((2, 15), dtype=float)
        self._relative_position_map[0, 0] = 1.0
        self._relative_position_map[1, 1] = 1.0
        self._relative_position_map[0, 6] = -1.0
        self._relative_position_map[1, 7] = -1.0
        self.nominal_policy_fingerprint = self._policy_fingerprint()
        self.constraint_contract_fingerprint = self._constraint_fingerprint()

    @staticmethod
    def _update_array(hasher: "hashlib._Hash", array: np.ndarray) -> None:
        canonical = np.ascontiguousarray(array, dtype="<f8")
        hasher.update(str(canonical.shape).encode("ascii"))
        hasher.update(canonical.tobytes())

    def _policy_fingerprint(self) -> str:
        hasher = hashlib.sha256()
        for array in (
            self.uav_goal, self.ugv_goal,
            np.array([
                self.uav_position_gain, self.uav_velocity_gain,
                self.ugv_position_gain, self.ugv_velocity_gain,
            ]),
            self.action_lower, self.action_upper,
        ):
            self._update_array(hasher, array)
        return hasher.hexdigest()

    def _constraint_fingerprint(self) -> str:
        hasher = hashlib.sha256()
        for array in (
            self.A_model, self.B_model, self.state_lower, self.state_upper,
            self.action_lower, self.action_upper,
            np.array([self.minimum_separation]),
        ):
            self._update_array(hasher, array)
        return hasher.hexdigest()

    @staticmethod
    def _state(augmented_state: np.ndarray) -> np.ndarray:
        state = np.asarray(augmented_state, dtype=float).reshape(-1)
        if state.shape != (15,) or not np.all(np.isfinite(state)):
            raise ValueError("augmented state must be a finite vector of shape (15,)")
        return state

    def nominal_action(self, augmented_state: np.ndarray) -> np.ndarray:
        """Return the shared clipped PD goal-tracking action."""
        state = self._state(augmented_state)
        uav = (
            self.uav_position_gain * (self.uav_goal - state[:3])
            - self.uav_velocity_gain * state[3:6]
        )
        ugv = (
            self.ugv_position_gain * (self.ugv_goal - state[6:8])
            - self.ugv_velocity_gain * state[8:10]
        )
        return np.clip(np.concatenate([uav, ugv]), self.action_lower, self.action_upper)

    def point_constraint_margins(
        self, augmented_state: np.ndarray, *, step_index: int,
    ) -> ConstraintMargins:
        """Evaluate the sandbox's exact current-state margins without stepping."""
        state = self._state(augmented_state)
        if not isinstance(step_index, int) or not 0 <= step_index <= 200:
            raise ValueError("step_index must be an integer in [0, 200]")
        uav_box = min(
            10.0 - abs(state[0]), 10.0 - abs(state[1]),
            state[2] - 0.5, 5.0 - state[2],
        )
        uav_speed = 3.0 - float(np.linalg.norm(state[3:6]))
        ugv_box = min(10.0 - abs(state[6]), 10.0 - abs(state[7]))
        ugv_speed = 2.5 - float(np.linalg.norm(state[8:10]))
        coupling = float(np.linalg.norm(state[:2] - state[6:8])) - self.minimum_separation
        return ConstraintMargins(
            min(uav_box, uav_speed), min(ugv_box, ugv_speed),
            coupling, 200.0 - step_index,
        )

    def next_state_constraints(
        self,
        augmented_state: np.ndarray,
        *,
        separation_reference_action: np.ndarray | None = None,
    ) -> AffineConstraintBundle:
        """Return sufficient affine constraints for the next sandbox state.

        The state box is a sufficient inner approximation of the sandbox's
        norm constraints.  Separation uses a tangent lower bound of squared
        distance at the frozen reference action; satisfying the tangent implies
        the true next-step separation constraint.
        """
        state = self._state(augmented_state)
        reference_action = (
            self.nominal_action(state)
            if separation_reference_action is None
            else np.asarray(separation_reference_action, dtype=float).reshape(-1)
        )
        if reference_action.shape != (5,) or not np.all(np.isfinite(reference_action)):
            raise ValueError("separation reference action must be finite with shape (5,)")
        reference_action = np.clip(reference_action, self.action_lower, self.action_upper)
        constant = self.A_model @ state
        A_rows = [self.B_model, -self.B_model]
        b_rows = [self.state_upper - constant, constant - self.state_lower]

        relative_constant = self._relative_position_map @ constant
        relative_control = self._relative_position_map @ self.B_model
        reference = relative_constant + relative_control @ reference_action
        if np.linalg.norm(reference) <= 1e-12:
            raise ValueError("separation tangent is undefined at zero relative reference")
        separation_A = -2.0 * reference.reshape(1, 2) @ relative_control
        separation_b = np.array([
            2.0 * float(reference @ relative_constant)
            - float(reference @ reference)
            - self.minimum_separation ** 2
        ])
        A_rows.append(separation_A)
        b_rows.append(separation_b)
        return AffineConstraintBundle(
            np.vstack(A_rows), np.concatenate(b_rows), reference.copy(),
            self.constraint_contract_fingerprint,
        )

    def postcheck_next_state(
        self, augmented_state: np.ndarray, action: np.ndarray,
    ) -> bool:
        """Check the exact next state against the frozen sufficient contract."""
        state = self._state(augmented_state)
        control = np.asarray(action, dtype=float).reshape(-1)
        if control.shape != (5,) or not np.all(np.isfinite(control)):
            raise ValueError("action must be finite with shape (5,)")
        next_state = self.A_model @ state + self.B_model @ control
        in_box = bool(np.all(next_state >= self.state_lower - 1e-9)) and bool(
            np.all(next_state <= self.state_upper + 1e-9)
        )
        separation = np.linalg.norm(self._relative_position_map @ next_state)
        return in_box and bool(separation >= self.minimum_separation - 1e-9)
