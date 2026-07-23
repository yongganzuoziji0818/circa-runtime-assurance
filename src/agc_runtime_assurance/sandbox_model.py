"""Auditable linear model and box bounds for the development-only 1U1G sandbox.

Nothing in this module is evidence for a flight/ground-vehicle platform.  It
extracts the exact discrete dynamics implemented by ``AirGroundRuntimeEnv`` so
that G0 assurance contracts can be falsified without silently changing models.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product

import numpy as np

from .environment import CompoundShift
from .risk import ConstraintMargins


@dataclass(frozen=True)
class ShiftParameterBox:
    uav_mass: tuple[float, float]
    uav_drag: tuple[float, float]
    ugv_friction: tuple[float, float]
    actuator_lag: tuple[float, float]

    def validate(self) -> None:
        for name, bounds in (
            ("uav_mass", self.uav_mass), ("uav_drag", self.uav_drag),
            ("ugv_friction", self.ugv_friction),
            ("actuator_lag", self.actuator_lag),
        ):
            if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
                raise ValueError(f"invalid {name} bounds")
        if self.uav_mass[0] <= 0.0 or self.uav_drag[0] < 0.0:
            raise ValueError("mass must be positive and drag non-negative")
        if self.ugv_friction[0] < 0.0:
            raise ValueError("friction must be non-negative")
        if self.actuator_lag[0] < 0.0 or self.actuator_lag[1] >= 1.0:
            raise ValueError("actuator lag bounds must lie in [0, 1)")

    def vertices(self) -> tuple[CompoundShift, ...]:
        self.validate()
        return tuple(
            CompoundShift(
                uav_mass=mass, uav_drag=drag, ugv_friction=friction,
                actuator_lag=lag, sensor_bias=0.0,
            )
            for mass, drag, friction, lag in product(
                self.uav_mass, self.uav_drag, self.ugv_friction,
                self.actuator_lag,
            )
        )


@dataclass(frozen=True)
class ModelUncertaintyEnvelope:
    disturbance_radius: np.ndarray
    max_abs_A_delta: np.ndarray
    max_abs_B_delta: np.ndarray
    vertex_count: int
    fingerprint: str


def air_ground_augmented_matrices(
    shift: CompoundShift, *, dt: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact matrices for z=[environment state, previous applied action]."""

    shift.validate()
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    A = np.zeros((15, 15), dtype=float)
    B = np.zeros((15, 5), dtype=float)
    lag = shift.actuator_lag

    for axis in range(3):
        position, velocity, applied, control = axis, 3 + axis, 10 + axis, axis
        velocity_factor = 1.0 - dt * shift.uav_drag / shift.uav_mass
        applied_factor = dt * lag / shift.uav_mass
        control_factor = dt * (1.0 - lag) / shift.uav_mass
        A[velocity, velocity] = velocity_factor
        A[velocity, applied] = applied_factor
        B[velocity, control] = control_factor
        A[position, position] = 1.0
        A[position, velocity] = dt * velocity_factor
        A[position, applied] = dt * applied_factor
        B[position, control] = dt * control_factor
        A[applied, applied] = lag
        B[applied, control] = 1.0 - lag

    ground_gain = dt / (1.0 + shift.ugv_friction)
    for axis in range(2):
        position, velocity, applied, control = 6 + axis, 8 + axis, 13 + axis, 3 + axis
        A[velocity, velocity] = 1.0
        A[velocity, applied] = ground_gain * lag
        B[velocity, control] = ground_gain * (1.0 - lag)
        A[position, position] = 1.0
        A[position, velocity] = dt
        A[position, applied] = dt * ground_gain * lag
        B[position, control] = dt * ground_gain * (1.0 - lag)
        A[applied, applied] = lag
        B[applied, control] = 1.0 - lag
    return A, B


def sandbox_parameter_uncertainty_envelope(
    *,
    nominal_shift: CompoundShift,
    parameter_box: ShiftParameterBox,
    state_deviation_radius: np.ndarray,
    input_deviation_radius: np.ndarray,
    dt: float = 0.1,
) -> ModelUncertaintyEnvelope:
    """Bound one-step model error over the frozen parameter box.

    For this sandbox every uncertain matrix entry is monotone in the primitive
    parameters or their positive ratios, so the componentwise extrema occur at
    parameter-box vertices.  The returned bound satisfies
    |(A_theta-A0)e + (B_theta-B0)v| <= radius for |e|<=r and |v|<=s.
    Sensor bias is intentionally excluded because it belongs in the observation
    error contract rather than the process-dynamics disturbance.
    """

    nominal_shift.validate()
    state_radius = np.asarray(state_deviation_radius, dtype=float).reshape(-1)
    input_radius = np.asarray(input_deviation_radius, dtype=float).reshape(-1)
    if state_radius.shape != (15,) or input_radius.shape != (5,):
        raise ValueError("state and input deviation radii must have shapes (15,) and (5,)")
    if not np.all(np.isfinite(state_radius)) or not np.all(np.isfinite(input_radius)):
        raise ValueError("deviation radii must be finite")
    if np.any(state_radius < 0.0) or np.any(input_radius < 0.0):
        raise ValueError("deviation radii must be non-negative")

    A0, B0 = air_ground_augmented_matrices(nominal_shift, dt=dt)
    vertices = parameter_box.vertices()
    max_A = np.zeros_like(A0)
    max_B = np.zeros_like(B0)
    for vertex in vertices:
        A, B = air_ground_augmented_matrices(vertex, dt=dt)
        max_A = np.maximum(max_A, np.abs(A - A0))
        max_B = np.maximum(max_B, np.abs(B - B0))
    radius = max_A @ state_radius + max_B @ input_radius
    hasher = sha256()
    for array in (A0, B0, max_A, max_B, state_radius, input_radius):
        canonical = np.ascontiguousarray(array, dtype="<f8")
        hasher.update(str(canonical.shape).encode("ascii"))
        hasher.update(canonical.tobytes())
    return ModelUncertaintyEnvelope(
        radius, max_A, max_B, len(vertices), hasher.hexdigest()
    )


def sandbox_axis_aligned_state_constraints() -> tuple[np.ndarray, np.ndarray]:
    """Return a sufficient state box for the sandbox's norm constraints."""

    lower = np.array(
        [-10.0, -10.0, 0.5, *([-3.0 / np.sqrt(3.0)] * 3),
         -10.0, -10.0, *([-2.5 / np.sqrt(2.0)] * 2), *([-2.0] * 5)],
        dtype=float,
    )
    upper = np.array(
        [10.0, 10.0, 5.0, *([3.0 / np.sqrt(3.0)] * 3),
         10.0, 10.0, *([2.5 / np.sqrt(2.0)] * 2), *([2.0] * 5)],
        dtype=float,
    )
    return lower, upper


def sandbox_constraint_margin_lower_bound(
    *, center: np.ndarray, radius: np.ndarray, step_index_upper: int
) -> ConstraintMargins:
    """Lower-bound all nonlinear sandbox margins over an augmented state box."""

    x0 = np.asarray(center, dtype=float).reshape(-1)
    r = np.asarray(radius, dtype=float).reshape(-1)
    if x0.shape != (15,) or r.shape != (15,):
        raise ValueError("center and radius must have shape (15,)")
    if not np.all(np.isfinite(x0)) or not np.all(np.isfinite(r)) or np.any(r < 0.0):
        raise ValueError("center and radius must be finite with non-negative radius")
    if not isinstance(step_index_upper, int) or not 0 <= step_index_upper <= 200:
        raise ValueError("step_index_upper must be an integer in [0, 200]")

    uav_box = min(
        10.0 - abs(x0[0]) - r[0], 10.0 - abs(x0[1]) - r[1],
        x0[2] - r[2] - 0.5, 5.0 - x0[2] - r[2],
    )
    uav_speed = 3.0 - float(np.linalg.norm(np.abs(x0[3:6]) + r[3:6]))
    ugv_box = min(
        10.0 - abs(x0[6]) - r[6], 10.0 - abs(x0[7]) - r[7]
    )
    ugv_speed = 2.5 - float(np.linalg.norm(np.abs(x0[8:10]) + r[8:10]))
    separation_center = x0[:2] - x0[6:8]
    separation_uncertainty = r[:2] + r[6:8]
    separation = (
        float(np.linalg.norm(separation_center))
        - float(np.linalg.norm(separation_uncertainty)) - 1.0
    )
    return ConstraintMargins(
        min(uav_box, uav_speed), min(ugv_box, ugv_speed),
        separation, 200.0 - step_index_upper,
    )
