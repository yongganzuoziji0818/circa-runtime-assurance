"""Mechanical robust-positive-invariance check for a linear backup controller.

The verifier proves an axis-aligned box invariant for a declared linear model.
It is deliberately separate from the fallback-tube reachability check: reaching
an unverified terminal set is not sufficient runtime-assurance evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class BackupInvariantResult:
    """Auditable result of the linear-box backup-invariant check."""

    verified: bool
    reason: str
    next_radius: np.ndarray
    invariant_slack: np.ndarray
    state_slack: np.ndarray
    input_slack: np.ndarray
    equilibrium_residual: np.ndarray
    fingerprint: str


class LinearFeedbackInvariantBoxVerifier:
    """Verify a robust invariant box for a fixed linear feedback backup.

    The declared dynamics and controller are

        x+ = A x + B u + w,       u = u0 + K C (x - x0),

    with componentwise disturbance bound |w| <= disturbance_radius.  The set
    |x-x0| <= invariant_radius is robust positively invariant whenever
    |A+BKC| invariant_radius + disturbance_radius <= invariant_radius, after
    also checking equilibrium, state constraints, and input constraints.
    """

    def __init__(
        self,
        *,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        K: np.ndarray,
        equilibrium_state: np.ndarray,
        equilibrium_input: np.ndarray,
        invariant_radius: np.ndarray,
        disturbance_radius: np.ndarray,
        state_lower: np.ndarray,
        state_upper: np.ndarray,
        input_lower: np.ndarray,
        input_upper: np.ndarray,
        tolerance: float = 1e-9,
    ):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.equilibrium_state = np.asarray(equilibrium_state, dtype=float).reshape(-1)
        self.equilibrium_input = np.asarray(equilibrium_input, dtype=float).reshape(-1)
        self.invariant_radius = np.asarray(invariant_radius, dtype=float).reshape(-1)
        self.disturbance_radius = np.asarray(disturbance_radius, dtype=float).reshape(-1)
        self.state_lower = np.asarray(state_lower, dtype=float).reshape(-1)
        self.state_upper = np.asarray(state_upper, dtype=float).reshape(-1)
        self.input_lower = np.asarray(input_lower, dtype=float).reshape(-1)
        self.input_upper = np.asarray(input_upper, dtype=float).reshape(-1)
        self.tolerance = float(tolerance)
        self._validate()
        self.feedback_map = self.K @ self.C
        self.closed_loop = self.A + self.B @ self.feedback_map
        self.fingerprint = self._fingerprint()

    def _validate(self) -> None:
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be square")
        n = self.A.shape[0]
        if self.B.ndim != 2 or self.B.shape[0] != n:
            raise ValueError("B must have the same state dimension as A")
        m = self.B.shape[1]
        if self.C.ndim != 2 or self.C.shape[1] != n:
            raise ValueError("C must map states to backup measurements")
        if self.K.shape != (m, self.C.shape[0]):
            raise ValueError("K must map backup measurements to inputs")
        state_vectors = (
            self.equilibrium_state, self.invariant_radius,
            self.disturbance_radius, self.state_lower, self.state_upper,
        )
        if any(vector.size != n for vector in state_vectors):
            raise ValueError("state and invariant vectors must align with A")
        if any(vector.size != m for vector in (
            self.equilibrium_input, self.input_lower, self.input_upper,
        )):
            raise ValueError("input vectors must align with B")
        arrays = (
            self.A, self.B, self.C, self.K, *state_vectors,
            self.equilibrium_input, self.input_lower, self.input_upper,
        )
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("all model, controller, and set values must be finite")
        if np.any(self.invariant_radius <= 0.0):
            raise ValueError("invariant radii must be positive")
        if np.any(self.disturbance_radius < 0.0):
            raise ValueError("disturbance radii must be non-negative")
        if np.any(self.state_lower >= self.state_upper):
            raise ValueError("state lower bounds must be below upper bounds")
        if np.any(self.input_lower >= self.input_upper):
            raise ValueError("input lower bounds must be below upper bounds")
        if not np.isfinite(self.tolerance) or self.tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative")

    @staticmethod
    def _update_hash(hasher: "hashlib._Hash", label: str, array: np.ndarray) -> None:
        canonical = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
        hasher.update(label.encode("utf-8"))
        hasher.update(str(canonical.shape).encode("ascii"))
        hasher.update(canonical.tobytes())

    def _fingerprint(self) -> str:
        hasher = hashlib.sha256()
        for label, array in (
            ("A", self.A), ("B", self.B), ("C", self.C), ("K", self.K),
            ("equilibrium_state", self.equilibrium_state),
            ("equilibrium_input", self.equilibrium_input),
            ("invariant_radius", self.invariant_radius),
            ("disturbance_radius", self.disturbance_radius),
            ("state_lower", self.state_lower), ("state_upper", self.state_upper),
            ("input_lower", self.input_lower), ("input_upper", self.input_upper),
            ("tolerance", np.array([self.tolerance])),
        ):
            self._update_hash(hasher, label, array)
        return hasher.hexdigest()

    def verify(self) -> BackupInvariantResult:
        equilibrium_residual = (
            self.A @ self.equilibrium_state
            + self.B @ self.equilibrium_input
            - self.equilibrium_state
        )
        next_radius = (
            np.abs(self.closed_loop) @ self.invariant_radius
            + self.disturbance_radius
        )
        invariant_slack = self.invariant_radius - next_radius
        state_slack = np.minimum(
            self.equilibrium_state - self.invariant_radius - self.state_lower,
            self.state_upper - self.equilibrium_state - self.invariant_radius,
        )
        input_radius = np.abs(self.feedback_map) @ self.invariant_radius
        input_slack = np.minimum(
            self.equilibrium_input - input_radius - self.input_lower,
            self.input_upper - self.equilibrium_input - input_radius,
        )

        if np.max(np.abs(equilibrium_residual)) > self.tolerance:
            reason = "equilibrium_condition_failed"
        elif np.min(state_slack) < -self.tolerance:
            reason = "invariant_box_outside_state_constraints"
        elif np.min(input_slack) < -self.tolerance:
            reason = "backup_input_constraints_failed"
        elif np.min(invariant_slack) < -self.tolerance:
            reason = "robust_positive_invariance_failed"
        else:
            reason = "linear_box_backup_invariant_verified"
        return BackupInvariantResult(
            verified=reason == "linear_box_backup_invariant_verified",
            reason=reason,
            next_radius=next_radius,
            invariant_slack=invariant_slack,
            state_slack=state_slack,
            input_slack=input_slack,
            equilibrium_residual=equilibrium_residual,
            fingerprint=self.fingerprint,
        )
