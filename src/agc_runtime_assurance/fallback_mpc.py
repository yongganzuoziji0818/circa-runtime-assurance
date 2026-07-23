"""Equation-level feasibility kernel for linear Fallback-Safe MPC.

This module implements the box-set specialization of the tube tightening in
Sinha, Schmerling, and Pavone (CDC 2023), Eq. (14).  It verifies a supplied
fallback input sequence; it does not solve the paper's online optimal-control
problem and therefore remains a provisional baseline component.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize

from .backup_invariant import LinearFeedbackInvariantBoxVerifier


@dataclass(frozen=True)
class FallbackTubeResult:
    feasible: bool
    reason: str
    minimum_slack: float
    nominal_states: np.ndarray
    error_radii: np.ndarray


@dataclass(frozen=True)
class FallbackMPCSolution:
    feasible: bool
    action: np.ndarray
    fallback_inputs: np.ndarray
    objective: float
    solver_status: str
    tube_result: FallbackTubeResult | None


@dataclass(frozen=True)
class InvariantBoundFallbackResult:
    """Fallback-tube evidence bound to a matching verified invariant set."""

    feasible: bool
    reason: str
    action: np.ndarray
    fallback_inputs: np.ndarray
    tube_result: FallbackTubeResult | None
    backup_invariant_fingerprint: str


class LinearBoxFallbackTubeVerifier:
    """Verify Eq. (14) for axis-aligned state, input, W, E, and recovery sets."""

    def __init__(
        self,
        *,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        K: np.ndarray,
        state_lower: np.ndarray,
        state_upper: np.ndarray,
        input_lower: np.ndarray,
        input_upper: np.ndarray,
        disturbance_radius: np.ndarray,
        estimation_error_radius: np.ndarray,
        recovery_lower: np.ndarray,
        recovery_upper: np.ndarray,
        equality_tolerance: float = 1e-9,
    ):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.state_lower = np.asarray(state_lower, dtype=float).reshape(-1)
        self.state_upper = np.asarray(state_upper, dtype=float).reshape(-1)
        self.input_lower = np.asarray(input_lower, dtype=float).reshape(-1)
        self.input_upper = np.asarray(input_upper, dtype=float).reshape(-1)
        self.disturbance_radius = np.asarray(disturbance_radius, dtype=float).reshape(-1)
        self.estimation_error_radius = np.asarray(
            estimation_error_radius, dtype=float
        ).reshape(-1)
        self.recovery_lower = np.asarray(recovery_lower, dtype=float).reshape(-1)
        self.recovery_upper = np.asarray(recovery_upper, dtype=float).reshape(-1)
        self.equality_tolerance = float(equality_tolerance)
        self._validate_model()
        self.closed_loop = self.A + self.B @ self.K @ self.C
        self.input_error_map = self.K @ self.C

    def _validate_model(self) -> None:
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be square")
        n = self.A.shape[0]
        if self.B.ndim != 2 or self.B.shape[0] != n:
            raise ValueError("B must have the same state dimension as A")
        m = self.B.shape[1]
        if self.C.ndim != 2 or self.C.shape[1] != n:
            raise ValueError("C must map states to fallback measurements")
        if self.K.shape != (m, self.C.shape[0]):
            raise ValueError("K must map fallback measurements to inputs")
        state_vectors = (
            self.state_lower, self.state_upper, self.disturbance_radius,
            self.estimation_error_radius, self.recovery_lower, self.recovery_upper,
        )
        if any(vector.size != n for vector in state_vectors):
            raise ValueError("state, disturbance, error, and recovery vectors must align")
        if self.input_lower.size != m or self.input_upper.size != m:
            raise ValueError("input bounds must align with B")
        all_values = np.concatenate((*state_vectors, self.input_lower, self.input_upper))
        if not np.all(np.isfinite(all_values)):
            raise ValueError("all model and set values must be finite")
        if not all(np.all(np.isfinite(matrix)) for matrix in (self.A, self.B, self.C, self.K)):
            raise ValueError("system matrices must be finite")
        if np.any(self.state_lower >= self.state_upper):
            raise ValueError("state lower bounds must be below upper bounds")
        if np.any(self.input_lower >= self.input_upper):
            raise ValueError("input lower bounds must be below upper bounds")
        if np.any(self.recovery_lower >= self.recovery_upper):
            raise ValueError("recovery lower bounds must be below upper bounds")
        if np.any(self.disturbance_radius < 0.0) or np.any(self.estimation_error_radius < 0.0):
            raise ValueError("box radii must be non-negative")
        if not np.isfinite(self.equality_tolerance) or self.equality_tolerance < 0.0:
            raise ValueError("equality tolerance must be finite and non-negative")

    def error_radii(self, horizon: int) -> np.ndarray:
        """Return F_0,...,F_{T+1} box radii from Eq. (14)."""

        if not isinstance(horizon, int) or horizon < 0:
            raise ValueError("horizon must be a non-negative integer")
        radii = [np.zeros(self.A.shape[0], dtype=float)]
        absolute_closed_loop = np.abs(self.closed_loop)
        additive = (
            absolute_closed_loop @ self.estimation_error_radius
            + self.disturbance_radius
            + self.estimation_error_radius
        )
        for _ in range(horizon + 1):
            radii.append(absolute_closed_loop @ radii[-1] + additive)
        return np.vstack(radii)

    @staticmethod
    def _box_slack(
        value: np.ndarray, lower: np.ndarray, upper: np.ndarray, radius: np.ndarray
    ) -> float:
        tightened_lower = lower + radius
        tightened_upper = upper - radius
        return float(np.min(np.minimum(value - tightened_lower, tightened_upper - value)))

    def verify(
        self,
        *,
        state_estimate: np.ndarray,
        fallback_inputs: np.ndarray,
        nominal_first_input: np.ndarray,
    ) -> FallbackTubeResult:
        x0 = np.asarray(state_estimate, dtype=float).reshape(-1)
        inputs = np.asarray(fallback_inputs, dtype=float)
        nominal_first = np.asarray(nominal_first_input, dtype=float).reshape(-1)
        n, m = self.A.shape[0], self.B.shape[1]
        if x0.size != n or inputs.ndim != 2 or inputs.shape[1] != m or inputs.shape[0] < 1:
            raise ValueError("state and fallback input sequence have incompatible dimensions")
        if nominal_first.size != m:
            raise ValueError("nominal first input has incompatible dimension")
        if not np.all(np.isfinite(x0)) or not np.all(np.isfinite(inputs)) or not np.all(
            np.isfinite(nominal_first)
        ):
            raise ValueError("verification inputs must be finite")

        horizon = inputs.shape[0] - 1
        radii = self.error_radii(horizon)
        states = [x0]
        for control in inputs:
            states.append(self.A @ states[-1] + self.B @ control)
        nominal_states = np.vstack(states)

        if not np.allclose(
            nominal_first, inputs[0], atol=self.equality_tolerance, rtol=0.0
        ):
            return FallbackTubeResult(
                False, "nominal_fallback_first_input_mismatch", -np.inf,
                nominal_states, radii,
            )

        slacks: list[tuple[float, str]] = []
        for step in range(horizon + 1):
            state_radius = radii[step] + self.estimation_error_radius
            slacks.append((
                self._box_slack(
                    nominal_states[step], self.state_lower, self.state_upper, state_radius
                ),
                f"state_tightening_failed_at_step_{step}",
            ))
            input_radius = np.abs(self.input_error_map) @ state_radius
            slacks.append((
                self._box_slack(
                    inputs[step], self.input_lower, self.input_upper, input_radius
                ),
                f"input_tightening_failed_at_step_{step}",
            ))

        terminal_radius = radii[horizon + 1] + self.estimation_error_radius
        slacks.append((
            self._box_slack(
                nominal_states[horizon + 1], self.recovery_lower,
                self.recovery_upper, terminal_radius,
            ),
            "terminal_recovery_set_failed",
        ))
        minimum_slack, reason = min(slacks, key=lambda item: item[0])
        if minimum_slack < -self.equality_tolerance:
            return FallbackTubeResult(False, reason, minimum_slack, nominal_states, radii)
        return FallbackTubeResult(True, "fallback_tube_feasible", minimum_slack, nominal_states, radii)


class InvariantBoundFallbackTubeVerifier:
    """Bind a feasible fallback tube to the same verified linear backup contract.

    This wrapper checks exact equality of the frozen dynamics, controller,
    constraints, and disturbance box, and requires the tube recovery box to be
    exactly the verified invariant box.  It prevents a terminal-reachability
    result for one box from being presented as evidence for a different backup.
    """

    def __init__(
        self,
        *,
        tube_verifier: LinearBoxFallbackTubeVerifier,
        invariant_verifier: LinearFeedbackInvariantBoxVerifier,
    ):
        if not isinstance(tube_verifier, LinearBoxFallbackTubeVerifier):
            raise TypeError("a LinearBoxFallbackTubeVerifier is required")
        if not isinstance(invariant_verifier, LinearFeedbackInvariantBoxVerifier):
            raise TypeError("a LinearFeedbackInvariantBoxVerifier is required")
        self.tube_verifier = tube_verifier
        self.invariant_verifier = invariant_verifier

    @staticmethod
    def _same(left: np.ndarray, right: np.ndarray) -> bool:
        return bool(np.array_equal(np.asarray(left), np.asarray(right)))

    def _contract_mismatch(self) -> str | None:
        pairs = (
            ("A", self.tube_verifier.A, self.invariant_verifier.A),
            ("B", self.tube_verifier.B, self.invariant_verifier.B),
            ("C", self.tube_verifier.C, self.invariant_verifier.C),
            ("K", self.tube_verifier.K, self.invariant_verifier.K),
            ("state_lower", self.tube_verifier.state_lower, self.invariant_verifier.state_lower),
            ("state_upper", self.tube_verifier.state_upper, self.invariant_verifier.state_upper),
            ("input_lower", self.tube_verifier.input_lower, self.invariant_verifier.input_lower),
            ("input_upper", self.tube_verifier.input_upper, self.invariant_verifier.input_upper),
            (
                "disturbance_radius", self.tube_verifier.disturbance_radius,
                self.invariant_verifier.disturbance_radius,
            ),
            (
                "recovery_lower", self.tube_verifier.recovery_lower,
                self.invariant_verifier.equilibrium_state
                - self.invariant_verifier.invariant_radius,
            ),
            (
                "recovery_upper", self.tube_verifier.recovery_upper,
                self.invariant_verifier.equilibrium_state
                + self.invariant_verifier.invariant_radius,
            ),
        )
        for label, left, right in pairs:
            if not self._same(left, right):
                return label
        return None

    def verify(
        self,
        *,
        state_estimate: np.ndarray,
        fallback_inputs: np.ndarray,
        nominal_first_input: np.ndarray,
    ) -> InvariantBoundFallbackResult:
        invariant = self.invariant_verifier.verify()
        inputs = np.asarray(fallback_inputs, dtype=float)
        empty_action = np.empty(0, dtype=float)
        if not invariant.verified:
            return InvariantBoundFallbackResult(
                False, f"backup_invariant_not_verified:{invariant.reason}",
                empty_action, inputs.copy(), None, invariant.fingerprint,
            )
        mismatch = self._contract_mismatch()
        if mismatch is not None:
            return InvariantBoundFallbackResult(
                False, f"backup_invariant_contract_mismatch:{mismatch}",
                empty_action, inputs.copy(), None, invariant.fingerprint,
            )
        tube = self.tube_verifier.verify(
            state_estimate=state_estimate,
            fallback_inputs=inputs,
            nominal_first_input=nominal_first_input,
        )
        if not tube.feasible:
            return InvariantBoundFallbackResult(
                False, f"fallback_tube_not_verified:{tube.reason}",
                empty_action, inputs.copy(), tube, invariant.fingerprint,
            )
        return InvariantBoundFallbackResult(
            True, "invariant_bound_fallback_tube_feasible", inputs[0].copy(),
            inputs.copy(), tube, invariant.fingerprint,
        )


class LinearBoxFallbackSafeMPC:
    """SLSQP reference solver for a minimal-intervention Eq. (14) instance.

    The optimized sequence is the fallback tube plan.  Its first input is also
    the action issued to the nominal system, enforcing the paper's shared-first-
    input condition.  The cost keeps the sequence close to a frozen nominal
    reference.  A deployment implementation must reproduce this with a convex
    QP backend and compare post-checked solutions.
    """

    def __init__(self, verifier: LinearBoxFallbackTubeVerifier):
        self.verifier = verifier

    def _slack_vector(
        self, state_estimate: np.ndarray, flat_inputs: np.ndarray, horizon: int
    ) -> np.ndarray:
        m = self.verifier.B.shape[1]
        inputs = np.asarray(flat_inputs, dtype=float).reshape(horizon + 1, m)
        radii = self.verifier.error_radii(horizon)
        states = [state_estimate]
        for control in inputs:
            states.append(self.verifier.A @ states[-1] + self.verifier.B @ control)

        slacks: list[np.ndarray] = []
        for step in range(horizon + 1):
            state_radius = radii[step] + self.verifier.estimation_error_radius
            state_lower = self.verifier.state_lower + state_radius
            state_upper = self.verifier.state_upper - state_radius
            slacks.extend((states[step] - state_lower, state_upper - states[step]))

            input_radius = np.abs(self.verifier.input_error_map) @ state_radius
            input_lower = self.verifier.input_lower + input_radius
            input_upper = self.verifier.input_upper - input_radius
            slacks.extend((inputs[step] - input_lower, input_upper - inputs[step]))

        terminal_radius = radii[horizon + 1] + self.verifier.estimation_error_radius
        recovery_lower = self.verifier.recovery_lower + terminal_radius
        recovery_upper = self.verifier.recovery_upper - terminal_radius
        slacks.extend((states[-1] - recovery_lower, recovery_upper - states[-1]))
        return np.concatenate([np.asarray(slack, dtype=float).reshape(-1) for slack in slacks])

    def solve(
        self,
        *,
        state_estimate: np.ndarray,
        nominal_reference: np.ndarray,
        emergency_action: np.ndarray,
    ) -> FallbackMPCSolution:
        state = np.asarray(state_estimate, dtype=float).reshape(-1)
        reference = np.asarray(nominal_reference, dtype=float)
        emergency = np.asarray(emergency_action, dtype=float).reshape(-1)
        n, m = self.verifier.A.shape[0], self.verifier.B.shape[1]
        if state.size != n or reference.ndim != 2 or reference.shape[1] != m:
            raise ValueError("state and nominal reference have incompatible dimensions")
        if reference.shape[0] < 1 or emergency.size != m:
            raise ValueError("reference must be non-empty and emergency action must align")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(reference)) or not np.all(
            np.isfinite(emergency)
        ):
            raise ValueError("MPC inputs must be finite")

        horizon = reference.shape[0] - 1
        tiled_lower = np.tile(self.verifier.input_lower, horizon + 1)
        tiled_upper = np.tile(self.verifier.input_upper, horizon + 1)
        initial = np.clip(reference, self.verifier.input_lower, self.verifier.input_upper).reshape(-1)
        flat_reference = reference.reshape(-1)
        result = minimize(
            lambda value: float(np.dot(value - flat_reference, value - flat_reference)),
            initial,
            jac=lambda value: 2.0 * (value - flat_reference),
            bounds=Bounds(tiled_lower, tiled_upper),
            constraints=[NonlinearConstraint(
                lambda value: self._slack_vector(state, value, horizon),
                0.0, np.inf,
            )],
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not result.success or result.x is None:
            safe = np.clip(emergency, self.verifier.input_lower, self.verifier.input_upper)
            return FallbackMPCSolution(
                False, safe, np.empty((0, m)), math.inf,
                f"scipy_slsqp_failed:{result.message}", None,
            )

        sequence = np.asarray(result.x, dtype=float).reshape(horizon + 1, m)
        tube = self.verifier.verify(
            state_estimate=state,
            fallback_inputs=sequence,
            nominal_first_input=sequence[0],
        )
        if not tube.feasible:
            safe = np.clip(emergency, self.verifier.input_lower, self.verifier.input_upper)
            return FallbackMPCSolution(
                False, safe, sequence, float(result.fun),
                f"scipy_slsqp_postcheck:{tube.reason}", tube,
            )
        return FallbackMPCSolution(
            True, sequence[0].copy(), sequence.copy(), float(result.fun),
            "scipy_slsqp", tube,
        )


class LinearBoxFallbackSafeMPCQP:
    """Convex CVXPY/OSQP implementation of the same box-tube problem.

    This backend is intentionally separate from the SLSQP reference solver so
    task-level evidence can record which optimizer produced an action and can
    compare both implementations on identical instances.  Solver output is
    never trusted directly: every candidate sequence is rechecked by
    :class:`LinearBoxFallbackTubeVerifier` before it is marked feasible.
    """

    def __init__(
        self,
        verifier: LinearBoxFallbackTubeVerifier,
        *,
        eps_abs: float = 1e-8,
        eps_rel: float = 1e-8,
        max_iter: int = 100_000,
    ):
        if not np.isfinite(eps_abs) or eps_abs <= 0.0:
            raise ValueError("eps_abs must be finite and positive")
        if not np.isfinite(eps_rel) or eps_rel <= 0.0:
            raise ValueError("eps_rel must be finite and positive")
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        self.verifier = verifier
        self.eps_abs = float(eps_abs)
        self.eps_rel = float(eps_rel)
        self.max_iter = max_iter

    def solve(
        self,
        *,
        state_estimate: np.ndarray,
        nominal_reference: np.ndarray,
        emergency_action: np.ndarray,
    ) -> FallbackMPCSolution:
        try:
            import cvxpy as cp
        except ImportError as exc:  # pragma: no cover - environment gate
            raise RuntimeError(
                "LinearBoxFallbackSafeMPCQP requires cvxpy with OSQP support"
            ) from exc

        state = np.asarray(state_estimate, dtype=float).reshape(-1)
        reference = np.asarray(nominal_reference, dtype=float)
        emergency = np.asarray(emergency_action, dtype=float).reshape(-1)
        n, m = self.verifier.A.shape[0], self.verifier.B.shape[1]
        if state.size != n or reference.ndim != 2 or reference.shape[1] != m:
            raise ValueError("state and nominal reference have incompatible dimensions")
        if reference.shape[0] < 1 or emergency.size != m:
            raise ValueError("reference must be non-empty and emergency action must align")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(reference)) or not np.all(
            np.isfinite(emergency)
        ):
            raise ValueError("MPC inputs must be finite")

        horizon = reference.shape[0] - 1
        radii = self.verifier.error_radii(horizon)
        controls = cp.Variable((horizon + 1, m))
        states: list[object] = [cp.Constant(state)]
        constraints: list[object] = []
        for step in range(horizon + 1):
            state_radius = radii[step] + self.verifier.estimation_error_radius
            constraints.extend((
                states[step] >= self.verifier.state_lower + state_radius,
                states[step] <= self.verifier.state_upper - state_radius,
            ))
            input_radius = np.abs(self.verifier.input_error_map) @ state_radius
            constraints.extend((
                controls[step] >= self.verifier.input_lower + input_radius,
                controls[step] <= self.verifier.input_upper - input_radius,
            ))
            states.append(self.verifier.A @ states[step] + self.verifier.B @ controls[step])

        terminal_radius = radii[horizon + 1] + self.verifier.estimation_error_radius
        constraints.extend((
            states[-1] >= self.verifier.recovery_lower + terminal_radius,
            states[-1] <= self.verifier.recovery_upper - terminal_radius,
        ))
        problem = cp.Problem(cp.Minimize(cp.sum_squares(controls - reference)), constraints)
        try:
            problem.solve(
                solver=cp.OSQP,
                eps_abs=self.eps_abs,
                eps_rel=self.eps_rel,
                max_iter=self.max_iter,
                polishing=True,
                warm_start=False,
                verbose=False,
            )
        except cp.error.SolverError as exc:
            safe = np.clip(emergency, self.verifier.input_lower, self.verifier.input_upper)
            return FallbackMPCSolution(
                False, safe, np.empty((0, m)), math.inf,
                f"cvxpy_osqp_error:{type(exc).__name__}", None,
            )

        if problem.status != cp.OPTIMAL or controls.value is None:
            safe = np.clip(emergency, self.verifier.input_lower, self.verifier.input_upper)
            return FallbackMPCSolution(
                False, safe, np.empty((0, m)), math.inf,
                f"cvxpy_osqp_status:{problem.status}", None,
            )

        sequence = np.asarray(controls.value, dtype=float).reshape(horizon + 1, m)
        tube = self.verifier.verify(
            state_estimate=state,
            fallback_inputs=sequence,
            nominal_first_input=sequence[0],
        )
        if not tube.feasible:
            safe = np.clip(emergency, self.verifier.input_lower, self.verifier.input_upper)
            return FallbackMPCSolution(
                False, safe, sequence, float(problem.value),
                f"cvxpy_osqp_postcheck:{tube.reason}", tube,
            )
        return FallbackMPCSolution(
            True, sequence[0].copy(), sequence.copy(), float(problem.value),
            "cvxpy_osqp", tube,
        )
