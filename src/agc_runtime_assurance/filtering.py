"""Minimal-intervention affine action filter with fail-closed fallback."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

try:  # CVXPY/OSQP is the deployment backend; SciPy is an independent G0 oracle.
    import cvxpy as cp
except ImportError:  # pragma: no cover - exercised in the lightweight local stack
    cp = None


class FilterStatus(str, Enum):
    PASSTHROUGH = "passthrough"
    FILTERED = "filtered"
    INFEASIBLE_FALLBACK = "infeasible_fallback"
    SOLVER_ERROR_FALLBACK = "solver_error_fallback"


@dataclass(frozen=True)
class FilterResult:
    action: np.ndarray
    status: FilterStatus
    intervention_norm: float
    backend: str


class AffineSafetyFilter:
    """Project a nominal action onto ``A @ u <= b`` and box constraints."""

    def __init__(self, lower: np.ndarray, upper: np.ndarray):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if self.lower.shape != self.upper.shape or np.any(self.lower > self.upper):
            raise ValueError("invalid action bounds")

    def filter(
        self,
        nominal: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        fallback: np.ndarray,
    ) -> FilterResult:
        nominal = np.asarray(nominal, dtype=float)
        fallback = np.asarray(fallback, dtype=float)
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        if nominal.shape != self.lower.shape or fallback.shape != nominal.shape:
            raise ValueError("action and fallback dimensions must match bounds")
        if A.ndim != 2 or A.shape[1] != nominal.size or b.shape != (A.shape[0],):
            raise ValueError("invalid affine constraint dimensions")
        clipped = np.clip(nominal, self.lower, self.upper)
        if np.all(A @ clipped <= b + 1e-9):
            intervention = float(np.linalg.norm(clipped - nominal))
            return FilterResult(
                clipped,
                FilterStatus.PASSTHROUGH if intervention <= 1e-12 else FilterStatus.FILTERED,
                intervention, "constraint_check",
            )

        if cp is None:
            return self._filter_scipy(nominal, A, b, fallback)

        u = cp.Variable(nominal.size)
        problem = cp.Problem(
            cp.Minimize(cp.sum_squares(u - nominal)),
            [A @ u <= b, u >= self.lower, u <= self.upper],
        )
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception:
            safe = np.clip(fallback, self.lower, self.upper)
            return FilterResult(
                safe, FilterStatus.SOLVER_ERROR_FALLBACK,
                float(np.linalg.norm(safe - nominal)), "cvxpy_osqp",
            )
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or u.value is None:
            safe = np.clip(fallback, self.lower, self.upper)
            return FilterResult(
                safe, FilterStatus.INFEASIBLE_FALLBACK,
                float(np.linalg.norm(safe - nominal)), "cvxpy_osqp",
            )
        action = np.asarray(u.value, dtype=float)
        if np.any(A @ action > b + 1e-6):
            safe = np.clip(fallback, self.lower, self.upper)
            return FilterResult(
                safe, FilterStatus.SOLVER_ERROR_FALLBACK,
                float(np.linalg.norm(safe - nominal)), "cvxpy_osqp_postcheck",
            )
        return FilterResult(
            action, FilterStatus.FILTERED,
            float(np.linalg.norm(action - nominal)), "cvxpy_osqp",
        )

    def _filter_scipy(
        self,
        nominal: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        fallback: np.ndarray,
    ) -> FilterResult:
        """Independent SLSQP oracle for G0 and environments without CVXPY.

        Every solution is post-checked.  Failure or tolerance violation is
        fail-closed to the provided fallback; formal deployment remains pinned
        to CVXPY/OSQP and must compare both backends during preflight.
        """

        result = minimize(
            lambda u: float(np.dot(u - nominal, u - nominal)),
            np.clip(nominal, self.lower, self.upper),
            jac=lambda u: 2.0 * (u - nominal),
            bounds=Bounds(self.lower, self.upper),
            constraints=[LinearConstraint(A, -np.inf, b)],
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if not result.success or result.x is None:
            safe = np.clip(fallback, self.lower, self.upper)
            return FilterResult(
                safe, FilterStatus.INFEASIBLE_FALLBACK,
                float(np.linalg.norm(safe - nominal)), "scipy_slsqp",
            )
        action = np.asarray(result.x, dtype=float)
        feasible = (
            np.all(A @ action <= b + 1e-7)
            and np.all(action >= self.lower - 1e-9)
            and np.all(action <= self.upper + 1e-9)
        )
        if not feasible:
            safe = np.clip(fallback, self.lower, self.upper)
            return FilterResult(
                safe, FilterStatus.SOLVER_ERROR_FALLBACK,
                float(np.linalg.norm(safe - nominal)), "scipy_slsqp_postcheck",
            )
        return FilterResult(
            action, FilterStatus.FILTERED,
            float(np.linalg.norm(action - nominal)), "scipy_slsqp",
        )
