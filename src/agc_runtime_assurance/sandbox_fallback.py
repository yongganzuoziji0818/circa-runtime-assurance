"""Invariant-bound Fallback-Safe MPC adapter for the nominal 1U1G sandbox."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backup_invariant import LinearFeedbackInvariantBoxVerifier
from .fallback_mpc import (
    FallbackMPCSolution,
    InvariantBoundFallbackResult,
    InvariantBoundFallbackTubeVerifier,
    LinearBoxFallbackSafeMPC,
    LinearBoxFallbackTubeVerifier,
)
from .sandbox_task import SandboxComparisonTask


def sandbox_backup_equilibrium() -> np.ndarray:
    return np.array(
        [-4.0, -2.0, 2.0, 0.0, 0.0, 0.0,
         4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=float,
    )


def sandbox_backup_gain() -> np.ndarray:
    gain = np.zeros((5, 15), dtype=float)
    for axis in range(3):
        gain[axis, axis] = -5.0
        gain[axis, 3 + axis] = -9.9
    for axis in range(2):
        gain[3 + axis, 6 + axis] = -5.0
        gain[3 + axis, 8 + axis] = -11.0
    return gain


def sandbox_backup_invariant_radius() -> np.ndarray:
    return np.array(
        [0.1, 0.1, 0.1, 0.05, 0.05, 0.05,
         0.1, 0.1, 0.05, 0.05, 1.0, 1.0, 1.0, 1.1, 1.1],
        dtype=float,
    )


@dataclass(frozen=True)
class SandboxFallbackDecision:
    feasible: bool
    action: np.ndarray
    solution: FallbackMPCSolution
    bound_result: InvariantBoundFallbackResult | None
    nominal_policy_fingerprint: str
    constraint_contract_fingerprint: str
    backup_invariant_fingerprint: str


class SandboxFallbackSafeMPCAdapter:
    """Solve and mechanically bind a fallback tube to one frozen backup box."""

    def __init__(self, task: SandboxComparisonTask | None = None):
        self.task = task or SandboxComparisonTask()
        self.center = sandbox_backup_equilibrium()
        self.radius = sandbox_backup_invariant_radius()
        self.gain = sandbox_backup_gain()
        identity = np.eye(15)
        disturbance = np.zeros(15)
        estimation = np.zeros(15)
        self.invariant_verifier = LinearFeedbackInvariantBoxVerifier(
            A=self.task.A_model,
            B=self.task.B_model,
            C=identity,
            K=self.gain,
            equilibrium_state=self.center,
            equilibrium_input=np.zeros(5),
            invariant_radius=self.radius,
            disturbance_radius=disturbance,
            state_lower=self.task.state_lower,
            state_upper=self.task.state_upper,
            input_lower=self.task.action_lower,
            input_upper=self.task.action_upper,
        )
        invariant = self.invariant_verifier.verify()
        if not invariant.verified:
            raise ValueError(f"sandbox backup invariant is not verified: {invariant.reason}")
        self.tube_verifier = LinearBoxFallbackTubeVerifier(
            A=self.task.A_model,
            B=self.task.B_model,
            C=identity,
            K=self.gain,
            state_lower=self.task.state_lower,
            state_upper=self.task.state_upper,
            input_lower=self.task.action_lower,
            input_upper=self.task.action_upper,
            disturbance_radius=disturbance,
            estimation_error_radius=estimation,
            recovery_lower=self.center - self.radius,
            recovery_upper=self.center + self.radius,
        )
        self.bound_verifier = InvariantBoundFallbackTubeVerifier(
            tube_verifier=self.tube_verifier,
            invariant_verifier=self.invariant_verifier,
        )
        self.solver = LinearBoxFallbackSafeMPC(self.tube_verifier)
        self.backup_invariant_fingerprint = invariant.fingerprint

    def emergency_action(self, augmented_state: np.ndarray) -> np.ndarray:
        state = np.asarray(augmented_state, dtype=float).reshape(-1)
        if state.shape != (15,) or not np.all(np.isfinite(state)):
            raise ValueError("augmented state must be a finite vector of shape (15,)")
        return np.clip(
            self.gain @ (state - self.center),
            self.task.action_lower,
            self.task.action_upper,
        )

    def decide(
        self, augmented_state: np.ndarray, *, horizon: int = 1,
    ) -> SandboxFallbackDecision:
        state = np.asarray(augmented_state, dtype=float).reshape(-1)
        if state.shape != (15,) or not np.all(np.isfinite(state)):
            raise ValueError("augmented state must be a finite vector of shape (15,)")
        if not isinstance(horizon, int) or horizon < 0:
            raise ValueError("horizon must be a non-negative integer")
        nominal = self.task.nominal_action(state)
        reference = np.tile(nominal, (horizon + 1, 1))
        emergency = self.emergency_action(state)
        solution = self.solver.solve(
            state_estimate=state,
            nominal_reference=reference,
            emergency_action=emergency,
        )
        if not solution.feasible:
            return SandboxFallbackDecision(
                False, solution.action.copy(), solution, None,
                self.task.nominal_policy_fingerprint,
                self.task.constraint_contract_fingerprint,
                self.backup_invariant_fingerprint,
            )
        bound = self.bound_verifier.verify(
            state_estimate=state,
            fallback_inputs=solution.fallback_inputs,
            nominal_first_input=solution.action,
        )
        return SandboxFallbackDecision(
            bound.feasible,
            (bound.action if bound.feasible else emergency).copy(),
            solution,
            bound,
            self.task.nominal_policy_fingerprint,
            self.task.constraint_contract_fingerprint,
            self.backup_invariant_fingerprint,
        )
