"""Auditable fail-closed runtime assurance state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AssuranceMode(str, Enum):
    NOMINAL = "nominal"
    FILTERED = "filtered"
    DEGRADED = "degraded"
    BACKUP = "backup"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class Transition:
    previous: AssuranceMode
    current: AssuranceMode
    reason: str


class RuntimeAssuranceStateMachine:
    def __init__(self, recovery_hold_steps: int = 5):
        if recovery_hold_steps < 1:
            raise ValueError("recovery_hold_steps must be positive")
        self.mode = AssuranceMode.NOMINAL
        self.recovery_hold_steps = recovery_hold_steps
        self._stable_steps = 0

    def step(
        self,
        *,
        certificate_valid: bool,
        nominal_safe: bool,
        filter_feasible: bool,
        backup_recoverable: bool,
        observations_fresh: bool = True,
    ) -> Transition:
        previous = self.mode
        if not observations_fresh:
            self.mode = AssuranceMode.BACKUP
            self._stable_steps = 0
            return Transition(previous, self.mode, "stale_observation_fail_closed")
        if not certificate_valid:
            self.mode = AssuranceMode.BACKUP
            self._stable_steps = 0
            return Transition(previous, self.mode, "certificate_invalid")
        if not backup_recoverable:
            self.mode = AssuranceMode.BACKUP
            self._stable_steps = 0
            return Transition(previous, self.mode, "recoverability_gate_triggered")

        if self.mode in {AssuranceMode.BACKUP, AssuranceMode.DEGRADED, AssuranceMode.RECOVERY}:
            if nominal_safe and filter_feasible:
                self._stable_steps += 1
                self.mode = AssuranceMode.RECOVERY
                if self._stable_steps >= self.recovery_hold_steps:
                    self.mode = AssuranceMode.NOMINAL
                    self._stable_steps = 0
                    return Transition(previous, self.mode, "recovery_hysteresis_satisfied")
                return Transition(previous, self.mode, "recovery_hold")
            self._stable_steps = 0

        if nominal_safe:
            self.mode = AssuranceMode.NOMINAL
            return Transition(previous, self.mode, "nominal_certified")
        if filter_feasible:
            self.mode = AssuranceMode.FILTERED
            return Transition(previous, self.mode, "minimal_intervention")
        self.mode = AssuranceMode.DEGRADED
        return Transition(previous, self.mode, "filter_infeasible_use_degraded_control")
