"""Runtime adapter for the ACoFi baseline (Huriot et al., L4DC 2026).

The adapter implements Algorithm 1's delayed target update and task/safe policy
switch.  Learned world, Q/HJ value, task policy, and safe policy are injected
as values/actions so the same frozen models can be shared across comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .baselines import ACoFiUpdateKernel, acofi_target


@dataclass(frozen=True)
class ACoFiDecision:
    action: np.ndarray
    source: str
    predicted_task_q: float
    switching_threshold: float
    quantile: float
    effective_alpha: float


class ACoFiRuntimeAdapter:
    """Execute ACoFi switching with externally supplied learned components."""

    def __init__(
        self,
        *,
        target_alpha: float,
        learning_rate: float,
        gamma: float,
        safety_threshold: float,
        initial_alpha: float | None = None,
    ):
        if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")
        if not np.isfinite(safety_threshold) or safety_threshold < 0.0:
            raise ValueError("safety threshold must be finite and non-negative")
        self.gamma = float(gamma)
        self.safety_threshold = float(safety_threshold)
        self.kernel = ACoFiUpdateKernel(
            target_alpha=target_alpha,
            learning_rate=learning_rate,
            initial_alpha=initial_alpha,
        )

    def observe_transition(
        self,
        *,
        previous_predicted_q: float,
        previous_local_margin: float,
        next_learned_value: float,
    ) -> tuple[float, bool]:
        """Process delayed ground truth after the previous action executes."""

        target = acofi_target(
            previous_local_margin, next_learned_value, gamma=self.gamma
        )
        error = self.kernel.observe(
            predicted_q=previous_predicted_q, realized_target=target
        )
        return target, error

    def decide(
        self,
        *,
        predicted_task_q: float,
        current_local_margin: float,
        task_action: np.ndarray,
        safe_action: np.ndarray,
    ) -> ACoFiDecision:
        task = np.asarray(task_action, dtype=float)
        safe = np.asarray(safe_action, dtype=float)
        if task.ndim != 1 or task.shape != safe.shape or task.size == 0:
            raise ValueError("task and safe actions must be aligned one-dimensional vectors")
        if not np.all(np.isfinite(task)) or not np.all(np.isfinite(safe)):
            raise ValueError("task and safe actions must be finite")
        if not np.isfinite(predicted_task_q) or not np.isfinite(current_local_margin):
            raise ValueError("Q and local margin must be finite")

        threshold = (
            self.kernel.quantile
            + self.gamma * self.safety_threshold
            + (1.0 - self.gamma) * float(current_local_margin)
        )
        allow_task = self.kernel.allow_task_action(
            predicted_task_q=predicted_task_q,
            local_margin=current_local_margin,
            gamma=self.gamma,
            safety_threshold=self.safety_threshold,
        )
        return ACoFiDecision(
            action=(task if allow_task else safe).copy(),
            source="task_policy" if allow_task else "learned_safe_policy",
            predicted_task_q=float(predicted_task_q),
            switching_threshold=float(threshold),
            quantile=float(self.kernel.quantile),
            effective_alpha=float(self.kernel.effective_alpha),
        )

    @property
    def empirical_error_rate(self) -> float:
        if not self.kernel.scores:
            return math.nan
        return self.kernel.error_count / len(self.kernel.scores)
