"""Transparent baseline kernels for development comparisons.

The online conformal classes below are equation-level, provisional
reimplementations.  They deliberately exclude task-specific learned-HJ and CBF
controllers, so their presence must not be reported as a complete strong
baseline reproduction.
"""

from __future__ import annotations

from bisect import insort
from dataclasses import dataclass, field
import math

import numpy as np

from .contracts import ActionEnvelope


def fixed_ttl_envelope(
    action: np.ndarray, *, issued_at: float, ttl: float, source: str = "fixed_ttl"
) -> ActionEnvelope:
    if not np.isfinite(ttl) or ttl < 0.0:
        raise ValueError("ttl must be finite and non-negative")
    return ActionEnvelope(
        np.asarray(action, dtype=float), float(issued_at), float(issued_at) + float(ttl),
        source, "fixed_ttl_no_statistical_certificate",
    )


def point_horizon_envelope(
    action: np.ndarray,
    *,
    issued_at: float,
    predicted_horizon: float,
    observation_age: float,
    compute_delay: float,
    communication_delay: float,
    actuation_delay: float,
    guard_time: float = 0.0,
    source: str = "point_horizon",
) -> ActionEnvelope:
    values = np.asarray(
        [predicted_horizon, observation_age, compute_delay, communication_delay,
         actuation_delay, guard_time], dtype=float,
    )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("horizon and timing debits must be finite and non-negative")
    duration = max(0.0, float(values[0] - np.sum(values[1:])))
    return ActionEnvelope(
        np.asarray(action, dtype=float), float(issued_at), float(issued_at) + duration,
        source, "point_horizon_without_calibration" if duration > 0 else "reject_zero_horizon",
    )


def acofi_target(local_margin: float, next_value: float, *, gamma: float) -> float:
    """Return ACoFi's one-step target R_t from arXiv:2604.18482, Eq. (2).

    The helper is intentionally independent of the learned world/HJ models so
    the equation can be unit-tested before those components are reproduced.
    """

    values = np.asarray([local_margin, next_value, gamma], dtype=float)
    if not np.all(np.isfinite(values)) or not 0.0 <= gamma <= 1.0:
        raise ValueError("margins must be finite and gamma must lie in [0, 1]")
    return float((1.0 - gamma) * local_margin + gamma * min(local_margin, next_value))


@dataclass
class ACoFiUpdateKernel:
    """Provisional equation-level implementation of ACoFi Algorithm 1.

    It implements the one-sided score, delayed error feedback, effective-alpha
    update, finite-history quantile, and task/safe switching test.  It does not
    implement or train the paper's world model, HJ value/Q functions, or safe
    controller.
    """

    target_alpha: float
    learning_rate: float
    initial_alpha: float | None = None
    effective_alpha: float = field(init=False)
    quantile: float = field(init=False, default=0.0)
    scores: list[float] = field(init=False, default_factory=list)
    error_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        initial = self.target_alpha if self.initial_alpha is None else self.initial_alpha
        values = np.asarray([self.target_alpha, self.learning_rate, initial], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("ACoFi parameters must be finite")
        if not 0.0 <= self.target_alpha <= 1.0:
            raise ValueError("target_alpha must lie in [0, 1]")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        self.effective_alpha = float(initial)

    @staticmethod
    def one_sided_score(predicted_q: float, realized_target: float) -> float:
        values = np.asarray([predicted_q, realized_target], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("predicted and realized values must be finite")
        return max(float(predicted_q - realized_target), 0.0)

    @staticmethod
    def _paper_quantile(sorted_scores: list[float], probability: float) -> float:
        """Algorithm-1 quantile with the paper's 0/+inf boundary semantics."""

        if not math.isfinite(probability):
            raise ValueError("probability must be finite")
        n = len(sorted_scores)
        if probability < 0.0:
            return 0.0
        if probability > n / (n + 1.0):
            return math.inf
        rank = math.ceil(probability * (n + 1.0))
        if rank <= 0:
            return 0.0
        return float(sorted_scores[rank - 1])

    def observe(self, *, predicted_q: float, realized_target: float) -> bool:
        """Consume delayed ground truth and update q for the next decision."""

        score = self.one_sided_score(predicted_q, realized_target)
        error = score > self.quantile
        self.error_count += int(error)
        self.effective_alpha += self.learning_rate * (
            self.target_alpha - float(error)
        )
        insort(self.scores, score)
        self.quantile = self._paper_quantile(
            self.scores, 1.0 - self.effective_alpha
        )
        return error

    def allow_task_action(
        self,
        *,
        predicted_task_q: float,
        local_margin: float,
        gamma: float,
        safety_threshold: float,
    ) -> bool:
        """Apply the ACoFi task/safe-policy switching inequality."""

        values = np.asarray(
            [predicted_task_q, local_margin, gamma, safety_threshold], dtype=float
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("switching inputs must be finite")
        if not 0.0 <= gamma <= 1.0 or safety_threshold < 0.0:
            raise ValueError("gamma must lie in [0, 1] and threshold be non-negative")
        threshold = (
            self.quantile
            + gamma * safety_threshold
            + (1.0 - gamma) * local_margin
        )
        return bool(predicted_task_q >= threshold)


@dataclass
class ConformalDecisionVariable:
    """CDT conformal-variable update from arXiv:2409.18862, Theorem 1.

    This is only the online adaptation kernel.  A complete baseline must also
    add the variable to the paper's pairwise CBF constraint and reproduce its
    loss definition and controller assumptions.
    """

    target_loss: float
    learning_rate: float
    value: float
    loss_lower: float = 0.0
    loss_upper: float = 1.0
    steps: int = 0
    cumulative_loss: float = 0.0

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.target_loss, self.learning_rate, self.value,
             self.loss_lower, self.loss_upper], dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("CDT parameters must be finite")
        if self.loss_lower >= self.loss_upper:
            raise ValueError("loss_lower must be below loss_upper")
        if not self.loss_lower <= self.target_loss <= self.loss_upper:
            raise ValueError("target_loss must lie within the declared loss range")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")

    def update(self, loss: float) -> float:
        if not math.isfinite(loss) or not self.loss_lower <= loss <= self.loss_upper:
            raise ValueError("loss must be finite and lie within the declared range")
        self.value += self.learning_rate * (self.target_loss - float(loss))
        self.steps += 1
        self.cumulative_loss += float(loss)
        return self.value

    @property
    def average_loss(self) -> float:
        return self.cumulative_loss / self.steps if self.steps else math.nan
