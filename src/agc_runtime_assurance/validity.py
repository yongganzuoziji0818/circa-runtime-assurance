"""Finite-sample action-validity horizons for stale multi-agent observations.

This module calibrates a lower bound on the time remaining before the first
team-constraint violation.  It is deliberately separate from recovery-time
certificates: the returned deadline says when an action must stop executing,
not when an adapting controller is expected to recover.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math

import numpy as np
from scipy.stats import beta

from .contracts import ActionEnvelope


def first_violation_time(times: np.ndarray, margins: np.ndarray) -> float:
    """Return the first sampled time at which any team margin is negative.

    ``margins`` has shape ``(time, constraint)``.  If no violation is observed,
    the final sample time is returned as a right-censored lower bound.  Formal
    experiments must treat censored values explicitly; G0 uses this helper only
    for deterministic contract tests and synthetic counterexamples.
    """

    times = np.asarray(times, dtype=float).reshape(-1)
    margins = np.asarray(margins, dtype=float)
    if times.size < 2 or margins.ndim != 2 or margins.shape[0] != times.size:
        raise ValueError("require at least two times and a matching 2-D margin array")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(margins)):
        raise ValueError("times and margins must be finite")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")
    violated = np.any(margins < 0.0, axis=1)
    indices = np.flatnonzero(violated)
    return float(times[indices[0]] if indices.size else times[-1])


@dataclass(frozen=True)
class ActionValidityCertificate:
    """Split-conformal correction for optimistic safe-horizon predictions.

    Calibration nonconformity is ``predicted_horizon - realized_safe_horizon``.
    Under exchangeability, subtracting its finite-sample upper quantile yields a
    marginal lower bound on the new realized first-violation time.  This does
    not survive arbitrary distribution shift and is not a deterministic safety
    proof; runtime staleness and actuation budgets are subtracted separately.
    """

    alpha: float
    optimism_correction: float
    calibration_size: int
    calibration_hash: str
    calibration_delta: float | None = None

    @classmethod
    def fit(
        cls,
        predicted_horizons: np.ndarray,
        realized_safe_horizons: np.ndarray,
        *,
        alpha: float,
    ) -> "ActionValidityCertificate":
        predicted = np.asarray(predicted_horizons, dtype=float).reshape(-1)
        realized = np.asarray(realized_safe_horizons, dtype=float).reshape(-1)
        if predicted.size == 0 or predicted.shape != realized.shape:
            raise ValueError("predicted and realized horizons must be non-empty and aligned")
        if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(realized)):
            raise ValueError("horizons must be finite")
        if np.any(predicted < 0.0) or np.any(realized < 0.0):
            raise ValueError("horizons must be non-negative")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")

        scores = np.sort(predicted - realized)
        rank = math.ceil((scores.size + 1) * (1.0 - alpha))
        correction = math.inf if rank > scores.size else float(scores[rank - 1])
        payload = np.column_stack((predicted, realized))
        order = np.lexsort((payload[:, 1], payload[:, 0]))
        digest = sha256(payload[order].tobytes()).hexdigest()
        return cls(alpha, correction, int(scores.size), digest, None)

    @classmethod
    def fit_training_conditional(
        cls,
        predicted_horizons: np.ndarray,
        realized_safe_horizons: np.ndarray,
        *,
        alpha: float,
        delta: float,
    ) -> "ActionValidityCertificate":
        """Inflate the order statistic for a fixed-calibration coverage claim."""

        predicted = np.asarray(predicted_horizons, dtype=float).reshape(-1)
        realized = np.asarray(realized_safe_horizons, dtype=float).reshape(-1)
        if predicted.size == 0 or predicted.shape != realized.shape:
            raise ValueError("predicted and realized horizons must be non-empty and aligned")
        if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(realized)):
            raise ValueError("horizons must be finite")
        if np.any(predicted < 0.0) or np.any(realized < 0.0):
            raise ValueError("horizons must be non-negative")
        if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0:
            raise ValueError("alpha and delta must lie strictly between 0 and 1")

        scores = np.sort(predicted - realized)
        n = scores.size
        rank = None
        for candidate in range(1, n + 1):
            if beta.cdf(1.0 - alpha, candidate, n + 1 - candidate) <= delta:
                rank = candidate
                break
        correction = math.inf if rank is None else float(scores[rank - 1])
        payload = np.column_stack((predicted, realized))
        order = np.lexsort((payload[:, 1], payload[:, 0]))
        digest = sha256(payload[order].tobytes()).hexdigest()
        return cls(alpha, correction, int(n), digest, float(delta))

    def certified_duration(
        self,
        predicted_horizon: float,
        *,
        observation_age: float,
        compute_delay: float,
        communication_delay: float,
        actuation_delay: float,
        guard_time: float = 0.0,
    ) -> float:
        """Return the remaining executable duration after all timing debits."""

        values = np.asarray(
            [predicted_horizon, observation_age, compute_delay, communication_delay,
             actuation_delay, guard_time],
            dtype=float,
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("horizon and timing debits must be finite and non-negative")
        duration = predicted_horizon - self.optimism_correction - float(np.sum(values[1:]))
        return max(0.0, float(duration)) if math.isfinite(duration) else 0.0

    def issue(
        self,
        action: np.ndarray,
        *,
        issued_at: float,
        predicted_horizon: float,
        observation_age: float,
        compute_delay: float,
        communication_delay: float,
        actuation_delay: float,
        guard_time: float = 0.0,
        source: str = "action_validity_certificate",
    ) -> ActionEnvelope:
        duration = self.certified_duration(
            predicted_horizon,
            observation_age=observation_age,
            compute_delay=compute_delay,
            communication_delay=communication_delay,
            actuation_delay=actuation_delay,
            guard_time=guard_time,
        )
        return ActionEnvelope(
            action=np.asarray(action, dtype=float),
            issued_at=float(issued_at),
            valid_until=float(issued_at) + duration,
            source=source,
            constraint_state="validity_horizon_certified" if duration > 0 else "reject_zero_horizon",
        )

    def fingerprint(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, allow_nan=True).encode()
        return sha256(payload).hexdigest()
