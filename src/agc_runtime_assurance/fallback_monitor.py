"""Conformal stopping-time runtime monitor from Fallback-Safe MPC Algorithm 2."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class FallbackMonitorDecision:
    alarm: bool
    randomized_rank: float
    greater_count: int
    equal_count: int
    tie_draw: int


class ConformalStoppingTimeMonitor:
    """Randomized conformal alarm calibrated on first-fault stopping states.

    ``fault_anomaly_scores`` must contain only anomaly scores from calibration
    trajectories whose stopping state is a perception fault.  Time-limit stops
    without a fault are excluded exactly as in Algorithm 2.  The guarantee does
    not remove the total-variation shift term in the paper's Eq. (10).
    """

    def __init__(
        self,
        fault_anomaly_scores: np.ndarray,
        *,
        risk_tolerance: float,
        random_seed: int,
    ):
        scores = np.asarray(fault_anomaly_scores, dtype=float).reshape(-1)
        if not np.all(np.isfinite(scores)):
            raise ValueError("fault anomaly scores must be finite")
        if not np.isfinite(risk_tolerance) or not 0.0 < risk_tolerance <= 1.0:
            raise ValueError("risk_tolerance must lie in (0, 1]")
        if not isinstance(random_seed, int):
            raise ValueError("random_seed must be an integer")
        self.scores = scores.copy()
        self.risk_tolerance = float(risk_tolerance)
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

    @property
    def minimum_nontrivial_fault_samples(self) -> int:
        return max(0, math.ceil(1.0 / self.risk_tolerance) - 1)

    @property
    def nontrivial_sample_gate(self) -> bool:
        return self.scores.size >= self.minimum_nontrivial_fault_samples

    def evaluate(self, anomaly_score: float) -> FallbackMonitorDecision:
        if not np.isfinite(anomaly_score):
            raise ValueError("test anomaly score must be finite")
        greater = int(np.count_nonzero(self.scores > anomaly_score))
        equal = int(np.count_nonzero(self.scores == anomaly_score))
        tie_draw = int(self._rng.integers(0, equal + 1))
        q = (greater + tie_draw + 1.0) / (self.scores.size + 1.0)
        alarm = q <= 1.0 - self.risk_tolerance
        return FallbackMonitorDecision(alarm, float(q), greater, equal, tie_draw)
