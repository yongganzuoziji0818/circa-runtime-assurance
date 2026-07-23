"""Weighted conformal action-horizon baseline under known covariate shift."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np


@dataclass(frozen=True)
class WeightedActionValidityCertificate:
    """Test-weight-dependent lower horizon correction.

    Calibration samples have likelihood-ratio weights.  For a new context, the
    weighted conformal distribution also places the test sample's normalized
    weight at positive infinity.  If the finite calibration mass cannot reach
    the requested quantile, the correction is infinite and execution fails
    closed.  Validity requires the weighted-exchangeability/covariate-shift
    assumptions and exact weights; estimated or arbitrary drift is not covered.
    """

    alpha: float
    ordered_scores: np.ndarray
    ordered_weights: np.ndarray
    calibration_hash: str

    @classmethod
    def fit(
        cls,
        predicted_horizons: np.ndarray,
        realized_horizons: np.ndarray,
        calibration_weights: np.ndarray,
        *,
        alpha: float,
    ) -> "WeightedActionValidityCertificate":
        predicted = np.asarray(predicted_horizons, dtype=float).reshape(-1)
        realized = np.asarray(realized_horizons, dtype=float).reshape(-1)
        weights = np.asarray(calibration_weights, dtype=float).reshape(-1)
        if (
            predicted.size == 0
            or predicted.shape != realized.shape
            or predicted.shape != weights.shape
        ):
            raise ValueError("horizons and weights must be non-empty and aligned")
        if (
            not np.all(np.isfinite(predicted))
            or not np.all(np.isfinite(realized))
            or not np.all(np.isfinite(weights))
        ):
            raise ValueError("horizons and weights must be finite")
        if np.any(predicted < 0.0) or np.any(realized < 0.0):
            raise ValueError("horizons must be non-negative")
        if np.any(weights <= 0.0):
            raise ValueError("calibration weights must be strictly positive")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")

        scores = predicted - realized
        order = np.argsort(scores, kind="stable")
        ordered_scores = scores[order].copy()
        ordered_weights = weights[order].copy()
        payload = np.column_stack((ordered_scores, ordered_weights))
        digest = sha256(payload.tobytes()).hexdigest()
        return cls(float(alpha), ordered_scores, ordered_weights, digest)

    def optimism_correction(self, *, test_weight: float) -> float:
        if not math.isfinite(test_weight) or test_weight <= 0.0:
            raise ValueError("test_weight must be positive and finite")
        denominator = float(np.sum(self.ordered_weights)) + float(test_weight)
        cumulative = np.cumsum(self.ordered_weights) / denominator
        indices = np.flatnonzero(cumulative >= 1.0 - self.alpha)
        if indices.size == 0:
            return math.inf
        return float(self.ordered_scores[int(indices[0])])

    def certified_duration(
        self,
        predicted_horizon: float,
        *,
        test_weight: float,
        observation_age: float,
        compute_delay: float,
        communication_delay: float,
        actuation_delay: float,
        guard_time: float = 0.0,
    ) -> float:
        values = np.asarray(
            [
                predicted_horizon,
                observation_age,
                compute_delay,
                communication_delay,
                actuation_delay,
                guard_time,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("horizon and timing debits must be finite and non-negative")
        correction = self.optimism_correction(test_weight=test_weight)
        if not math.isfinite(correction):
            return 0.0
        return max(
            0.0,
            float(predicted_horizon - correction - np.sum(values[1:])),
        )
