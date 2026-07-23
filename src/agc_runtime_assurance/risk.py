"""Team-level safety scores and finite-sample calibration primitives."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class ConstraintMargins:
    """Positive margins are safe; negative margins indicate violation."""

    uav_local: float
    ugv_local: float
    coupling: float
    mission: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.uav_local, self.ugv_local, self.coupling, self.mission], dtype=float
        )


def team_safety_score(margins: ConstraintMargins, scales: np.ndarray | None = None) -> float:
    """Return the maximum normalized violation severity across team constraints."""

    values = margins.as_array()
    if scales is None:
        scales = np.ones_like(values)
    scales = np.asarray(scales, dtype=float)
    if scales.shape != values.shape or np.any(scales <= 0):
        raise ValueError("scales must be positive and match the four constraint groups")
    return float(np.max(np.maximum(0.0, -values) / scales))


@dataclass(frozen=True)
class ConformalQuantileCertificate:
    """Split-conformal upper prediction bound for a scalar team safety score.

    Under exchangeability, ``P(score_test <= threshold) >= 1 - alpha``.  This is
    a marginal coverage statement, not an unconditional closed-loop safety proof.
    """

    alpha: float
    threshold: float
    calibration_size: int
    calibration_hash: str

    @classmethod
    def fit(cls, scores: np.ndarray, alpha: float) -> "ConformalQuantileCertificate":
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if scores.size == 0 or not np.all(np.isfinite(scores)):
            raise ValueError("calibration scores must be non-empty and finite")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")
        ordered = np.sort(scores)
        rank = math.ceil((scores.size + 1) * (1.0 - alpha))
        threshold = math.inf if rank > scores.size else float(ordered[rank - 1])
        digest = sha256(ordered.tobytes()).hexdigest()
        return cls(alpha, threshold, int(scores.size), digest)

    def covers(self, score: float) -> bool:
        return bool(float(score) <= self.threshold)

    def fingerprint(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, allow_nan=True).encode()
        return sha256(payload).hexdigest()


def clopper_pearson_upper(violations: int, trials: int, delta: float = 0.05) -> float:
    """Exact one-sided upper confidence bound for an episode violation rate."""

    if not 0 <= violations <= trials or trials <= 0:
        raise ValueError("require 0 <= violations <= trials and trials > 0")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between 0 and 1")
    if violations == trials:
        return 1.0
    return float(beta.ppf(1.0 - delta, violations + 1, trials - violations))
