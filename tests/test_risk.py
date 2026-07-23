import math

import numpy as np

from agc_runtime_assurance.risk import (
    ConformalQuantileCertificate,
    ConstraintMargins,
    clopper_pearson_upper,
    team_safety_score,
)


def test_team_score_includes_coupled_constraint():
    score = team_safety_score(ConstraintMargins(1.0, 1.0, -0.25, 2.0))
    assert score == 0.25


def test_conformal_rank_and_hash_are_deterministic():
    scores = np.arange(20, dtype=float)
    a = ConformalQuantileCertificate.fit(scores, alpha=0.2)
    b = ConformalQuantileCertificate.fit(scores[::-1], alpha=0.2)
    assert a.threshold == 16.0
    assert a.fingerprint() == b.fingerprint()


def test_small_calibration_set_fails_conservatively():
    certificate = ConformalQuantileCertificate.fit(np.array([0.0, 1.0]), alpha=0.1)
    assert math.isinf(certificate.threshold)


def test_exact_binomial_upper_bound_is_not_optimistic():
    assert 0.0 < clopper_pearson_upper(0, 100) < 0.05
    assert clopper_pearson_upper(100, 100) == 1.0
