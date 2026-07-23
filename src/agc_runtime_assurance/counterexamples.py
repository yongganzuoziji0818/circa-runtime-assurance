"""Deterministic falsification examples for P4's non-triviality gates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .validity import ActionValidityCertificate, first_violation_time


@dataclass(frozen=True)
class CouplingCounterexample:
    individual_censored_horizon: float
    team_first_violation_time: float
    times: np.ndarray
    margins: np.ndarray


def coupling_before_individual_failure() -> CouplingCounterexample:
    """Both agents remain locally safe while their coupling margin fails."""

    times = np.array([0.0, 0.1, 0.2, 0.3])
    margins = np.array([
        [1.0, 1.0, 0.8, 1.0],
        [0.9, 0.9, 0.3, 1.0],
        [0.8, 0.8, -0.1, 1.0],
        [0.7, 0.7, -0.4, 1.0],
    ])
    local_only = margins[:, :2]
    return CouplingCounterexample(
        individual_censored_horizon=first_violation_time(times, local_only),
        team_first_violation_time=first_violation_time(times, margins),
        times=times,
        margins=margins,
    )


@dataclass(frozen=True)
class SelectionCounterexample:
    optimism_correction: float
    marginal_coverage: float
    selected_hard_coverage: float
    selected_fraction: float


def marginal_coverage_selection_failure() -> SelectionCounterexample:
    """Split conformal is marginally valid but fails on a selected hard subset.

    Calibration has 90 easy and 9 hard actions.  At alpha=.1, the finite-sample
    order statistic is zero.  A fresh population with 90% easy and 10% hard
    actions achieves exactly 90% marginal coverage, while a controller that
    selects the hard action subset has zero coverage on executed actions.
    """

    predicted_calibration = np.ones(99)
    realized_calibration = np.concatenate((np.ones(90), np.zeros(9)))
    certificate = ActionValidityCertificate.fit(
        predicted_calibration, realized_calibration, alpha=0.1
    )
    predicted_test = np.ones(1000)
    realized_test = np.concatenate((np.ones(900), np.zeros(100)))
    certified_lower = predicted_test - certificate.optimism_correction
    covered = certified_lower <= realized_test
    selected_hard = np.arange(1000) >= 900
    return SelectionCounterexample(
        optimism_correction=certificate.optimism_correction,
        marginal_coverage=float(np.mean(covered)),
        selected_hard_coverage=float(np.mean(covered[selected_hard])),
        selected_fraction=float(np.mean(selected_hard)),
    )


@dataclass(frozen=True)
class StalenessCounterexample:
    true_remaining_safe_time: float
    duration_without_age_debit: float
    duration_with_all_debits: float


def stale_observation_deadline_failure() -> StalenessCounterexample:
    """A statistically corrected horizon still overruns failure if AoI is omitted."""

    certificate = ActionValidityCertificate(
        alpha=0.1,
        optimism_correction=0.2,
        calibration_size=99,
        calibration_hash="synthetic_counterexample",
    )
    predicted_horizon = 1.0
    without_age = certificate.certified_duration(
        predicted_horizon,
        observation_age=0.0,
        compute_delay=0.1,
        communication_delay=0.05,
        actuation_delay=0.05,
        guard_time=0.0,
    )
    with_all = certificate.certified_duration(
        predicted_horizon,
        observation_age=0.4,
        compute_delay=0.1,
        communication_delay=0.05,
        actuation_delay=0.05,
        guard_time=0.0,
    )
    return StalenessCounterexample(0.45, without_age, with_all)
