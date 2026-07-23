import numpy as np
import pytest

from agc_runtime_assurance.fallback_monitor import ConformalStoppingTimeMonitor


def test_large_anomaly_score_triggers_fallback_under_randomized_rank_rule():
    monitor = ConformalStoppingTimeMonitor(
        np.array([0.1, 0.2, 0.3]), risk_tolerance=0.2, random_seed=7
    )
    decision = monitor.evaluate(0.25)
    assert decision.randomized_rank == pytest.approx(0.5)
    assert decision.alarm


def test_small_anomaly_score_does_not_trigger():
    monitor = ConformalStoppingTimeMonitor(
        np.array([0.1, 0.2, 0.3]), risk_tolerance=0.2, random_seed=7
    )
    decision = monitor.evaluate(0.05)
    assert decision.randomized_rank == 1.0
    assert not decision.alarm


def test_tie_breaking_is_reproducible_from_frozen_seed():
    first = ConformalStoppingTimeMonitor(
        np.array([0.2, 0.2, 0.3]), risk_tolerance=0.2, random_seed=11
    ).evaluate(0.2)
    second = ConformalStoppingTimeMonitor(
        np.array([0.2, 0.2, 0.3]), risk_tolerance=0.2, random_seed=11
    ).evaluate(0.2)
    assert first == second


def test_nontrivial_gate_matches_paper_sample_condition():
    insufficient = ConformalStoppingTimeMonitor(
        np.zeros(18), risk_tolerance=0.05, random_seed=0
    )
    sufficient = ConformalStoppingTimeMonitor(
        np.zeros(19), risk_tolerance=0.05, random_seed=0
    )
    assert insufficient.minimum_nontrivial_fault_samples == 19
    assert not insufficient.nontrivial_sample_gate
    assert sufficient.nontrivial_sample_gate


def test_empty_fault_calibration_is_explicitly_nontrivial_false():
    monitor = ConformalStoppingTimeMonitor(
        np.array([]), risk_tolerance=0.1, random_seed=0
    )
    assert not monitor.nontrivial_sample_gate
    assert not monitor.evaluate(100.0).alarm
