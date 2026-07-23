import pytest

from agc_runtime_assurance.counterexamples import (
    coupling_before_individual_failure,
    marginal_coverage_selection_failure,
    stale_observation_deadline_failure,
)


def test_individual_horizons_miss_earlier_team_coupling_failure():
    example = coupling_before_individual_failure()
    assert example.individual_censored_horizon == pytest.approx(0.3)
    assert example.team_first_violation_time == pytest.approx(0.2)
    assert example.team_first_violation_time < example.individual_censored_horizon


def test_marginal_conformal_coverage_can_be_zero_on_selected_actions():
    example = marginal_coverage_selection_failure()
    assert example.optimism_correction == 0.0
    assert example.marginal_coverage == pytest.approx(0.9)
    assert example.selected_fraction == pytest.approx(0.1)
    assert example.selected_hard_coverage == 0.0


def test_omitting_observation_age_executes_past_true_failure_time():
    example = stale_observation_deadline_failure()
    assert example.duration_without_age_debit > example.true_remaining_safe_time
    assert example.duration_with_all_debits <= example.true_remaining_safe_time
