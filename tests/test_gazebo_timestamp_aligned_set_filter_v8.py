from __future__ import annotations

import numpy as np

from agc_runtime_assurance.gazebo_robust_backup_filter import (
    GazeboPlanarPlant,
    RobustBackupConfig,
    propagate_planar_state,
)
from agc_runtime_assurance.gazebo_timestamp_aligned_set_filter import (
    TimestampAlignedSetBackupFilter,
    TimestampAlignmentConfig,
    align_async_state_history,
)


def _plant() -> GazeboPlanarPlant:
    return GazeboPlanarPlant(1.0, 0.1, 0.1, 0.2, dt=0.1, speed_limit_mps=0.5)


def _history(steps: int = 4) -> tuple[list[np.ndarray], list[np.ndarray]]:
    plant = _plant()
    state = np.array([-2.0, 0.1, 2.0, 0.3, 0.0, 0.0, 2.0, -0.1, -0.3, 0.0])
    applied = np.zeros(5)
    states = [state.copy()]
    actions = [applied.copy()]
    commands = (
        np.array([0.2, 0.0, 0.0, -0.2, 0.0]),
        np.array([0.1, 0.1, 0.0, -0.1, -0.1]),
        np.array([-0.1, 0.2, 0.0, 0.1, -0.2]),
        np.array([-0.2, 0.0, 0.0, 0.2, 0.0]),
    )
    for command in commands[:steps]:
        state, applied = propagate_planar_state(state, applied, command, plant)
        states.append(state.copy())
        actions.append(applied.copy())
    return states, actions


def test_zero_age_alignment_preserves_latest_center_and_registered_radii() -> None:
    states, actions = _history(2)
    aligned = align_async_state_history(
        states, actions, 0, 0, _plant(), TimestampAlignmentConfig()
    )
    assert np.allclose(aligned.center, states[-1])
    assert np.allclose(aligned.radius[[0, 1, 6, 7]], 0.02)
    assert np.allclose(aligned.radius[[3, 4, 8, 9]], 0.02)
    assert aligned.local_age_steps == aligned.neighbor_age_steps == 0


def test_delayed_alignment_encloses_current_truth_under_registered_model() -> None:
    states, actions = _history(4)
    aligned = align_async_state_history(
        states, actions, 2, 1, _plant(), TimestampAlignmentConfig()
    )
    relevant = np.array([0, 1, 3, 4, 6, 7, 8, 9])
    error = np.abs(aligned.center[relevant] - states[-1][relevant])
    assert np.all(error <= aligned.radius[relevant] + 1e-12)
    assert aligned.local_age_steps == 2
    assert aligned.neighbor_age_steps == 3


def test_common_mode_bias_cancels_from_relative_separation_center() -> None:
    states, actions = _history(3)
    unbiased = align_async_state_history(
        states, actions, 2, 1, _plant(), TimestampAlignmentConfig()
    )
    biased = align_async_state_history(
        states,
        actions,
        2,
        1,
        _plant(),
        TimestampAlignmentConfig(),
        observed_common_mode_bias_m=0.15,
    )
    assert np.allclose(
        unbiased.center[:2] - unbiased.center[6:8],
        biased.center[:2] - biased.center[6:8],
    )
    assert not np.allclose(unbiased.center[:2], biased.center[:2])


def test_alignment_provenance_binds_applied_action_history() -> None:
    states, actions = _history(3)
    first = align_async_state_history(
        states, actions, 2, 1, _plant(), TimestampAlignmentConfig()
    )
    changed = [value.copy() for value in actions]
    changed[-1][0] += 0.01
    second = align_async_state_history(
        states, changed, 2, 1, _plant(), TimestampAlignmentConfig()
    )
    assert first.applied_action_history_digest != second.applied_action_history_digest
    assert first.provenance_hash != second.provenance_hash


def test_invalid_age_fails_closed_without_numeric_certificate() -> None:
    states, actions = _history(3)
    safety_filter = TimestampAlignedSetBackupFilter(
        _plant(),
        RobustBackupConfig(operational_separation_m=1.5, action_limit=0.5),
        TimestampAlignmentConfig(),
    )
    decision = safety_filter.align_and_decide(
        states,
        actions,
        observation_delay_steps=2,
        communication_delay_steps=4,
        nominal_action=np.zeros(5),
    )
    assert decision.evidence_valid is False
    assert decision.certificate_emitted is False
    assert decision.certificate is None
    assert decision.intervened is True and decision.fail_closed is True
    assert decision.reason.startswith("alignment_evidence_refused:")


def test_malformed_state_history_returns_bounded_stop_without_certificate() -> None:
    safety_filter = TimestampAlignedSetBackupFilter(
        _plant(),
        RobustBackupConfig(operational_separation_m=1.5, action_limit=0.5),
        TimestampAlignmentConfig(),
    )
    decision = safety_filter.align_and_decide(
        [], [], 0, 0, np.full(5, np.nan)
    )
    assert np.array_equal(decision.action, np.zeros(5))
    assert decision.evidence_valid is False
    assert decision.certificate_emitted is False
    assert decision.certificate is None
    assert decision.intervened is True and decision.fail_closed is True


def test_full_set_is_never_less_conservative_than_point_ablation_for_same_plan() -> None:
    states, actions = _history(3)
    aligned = align_async_state_history(
        states, actions, 2, 1, _plant(), TimestampAlignmentConfig()
    )
    safety_filter = TimestampAlignedSetBackupFilter(
        _plant(),
        RobustBackupConfig(operational_separation_m=1.5, action_limit=0.5),
        TimestampAlignmentConfig(),
    )
    command = np.zeros(5)
    full = safety_filter.evaluate_plan(aligned, actions[-1], command)
    point = safety_filter.evaluate_plan(
        aligned, actions[-1], command, point_ablation=True
    )
    assert full.minimum_tightened_margin_m <= point.minimum_tightened_margin_m + 1e-12


def test_valid_decision_emits_a_provenance_bound_certificate() -> None:
    states, actions = _history(3)
    safety_filter = TimestampAlignedSetBackupFilter(
        _plant(),
        RobustBackupConfig(operational_separation_m=1.5, action_limit=0.5),
        TimestampAlignmentConfig(),
    )
    decision = safety_filter.align_and_decide(
        states, actions, 2, 1, np.zeros(5)
    )
    assert decision.evidence_valid is True
    assert decision.certificate_emitted is True
    assert decision.certificate is not None
    assert decision.certificate["provenance_hash"] == decision.aligned_state.provenance_hash
    assert decision.certificate["validity_interval_steps"] == [3, 3]
