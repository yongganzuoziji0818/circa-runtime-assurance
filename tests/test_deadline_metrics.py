import pytest

from agc_runtime_assurance.censored_validity import (
    FirstPassageObservation,
    FirstPassageObservationKind,
)
from agc_runtime_assurance.deadline_metrics import (
    DeadlineAuditRow,
    summarize_deadline_coverage,
)


def obs(
    time: float,
    kind: FirstPassageObservationKind,
    *,
    safe: bool = True,
) -> FirstPassageObservation:
    return FirstPassageObservation(time, kind, safe, f"audit:{kind.value}:{time}")


def test_event_deadline_coverage_is_directly_observed() -> None:
    summary = summarize_deadline_coverage(
        [
            DeadlineAuditRow(0.4, obs(0.5, FirstPassageObservationKind.EVENT)),
            DeadlineAuditRow(0.6, obs(0.5, FirstPassageObservationKind.EVENT)),
        ]
    )

    assert summary.known_covered == 1
    assert summary.observed_noncoverages == 1
    assert summary.best_case_noncoverage_rate == 0.5
    assert summary.worst_case_noncoverage_rate == 0.5


def test_deadline_beyond_administrative_cap_is_indeterminate() -> None:
    summary = summarize_deadline_coverage(
        [
            DeadlineAuditRow(
                0.4,
                obs(0.5, FirstPassageObservationKind.ADMINISTRATIVE_CENSOR),
            ),
            DeadlineAuditRow(
                0.6,
                obs(0.5, FirstPassageObservationKind.ADMINISTRATIVE_CENSOR),
            ),
        ]
    )

    assert summary.known_covered == 1
    assert summary.censoring_indeterminate == 1
    assert summary.best_case_noncoverage_rate == 0.0
    assert summary.worst_case_noncoverage_rate == 0.5
    assert summary.worst_case_noncoverage_upper > 0.5


def test_intervention_and_missing_records_are_separate_from_censoring() -> None:
    summary = summarize_deadline_coverage(
        [
            DeadlineAuditRow(0.2, obs(0.5, FirstPassageObservationKind.EVENT)),
            DeadlineAuditRow(
                0.2,
                obs(
                    0.3,
                    FirstPassageObservationKind.INTERVENTION_TRUNCATION,
                ),
            ),
            DeadlineAuditRow(
                0.0,
                obs(0.0, FirstPassageObservationKind.INVALID_OR_MISSING),
            ),
        ]
    )

    assert summary.valid_target_records == 1
    assert summary.intervention_truncations == 1
    assert summary.invalid_or_missing == 1


def test_summary_refuses_when_no_valid_target_record_exists() -> None:
    with pytest.raises(ValueError, match="event or administrative censor"):
        summarize_deadline_coverage(
            [
                DeadlineAuditRow(
                    0.0,
                    obs(0.0, FirstPassageObservationKind.INVALID_OR_MISSING),
                )
            ]
        )
