"""Censoring-aware audit metrics for action-validity deadline coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .censored_validity import (
    FirstPassageObservation,
    FirstPassageObservationKind,
)
from .risk import clopper_pearson_upper


@dataclass(frozen=True)
class DeadlineAuditRow:
    issued_duration: float
    observation: FirstPassageObservation

    def validate(self) -> None:
        if not np.isfinite(self.issued_duration) or self.issued_duration < 0.0:
            raise ValueError("issued_duration must be finite and non-negative")
        self.observation.validate()


@dataclass(frozen=True)
class DeadlineCoverageSummary:
    event_records: int
    administrative_censors: int
    known_covered: int
    observed_noncoverages: int
    censoring_indeterminate: int
    intervention_truncations: int
    invalid_or_missing: int
    valid_target_records: int
    best_case_noncoverage_rate: float
    worst_case_noncoverage_rate: float
    worst_case_noncoverage_upper: float


def summarize_deadline_coverage(
    rows: Iterable[DeadlineAuditRow],
    *,
    delta: float = 0.05,
) -> DeadlineCoverageSummary:
    """Return identifiable and worst-case deadline noncoverage rates.

    For an administrative censor at C, a duration <= C is known covered because
    T >= C.  A duration > C is indeterminate rather than a success.  The
    reported worst-case rate counts every indeterminate record as noncoverage.
    Intervention truncation and invalid/missing runs are reported separately
    and excluded from the right-censoring target denominator.
    """

    counts = {
        "event": 0,
        "censor": 0,
        "covered": 0,
        "failure": 0,
        "indeterminate": 0,
        "intervention": 0,
        "invalid": 0,
    }
    observed_any = False
    for row in rows:
        observed_any = True
        row.validate()
        kind = row.observation.kind
        if kind is FirstPassageObservationKind.EVENT:
            counts["event"] += 1
            if row.issued_duration > row.observation.observed_time:
                counts["failure"] += 1
            else:
                counts["covered"] += 1
        elif kind is FirstPassageObservationKind.ADMINISTRATIVE_CENSOR:
            counts["censor"] += 1
            if row.issued_duration <= row.observation.observed_time:
                counts["covered"] += 1
            else:
                counts["indeterminate"] += 1
        elif kind is FirstPassageObservationKind.INTERVENTION_TRUNCATION:
            counts["intervention"] += 1
        else:
            counts["invalid"] += 1

    if not observed_any:
        raise ValueError("at least one deadline audit row is required")
    valid = counts["event"] + counts["censor"]
    if valid == 0:
        raise ValueError("at least one event or administrative censor is required")
    best_failures = counts["failure"]
    worst_failures = counts["failure"] + counts["indeterminate"]
    return DeadlineCoverageSummary(
        event_records=counts["event"],
        administrative_censors=counts["censor"],
        known_covered=counts["covered"],
        observed_noncoverages=counts["failure"],
        censoring_indeterminate=counts["indeterminate"],
        intervention_truncations=counts["intervention"],
        invalid_or_missing=counts["invalid"],
        valid_target_records=valid,
        best_case_noncoverage_rate=best_failures / valid,
        worst_case_noncoverage_rate=worst_failures / valid,
        worst_case_noncoverage_upper=clopper_pearson_upper(
            worst_failures,
            valid,
            delta,
        ),
    )
