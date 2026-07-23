"""Executable evidence gates for first-passage-time transport.

This module does not estimate proof constants from development outcomes. It
only checks externally supplied analytic, interval, or otherwise frozen
evidence and fails closed when the local crossing argument is insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable


def _positive_finite(value: float | None) -> bool:
    return value is not None and isfinite(value) and value > 0.0


@dataclass(frozen=True)
class ConstraintPassageEvidence:
    """Frozen evidence for one team constraint."""

    constraint_id: str
    reference_horizon: float
    crossing_observed: bool
    constraint_lipschitz: float
    transversality_kappa: float | None = None
    crossing_tube_width: float | None = None
    pre_tube_minimum_margin: float | None = None
    censored_minimum_margin: float | None = None

    def __post_init__(self) -> None:
        if not self.constraint_id:
            raise ValueError("constraint_id must be non-empty")
        if not isfinite(self.reference_horizon) or self.reference_horizon <= 0.0:
            raise ValueError("reference_horizon must be positive and finite")
        if (
            not isfinite(self.constraint_lipschitz)
            or self.constraint_lipschitz < 0.0
        ):
            raise ValueError("constraint_lipschitz must be non-negative and finite")

        if self.crossing_observed:
            if not _positive_finite(self.transversality_kappa):
                raise ValueError(
                    "crossing evidence requires positive finite transversality_kappa"
                )
            if not _positive_finite(self.crossing_tube_width):
                raise ValueError(
                    "crossing evidence requires positive finite crossing_tube_width"
                )
            if not _positive_finite(self.pre_tube_minimum_margin):
                raise ValueError(
                    "crossing evidence requires positive finite "
                    "pre_tube_minimum_margin"
                )
        elif not _positive_finite(self.censored_minimum_margin):
            raise ValueError(
                "censored evidence requires positive finite censored_minimum_margin"
            )


@dataclass(frozen=True)
class ConstraintTransportResult:
    constraint_id: str
    valid: bool
    transported_horizon: float
    time_debit: float
    barrier_deviation: float
    reason: str


@dataclass(frozen=True)
class TeamTransportResult:
    valid: bool
    transported_team_horizon: float
    critical_constraint: str | None
    constraints: tuple[ConstraintTransportResult, ...]
    reason: str


def transport_constraint_horizon(
    evidence: ConstraintPassageEvidence,
    trajectory_state_deviation: float,
) -> ConstraintTransportResult:
    """Transport a reference first-passage horizon under a state-error bound."""

    if (
        not isfinite(trajectory_state_deviation)
        or trajectory_state_deviation < 0.0
    ):
        raise ValueError(
            "trajectory_state_deviation must be non-negative and finite"
        )

    barrier_deviation = (
        evidence.constraint_lipschitz * trajectory_state_deviation
    )

    if evidence.crossing_observed:
        assert evidence.pre_tube_minimum_margin is not None
        assert evidence.transversality_kappa is not None
        assert evidence.crossing_tube_width is not None

        if barrier_deviation > evidence.pre_tube_minimum_margin:
            return ConstraintTransportResult(
                constraint_id=evidence.constraint_id,
                valid=False,
                transported_horizon=0.0,
                time_debit=float("inf"),
                barrier_deviation=barrier_deviation,
                reason="pre_crossing_tube_margin_not_robust",
            )

        time_debit = barrier_deviation / evidence.transversality_kappa
        if time_debit > evidence.crossing_tube_width:
            return ConstraintTransportResult(
                constraint_id=evidence.constraint_id,
                valid=False,
                transported_horizon=0.0,
                time_debit=time_debit,
                barrier_deviation=barrier_deviation,
                reason="transversality_tube_too_narrow",
            )

        return ConstraintTransportResult(
            constraint_id=evidence.constraint_id,
            valid=True,
            transported_horizon=max(
                0.0, evidence.reference_horizon - time_debit
            ),
            time_debit=time_debit,
            barrier_deviation=barrier_deviation,
            reason="transported_by_uniform_transversality",
        )

    assert evidence.censored_minimum_margin is not None
    if barrier_deviation >= evidence.censored_minimum_margin:
        return ConstraintTransportResult(
            constraint_id=evidence.constraint_id,
            valid=False,
            transported_horizon=0.0,
            time_debit=float("inf"),
            barrier_deviation=barrier_deviation,
            reason="censored_horizon_margin_not_robust",
        )

    return ConstraintTransportResult(
        constraint_id=evidence.constraint_id,
        valid=True,
        transported_horizon=evidence.reference_horizon,
        time_debit=0.0,
        barrier_deviation=barrier_deviation,
        reason="censored_horizon_remains_strictly_safe",
    )


def transport_team_horizon(
    evidence: Iterable[ConstraintPassageEvidence],
    trajectory_state_deviation: float,
) -> TeamTransportResult:
    """Return the minimum transported horizon, failing closed on any gap."""

    results = tuple(
        transport_constraint_horizon(item, trajectory_state_deviation)
        for item in evidence
    )
    if not results:
        return TeamTransportResult(
            valid=False,
            transported_team_horizon=0.0,
            critical_constraint=None,
            constraints=(),
            reason="no_constraint_evidence",
        )

    invalid = next((item for item in results if not item.valid), None)
    if invalid is not None:
        return TeamTransportResult(
            valid=False,
            transported_team_horizon=0.0,
            critical_constraint=invalid.constraint_id,
            constraints=results,
            reason=f"constraint_transport_failed:{invalid.reason}",
        )

    critical = min(results, key=lambda item: item.transported_horizon)
    return TeamTransportResult(
        valid=True,
        transported_team_horizon=critical.transported_horizon,
        critical_constraint=critical.constraint_id,
        constraints=results,
        reason="all_team_constraints_transportable",
    )
