"""Conservative conformal lower bounds from right-censored horizons.

For an administrative right-censoring time C and a true first-passage time T,
the fully observed outcome is min(T, C).  A marginal conformal lower bound on
that observed outcome is also a lower bound on T by deterministic dominance.
This baseline is valid but potentially very conservative.  Intervention
truncation and missing/corrupt runs are not administrative censoring and are
rejected from calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Iterable

import numpy as np

from .contracts import ActionEnvelope
from .validity import ActionValidityCertificate


class FirstPassageObservationKind(str, Enum):
    EVENT = "event"
    ADMINISTRATIVE_CENSOR = "administrative_censor"
    INTERVENTION_TRUNCATION = "intervention_truncation"
    INVALID_OR_MISSING = "invalid_or_missing"


@dataclass(frozen=True)
class FirstPassageObservation:
    observed_time: float
    kind: FirstPassageObservationKind
    safe_through_observed_time: bool
    provenance: str

    def validate(self) -> None:
        if not np.isfinite(self.observed_time) or self.observed_time < 0.0:
            raise ValueError("observed_time must be finite and non-negative")
        if not isinstance(self.kind, FirstPassageObservationKind):
            raise ValueError("kind must be a FirstPassageObservationKind")
        if not self.provenance:
            raise ValueError("observation provenance is required")
        if (
            self.kind is FirstPassageObservationKind.ADMINISTRATIVE_CENSOR
            and not self.safe_through_observed_time
        ):
            raise ValueError(
                "administrative censoring requires verified safety through the cap"
            )


@dataclass(frozen=True)
class NaiveCensoredValidityCertificate:
    """Split-conformal LPB on min(T, C), hence a conservative LPB on T."""

    certificate: ActionValidityCertificate
    event_count: int
    administrative_censor_count: int
    observation_hash: str

    coverage_semantics = (
        "marginal_lower_bound_via_observed_min_time_dominance_"
        "not_conditional_not_arbitrary_drift"
    )

    @classmethod
    def fit(
        cls,
        predicted_horizons: np.ndarray,
        observations: Iterable[FirstPassageObservation],
        *,
        alpha: float,
    ) -> "NaiveCensoredValidityCertificate":
        predicted = np.asarray(predicted_horizons, dtype=float).reshape(-1)
        records = tuple(observations)
        if predicted.size == 0 or predicted.size != len(records):
            raise ValueError("predictions and observations must be non-empty and aligned")
        if not np.all(np.isfinite(predicted)) or np.any(predicted < 0.0):
            raise ValueError("predicted horizons must be finite and non-negative")

        allowed = {
            FirstPassageObservationKind.EVENT,
            FirstPassageObservationKind.ADMINISTRATIVE_CENSOR,
        }
        for record in records:
            record.validate()
            if record.kind not in allowed:
                raise ValueError(
                    f"{record.kind.value} is not valid right-censoring calibration data"
                )

        observed = np.asarray(
            [record.observed_time for record in records], dtype=float
        )
        certificate = ActionValidityCertificate.fit(
            predicted,
            observed,
            alpha=alpha,
        )
        event_count = sum(
            record.kind is FirstPassageObservationKind.EVENT for record in records
        )
        censor_count = len(records) - event_count
        digest = sha256()
        for record in records:
            digest.update(
                (
                    f"{record.observed_time:.17g}\t{record.kind.value}\t"
                    f"{int(record.safe_through_observed_time)}\t"
                    f"{record.provenance}\n"
                ).encode("utf-8")
            )
        return cls(
            certificate=certificate,
            event_count=int(event_count),
            administrative_censor_count=int(censor_count),
            observation_hash=digest.hexdigest(),
        )

    @classmethod
    def fit_training_conditional(
        cls,
        predicted_horizons: np.ndarray,
        observations: Iterable[FirstPassageObservation],
        *,
        alpha: float,
        delta: float,
    ) -> "NaiveCensoredValidityCertificate":
        """Inflate the censored-outcome order statistic for a PAC-style claim."""

        predicted = np.asarray(predicted_horizons, dtype=float).reshape(-1)
        records = tuple(observations)
        marginal = cls.fit(predicted, records, alpha=alpha)
        observed = np.asarray(
            [record.observed_time for record in records], dtype=float
        )
        conditional = ActionValidityCertificate.fit_training_conditional(
            predicted,
            observed,
            alpha=alpha,
            delta=delta,
        )
        return cls(
            certificate=conditional,
            event_count=marginal.event_count,
            administrative_censor_count=marginal.administrative_censor_count,
            observation_hash=marginal.observation_hash,
        )

    @property
    def optimism_correction(self) -> float:
        return self.certificate.optimism_correction

    @property
    def calibration_size(self) -> int:
        return self.certificate.calibration_size

    @property
    def censoring_fraction(self) -> float:
        return self.administrative_censor_count / self.calibration_size

    def certified_duration(
        self,
        predicted_horizon: float,
        *,
        observation_age: float,
        compute_delay: float,
        communication_delay: float,
        actuation_delay: float,
        guard_time: float = 0.0,
    ) -> float:
        return self.certificate.certified_duration(
            predicted_horizon,
            observation_age=observation_age,
            compute_delay=compute_delay,
            communication_delay=communication_delay,
            actuation_delay=actuation_delay,
            guard_time=guard_time,
        )

    def issue(
        self,
        action: np.ndarray,
        *,
        issued_at: float,
        predicted_horizon: float,
        observation_age: float,
        compute_delay: float,
        communication_delay: float,
        actuation_delay: float,
        guard_time: float = 0.0,
    ) -> ActionEnvelope:
        duration = self.certified_duration(
            predicted_horizon,
            observation_age=observation_age,
            compute_delay=compute_delay,
            communication_delay=communication_delay,
            actuation_delay=actuation_delay,
            guard_time=guard_time,
        )
        return ActionEnvelope(
            action=np.asarray(action, dtype=float),
            issued_at=float(issued_at),
            valid_until=float(issued_at) + duration,
            source="naive_censored_outcome_lpb",
            constraint_state=(
                "censored_outcome_marginal_lpb"
                if duration > 0.0
                else "reject_zero_horizon"
            ),
        )
