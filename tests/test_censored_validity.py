import numpy as np
import pytest

from agc_runtime_assurance.censored_validity import (
    FirstPassageObservation,
    FirstPassageObservationKind,
    NaiveCensoredValidityCertificate,
)


def observation(
    time: float,
    kind: FirstPassageObservationKind,
    *,
    safe: bool = True,
) -> FirstPassageObservation:
    return FirstPassageObservation(
        observed_time=time,
        kind=kind,
        safe_through_observed_time=safe,
        provenance=f"synthetic:{time}:{kind.value}",
    )


def test_naive_censored_lpb_calibrates_on_observed_minimum_time() -> None:
    predicted = np.ones(20)
    observations = [
        observation(0.8, FirstPassageObservationKind.EVENT)
        for _ in range(10)
    ] + [
        observation(0.9, FirstPassageObservationKind.ADMINISTRATIVE_CENSOR)
        for _ in range(10)
    ]

    certificate = NaiveCensoredValidityCertificate.fit(
        predicted, observations, alpha=0.1
    )

    assert certificate.optimism_correction == pytest.approx(0.2)
    assert certificate.event_count == 10
    assert certificate.administrative_censor_count == 10
    assert certificate.censoring_fraction == 0.5
    assert "observed_min_time_dominance" in certificate.coverage_semantics


def test_intervention_truncation_is_not_treated_as_right_censoring() -> None:
    with pytest.raises(ValueError, match="not valid right-censoring"):
        NaiveCensoredValidityCertificate.fit(
            np.ones(20),
            [
                observation(
                    0.5,
                    FirstPassageObservationKind.INTERVENTION_TRUNCATION,
                )
                for _ in range(20)
            ],
            alpha=0.1,
        )


def test_unverified_administrative_censor_is_rejected() -> None:
    with pytest.raises(ValueError, match="verified safety"):
        NaiveCensoredValidityCertificate.fit(
            np.ones(20),
            [
                observation(
                    0.5,
                    FirstPassageObservationKind.ADMINISTRATIVE_CENSOR,
                    safe=False,
                )
                for _ in range(20)
            ],
            alpha=0.1,
        )


def test_censored_certificate_subtracts_runtime_debits() -> None:
    certificate = NaiveCensoredValidityCertificate.fit(
        np.ones(20),
        [
            observation(0.8, FirstPassageObservationKind.EVENT)
            for _ in range(20)
        ],
        alpha=0.1,
    )

    assert certificate.certified_duration(
        1.0,
        observation_age=0.1,
        compute_delay=0.05,
        communication_delay=0.05,
        actuation_delay=0.05,
        guard_time=0.05,
    ) == pytest.approx(0.5)


def test_issue_preserves_censoring_semantics_in_audit_envelope() -> None:
    certificate = NaiveCensoredValidityCertificate.fit(
        np.ones(20),
        [
            observation(
                1.0,
                FirstPassageObservationKind.ADMINISTRATIVE_CENSOR,
            )
            for _ in range(20)
        ],
        alpha=0.1,
    )

    envelope = certificate.issue(
        np.zeros(2),
        issued_at=3.0,
        predicted_horizon=1.0,
        observation_age=0.0,
        compute_delay=0.0,
        communication_delay=0.0,
        actuation_delay=0.0,
    )
    assert envelope.constraint_state == "censored_outcome_marginal_lpb"
    assert envelope.source == "naive_censored_outcome_lpb"


def test_training_conditional_censored_lpb_is_no_less_conservative() -> None:
    predicted = np.ones(201)
    observations = [
        observation(
            1.0 - error,
            FirstPassageObservationKind.ADMINISTRATIVE_CENSOR,
        )
        for error in np.linspace(0.0, 0.8, 201)
    ]
    marginal = NaiveCensoredValidityCertificate.fit(
        predicted, observations, alpha=0.1
    )
    conditional = NaiveCensoredValidityCertificate.fit_training_conditional(
        predicted,
        observations,
        alpha=0.1,
        delta=0.05,
    )

    assert conditional.optimism_correction >= marginal.optimism_correction
    assert conditional.certificate.calibration_delta == 0.05
