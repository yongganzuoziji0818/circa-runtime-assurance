import math

import numpy as np
import pytest

from agc_runtime_assurance.weighted_validity import (
    WeightedActionValidityCertificate,
)


def test_equal_weights_recover_split_conformal_rank() -> None:
    predicted = np.arange(1.0, 21.0)
    realized = predicted - np.linspace(0.01, 0.20, 20)
    certificate = WeightedActionValidityCertificate.fit(
        predicted,
        realized,
        np.ones(20),
        alpha=0.1,
    )

    assert certificate.optimism_correction(test_weight=1.0) == pytest.approx(0.19)


def test_large_test_weight_places_too_much_mass_at_infinity() -> None:
    certificate = WeightedActionValidityCertificate.fit(
        np.ones(10),
        np.ones(10),
        np.ones(10),
        alpha=0.1,
    )

    assert math.isinf(certificate.optimism_correction(test_weight=2.0))
    assert certificate.certified_duration(
        1.0,
        test_weight=2.0,
        observation_age=0.0,
        compute_delay=0.0,
        communication_delay=0.0,
        actuation_delay=0.0,
    ) == 0.0


def test_joint_weight_scaling_does_not_change_correction() -> None:
    predicted = np.ones(20)
    realized = predicted - np.linspace(0.0, 0.5, 20)
    first = WeightedActionValidityCertificate.fit(
        predicted, realized, np.linspace(1.0, 2.0, 20), alpha=0.2
    )
    second = WeightedActionValidityCertificate.fit(
        predicted, realized, 7.0 * np.linspace(1.0, 2.0, 20), alpha=0.2
    )

    assert first.optimism_correction(test_weight=1.5) == (
        second.optimism_correction(test_weight=10.5)
    )


def test_weighted_duration_subtracts_pipeline_debits() -> None:
    certificate = WeightedActionValidityCertificate.fit(
        np.ones(20),
        np.ones(20) - 0.2,
        np.ones(20),
        alpha=0.1,
    )

    duration = certificate.certified_duration(
        1.0,
        test_weight=1.0,
        observation_age=0.1,
        compute_delay=0.1,
        communication_delay=0.05,
        actuation_delay=0.05,
    )
    assert duration == pytest.approx(0.5)
