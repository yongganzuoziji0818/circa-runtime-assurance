import math

import numpy as np
import pytest

from agc_runtime_assurance.simultaneous_cells import (
    SimultaneousCellCertificate,
)
from agc_runtime_assurance.transversality import (
    ConstraintPassageEvidence,
    transport_team_horizon,
)


def test_calibration_uses_contextwise_worst_cell_error() -> None:
    predicted = np.ones((10, 2))
    realized = predicted.copy()
    realized[:, 0] -= 0.1
    realized[-1, 1] -= 0.8

    certificate = SimultaneousCellCertificate.fit(
        cell_ids=("easy", "hard"),
        predicted_horizons=predicted,
        realized_horizons=realized,
        alpha=0.2,
    )

    assert certificate.optimism_correction == pytest.approx(0.1)
    assert certificate.calibration_contexts == 10
    assert "not_action_or_acceptance_conditional" in certificate.coverage_semantics


def test_any_registered_cell_uses_the_same_simultaneous_correction() -> None:
    predicted = np.ones((20, 2))
    realized = predicted - np.array([0.2, 0.4])
    certificate = SimultaneousCellCertificate.fit(
        cell_ids=("left", "right"),
        predicted_horizons=predicted,
        realized_horizons=realized,
        alpha=0.1,
    )

    left = certificate.certified_duration(
        cell_id="left",
        predicted_representative_horizon=1.0,
        deterministic_transport_debit=0.1,
        observation_age=0.05,
        compute_delay=0.05,
        communication_delay=0.0,
        actuation_delay=0.0,
    )
    right = certificate.certified_duration(
        cell_id="right",
        predicted_representative_horizon=1.0,
        deterministic_transport_debit=0.1,
        observation_age=0.05,
        compute_delay=0.05,
        communication_delay=0.0,
        actuation_delay=0.0,
    )

    assert left == pytest.approx(0.4)
    assert right == left


def test_unknown_cell_fails_closed() -> None:
    certificate = SimultaneousCellCertificate.fit(
        cell_ids=("known",),
        predicted_horizons=np.ones((20, 1)),
        realized_horizons=np.ones((20, 1)),
        alpha=0.1,
    )

    assert certificate.certified_duration(
        cell_id="unknown",
        predicted_representative_horizon=1.0,
        deterministic_transport_debit=0.0,
        observation_age=0.0,
        compute_delay=0.0,
        communication_delay=0.0,
        actuation_delay=0.0,
    ) == 0.0


def test_small_calibration_set_returns_zero_duration() -> None:
    certificate = SimultaneousCellCertificate.fit(
        cell_ids=("only",),
        predicted_horizons=np.ones((5, 1)),
        realized_horizons=np.ones((5, 1)),
        alpha=0.05,
    )

    assert math.isinf(certificate.optimism_correction)
    assert certificate.certified_duration(
        cell_id="only",
        predicted_representative_horizon=1.0,
        deterministic_transport_debit=0.0,
        observation_age=0.0,
        compute_delay=0.0,
        communication_delay=0.0,
        actuation_delay=0.0,
    ) == 0.0


def test_composition_takes_minimum_of_statistical_and_transport_horizons() -> None:
    certificate = SimultaneousCellCertificate.fit(
        cell_ids=("cell",),
        predicted_horizons=np.ones((20, 1)),
        realized_horizons=np.ones((20, 1)) - 0.1,
        alpha=0.1,
    )
    transport = transport_team_horizon(
        [
            ConstraintPassageEvidence(
                constraint_id="coupling",
                reference_horizon=0.8,
                crossing_observed=True,
                constraint_lipschitz=1.0,
                transversality_kappa=1.0,
                crossing_tube_width=0.2,
                pre_tube_minimum_margin=0.2,
            )
        ],
        trajectory_state_deviation=0.1,
    )

    duration = certificate.certified_duration_from_team_transport(
        cell_id="cell",
        predicted_representative_horizon=1.0,
        team_transport=transport,
        observation_age=0.05,
        compute_delay=0.05,
        communication_delay=0.0,
        actuation_delay=0.0,
    )

    assert duration == pytest.approx(0.6)


def test_invalid_team_transport_cannot_be_replaced_by_numeric_debit() -> None:
    certificate = SimultaneousCellCertificate.fit(
        cell_ids=("cell",),
        predicted_horizons=np.ones((20, 1)),
        realized_horizons=np.ones((20, 1)),
        alpha=0.1,
    )
    transport = transport_team_horizon([], trajectory_state_deviation=0.0)

    assert certificate.certified_duration_from_team_transport(
        cell_id="cell",
        predicted_representative_horizon=1.0,
        team_transport=transport,
        observation_age=0.0,
        compute_delay=0.0,
        communication_delay=0.0,
        actuation_delay=0.0,
    ) == 0.0
