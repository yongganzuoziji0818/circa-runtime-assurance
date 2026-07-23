import numpy as np
import pytest

from agc_runtime_assurance.action_cells import (
    CellConditionalValidityBank,
    FirstPassageCellSpec,
)


def _spec(cell_id, action_sensitivity=0.0, state_sensitivity=0.0):
    return FirstPassageCellSpec(
        cell_id=cell_id, action_radius=0.5, state_radius=0.5,
        transversality_kappa=0.5,
        action_barrier_sensitivity=action_sensitivity,
        state_barrier_sensitivity=state_sensitivity,
        proof_reference="synthetic-transverse-crossing-bound",
    )


def test_per_cell_calibration_exposes_hard_action_failure_hidden_marginally():
    labels = ["easy"] * 90 + ["hard"] * 20
    predicted = np.ones(110)
    realized = np.concatenate((np.ones(90), np.zeros(20)))
    bank = CellConditionalValidityBank.fit(
        cell_ids=labels, predicted_representative_horizons=predicted,
        realized_representative_horizons=realized,
        specs=[_spec("easy"), _spec("hard")], alpha=0.1,
        minimum_cell_samples=10,
    )
    assert bank.records["easy"].certificate.optimism_correction == 0.0
    assert bank.records["hard"].certificate.optimism_correction == 1.0
    assert bank.coverage_semantics.endswith("not_selection_conditional")


def test_transversality_derived_transport_reduces_duration():
    bank = CellConditionalValidityBank.fit(
        cell_ids=["c"] * 20,
        predicted_representative_horizons=np.ones(20),
        realized_representative_horizons=np.ones(20),
        specs=[_spec("c", action_sensitivity=1.0, state_sensitivity=0.5)],
        alpha=0.1, minimum_cell_samples=10,
    )
    duration = bank.certified_duration(
        cell_id="c", predicted_representative_horizon=1.0,
        action_deviation=0.2, state_uncertainty=0.1,
        observation_age=0.0, compute_delay=0.0,
        communication_delay=0.0, actuation_delay=0.0,
    )
    # L_u=2, L_x=1 from division by kappa=.5.
    assert duration == pytest.approx(0.5)


def test_unknown_undersampled_or_out_of_radius_cell_fails_closed():
    bank = CellConditionalValidityBank.fit(
        cell_ids=["c"] * 3,
        predicted_representative_horizons=np.ones(3),
        realized_representative_horizons=np.ones(3),
        specs=[_spec("c")], alpha=0.1, minimum_cell_samples=10,
    )
    common = dict(
        predicted_representative_horizon=1.0, action_deviation=0.0,
        state_uncertainty=0.0, observation_age=0.0, compute_delay=0.0,
        communication_delay=0.0, actuation_delay=0.0,
    )
    assert bank.certified_duration(cell_id="c", **common) == 0.0
    assert bank.certified_duration(cell_id="unknown", **common) == 0.0


def test_cell_issue_labels_semantics_without_overclaiming_selection_coverage():
    bank = CellConditionalValidityBank.fit(
        cell_ids=["c"] * 20,
        predicted_representative_horizons=np.ones(20),
        realized_representative_horizons=np.ones(20),
        specs=[_spec("c")], alpha=0.1, minimum_cell_samples=10,
    )
    envelope = bank.issue(
        np.array([0.2]), issued_at=1.0, cell_id="c",
        predicted_representative_horizon=1.0, action_deviation=0.0,
        state_uncertainty=0.0, observation_age=0.0, compute_delay=0.0,
        communication_delay=0.0, actuation_delay=0.0,
    )
    assert envelope.constraint_state == "cell_marginal_not_selection_conditional"
