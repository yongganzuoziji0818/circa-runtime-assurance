import pytest

from agc_runtime_assurance.transversality import (
    ConstraintPassageEvidence,
    transport_team_horizon,
)


def crossing(
    *,
    constraint_id: str = "coupling",
    kappa: float = 1.0,
    tube: float = 0.2,
    pre_margin: float = 0.2,
) -> ConstraintPassageEvidence:
    return ConstraintPassageEvidence(
        constraint_id=constraint_id,
        reference_horizon=1.0,
        crossing_observed=True,
        constraint_lipschitz=1.0,
        transversality_kappa=kappa,
        crossing_tube_width=tube,
        pre_tube_minimum_margin=pre_margin,
    )


def censored(*, minimum_margin: float = 0.2) -> ConstraintPassageEvidence:
    return ConstraintPassageEvidence(
        constraint_id="individual",
        reference_horizon=1.5,
        crossing_observed=False,
        constraint_lipschitz=1.0,
        censored_minimum_margin=minimum_margin,
    )


def test_team_transport_uses_earliest_valid_constraint() -> None:
    result = transport_team_horizon([crossing(), censored()], 0.1)

    assert result.valid
    assert result.transported_team_horizon == pytest.approx(0.9)
    assert result.critical_constraint == "coupling"


def test_crossing_tube_must_cover_time_debit() -> None:
    result = transport_team_horizon([crossing(tube=0.05)], 0.1)

    assert not result.valid
    assert result.transported_team_horizon == 0.0
    assert "transversality_tube_too_narrow" in result.reason


def test_pre_crossing_margin_must_be_robust() -> None:
    result = transport_team_horizon([crossing(pre_margin=0.05)], 0.1)

    assert not result.valid
    assert result.transported_team_horizon == 0.0
    assert "pre_crossing_tube_margin_not_robust" in result.reason


def test_censored_horizon_requires_strict_margin() -> None:
    result = transport_team_horizon([censored(minimum_margin=0.1)], 0.1)

    assert not result.valid
    assert result.transported_team_horizon == 0.0
    assert "censored_horizon_margin_not_robust" in result.reason


def test_simultaneous_crossings_do_not_require_unique_active_constraint() -> None:
    result = transport_team_horizon(
        [
            crossing(constraint_id="pairwise", kappa=1.0),
            crossing(constraint_id="coupling", kappa=2.0),
        ],
        0.1,
    )

    assert result.valid
    assert result.transported_team_horizon == pytest.approx(0.9)
    assert result.critical_constraint == "pairwise"
