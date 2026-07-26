import numpy as np
import pytest

from agc_runtime_assurance.bounded_loss_circa import (
    BoundedLossCircaError,
    IntervalWitnessGroup,
    bounded_loss_identification_interval,
    interval_witness_bounds,
    manski_bounded_loss_bounds,
    robust_interval_from_groups,
)


def test_general_bounded_loss_bounds_and_identification():
    r = np.array([0, 1, 1])
    z1 = np.array([0.25, 0.50, 1.25])
    lower, upper = manski_bounded_loss_bounds(r, z1, upper_bound=2.0)
    assert np.allclose(lower, [0.25, 0.0, 0.0])
    assert np.allclose(upper, [0.25, 2.0, 2.0])
    interval = bounded_loss_identification_interval(
        lower, upper, z1, upper_bound=2.0
    )
    assert interval == pytest.approx((-1.75 / 3.0, 2.25 / 3.0))


def test_interval_witnesses_tighten_only_registered_intervened_units():
    r = [0, 1, 1, 1]
    z1 = [0.2, 0.3, 0.4, 0.5]
    witnessed = [0, 1, 1, 0]
    witness_lower = [np.nan, 1.2, 0.0, np.nan]
    witness_upper = [np.nan, 1.7, 0.4, np.nan]
    lower, upper = interval_witness_bounds(
        r,
        z1,
        witnessed,
        witness_lower,
        witness_upper,
        upper_bound=2.0,
    )
    assert np.allclose(lower, [0.2, 1.2, 0.0, 0.0])
    assert np.allclose(upper, [0.2, 1.7, 0.4, 2.0])


def test_unresolved_unit_cannot_smuggle_a_numeric_witness():
    with pytest.raises(
        BoundedLossCircaError, match="NaN witness endpoints"
    ):
        interval_witness_bounds(
            [1], [0.0], [0], [0.0], [1.0], upper_bound=1.0
        )


def test_robust_width_decomposition_and_nesting():
    groups = [
        IntervalWitnessGroup(
            "high-loss",
            mass=0.20,
            lower=1.2,
            upper=1.6,
            lower_error_mass=0.02,
            upper_error_mass=0.01,
        ),
        IntervalWitnessGroup(
            "low-loss",
            mass=0.10,
            lower=0.1,
            upper=0.5,
            lower_error_mass=0.00,
            upper_error_mass=0.03,
        ),
    ]
    result = robust_interval_from_groups(
        -0.4,
        0.6,
        upper_bound=2.0,
        intervention_mass=0.5,
        groups=groups,
    )
    expected_lower = -0.4 + 0.18 * 1.2 + 0.10 * 0.1
    expected_upper = 0.6 - 0.19 * 0.4 - 0.07 * 1.5
    assert result.lower == pytest.approx(expected_lower)
    assert result.upper == pytest.approx(expected_upper)
    assert result.width == pytest.approx(expected_upper - expected_lower)
    assert result.unresolved_mass == pytest.approx(0.2)
    assert result.general_lower <= result.lower <= result.upper <= result.general_upper


def test_binary_proposition_three_is_recovered_exactly():
    q_iv, q_ns, q_u = 0.30, 0.20, 0.10
    b_iv, b_ns = 0.04, 0.03
    result = robust_interval_from_groups(
        general_lower=-0.25,
        general_upper=0.35,
        upper_bound=1.0,
        intervention_mass=q_iv + q_ns + q_u,
        groups=[
            IntervalWitnessGroup(
                "IV", q_iv, 1.0, 1.0, lower_error_mass=b_iv
            ),
            IntervalWitnessGroup(
                "NS", q_ns, 0.0, 0.0, upper_error_mass=b_ns
            ),
        ],
    )
    assert result.lower == pytest.approx(-0.25 + q_iv - b_iv)
    assert result.upper == pytest.approx(0.35 - q_ns + b_ns)
    assert result.width == pytest.approx(q_u + b_iv + b_ns)


def test_zero_error_and_exhausted_error_limits():
    exact = robust_interval_from_groups(
        0.0,
        1.0,
        upper_bound=2.0,
        intervention_mass=0.5,
        groups=[IntervalWitnessGroup("W", 0.5, 0.4, 1.4)],
    )
    assert exact.width == pytest.approx(0.5)
    exhausted = robust_interval_from_groups(
        0.0,
        1.0,
        upper_bound=2.0,
        intervention_mass=0.5,
        groups=[
            IntervalWitnessGroup(
                "W",
                0.5,
                0.4,
                1.4,
                lower_error_mass=0.5,
                upper_error_mass=0.5,
            )
        ],
    )
    assert exhausted.lower == pytest.approx(0.0)
    assert exhausted.upper == pytest.approx(1.0)
    assert exhausted.width == pytest.approx(1.0)


def test_general_width_identity_is_fail_closed():
    with pytest.raises(BoundedLossCircaError, match="general interval width"):
        robust_interval_from_groups(
            0.0,
            0.9,
            upper_bound=2.0,
            intervention_mass=0.5,
            groups=[],
        )


def test_invalid_directional_budget_is_rejected():
    with pytest.raises(BoundedLossCircaError, match="invalid witness group"):
        robust_interval_from_groups(
            0.0,
            1.0,
            upper_bound=2.0,
            intervention_mass=0.5,
            groups=[
                IntervalWitnessGroup(
                    "W", 0.5, 0.4, 1.4, lower_error_mass=0.6
                )
            ],
        )
