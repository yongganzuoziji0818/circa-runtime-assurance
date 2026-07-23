from itertools import product

import numpy as np

from agc_runtime_assurance.backup_invariant import LinearFeedbackInvariantBoxVerifier
from agc_runtime_assurance.environment import AirGroundRuntimeEnv, CompoundShift
from agc_runtime_assurance.sandbox_model import (
    ShiftParameterBox,
    air_ground_augmented_matrices,
    sandbox_axis_aligned_state_constraints,
    sandbox_constraint_margin_lower_bound,
    sandbox_parameter_uncertainty_envelope,
)


def test_augmented_linear_model_exactly_matches_sandbox_step_with_lag():
    shift = CompoundShift(
        uav_mass=1.3, uav_drag=0.17, ugv_friction=0.21, actuator_lag=0.3
    )
    env = AirGroundRuntimeEnv(shift)
    env.reset(seed=5)
    env.state = np.linspace(-0.5, 0.5, 10)
    env._applied_action = np.linspace(-0.2, 0.2, 5)
    augmented = np.concatenate([env.state, env._applied_action])
    action = np.array([0.4, -0.3, 0.2, -0.1, 0.5])
    A, B = air_ground_augmented_matrices(shift)
    expected = A @ augmented + B @ action
    env.step(action)
    observed = np.concatenate([env.state, env._applied_action])
    assert np.allclose(observed, expected, atol=1e-12, rtol=0.0)


def test_parameter_box_envelope_contains_every_vertex_error():
    nominal = CompoundShift()
    bounds = ShiftParameterBox(
        uav_mass=(0.9, 1.1), uav_drag=(0.05, 0.15),
        ugv_friction=(0.05, 0.15), actuator_lag=(0.0, 0.2),
    )
    state_radius = np.linspace(0.05, 0.19, 15)
    input_radius = np.linspace(0.1, 0.5, 5)
    envelope = sandbox_parameter_uncertainty_envelope(
        nominal_shift=nominal, parameter_box=bounds,
        state_deviation_radius=state_radius, input_deviation_radius=input_radius,
    )
    A0, B0 = air_ground_augmented_matrices(nominal)
    for vertex, state_sign, input_sign in product(
        bounds.vertices(), (-1.0, 1.0), (-1.0, 1.0)
    ):
        A, B = air_ground_augmented_matrices(vertex)
        error = (A - A0) @ (state_sign * state_radius) + (B - B0) @ (
            input_sign * input_radius
        )
        assert np.all(np.abs(error) <= envelope.disturbance_radius + 1e-15)
    assert envelope.vertex_count == 16
    assert len(envelope.fingerprint) == 64


def _nominal_backup_gain():
    K = np.zeros((5, 15), dtype=float)
    for axis in range(3):
        K[axis, axis] = -5.0
        K[axis, 3 + axis] = -9.9
    for axis in range(2):
        K[3 + axis, 6 + axis] = -5.0
        K[3 + axis, 8 + axis] = -11.0
    return K


def _sandbox_equilibrium():
    return np.array(
        [-4.0, -2.0, 2.0, 0.0, 0.0, 0.0, 4.0, 2.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0], dtype=float,
    )


def _sandbox_invariant_radius():
    return np.array(
        [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05,
         1.0, 1.0, 1.0, 1.1, 1.1], dtype=float,
    )


def test_nominal_development_sandbox_has_a_mechanically_verified_backup_box():
    A, B = air_ground_augmented_matrices(CompoundShift())
    lower, upper = sandbox_axis_aligned_state_constraints()
    center = _sandbox_equilibrium()
    radius = _sandbox_invariant_radius()
    result = LinearFeedbackInvariantBoxVerifier(
        A=A, B=B, C=np.eye(15), K=_nominal_backup_gain(),
        equilibrium_state=center, equilibrium_input=np.zeros(5),
        invariant_radius=radius, disturbance_radius=np.zeros(15),
        state_lower=lower, state_upper=upper,
        input_lower=-2.0 * np.ones(5), input_upper=2.0 * np.ones(5),
    ).verify()
    assert result.verified
    assert np.min(result.invariant_slack) >= -1e-12


def test_invariant_box_has_positive_lower_bounds_for_all_sandbox_constraints():
    margins = sandbox_constraint_margin_lower_bound(
        center=_sandbox_equilibrium(), radius=_sandbox_invariant_radius(),
        step_index_upper=199,
    )
    assert np.all(margins.as_array() > 0.0)
