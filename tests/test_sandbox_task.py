import numpy as np

from agc_runtime_assurance.environment import AirGroundRuntimeEnv
from agc_runtime_assurance.filtering import AffineSafetyFilter, FilterStatus
from agc_runtime_assurance.sandbox_task import SandboxComparisonTask


def _initial_augmented(seed=7):
    env = AirGroundRuntimeEnv()
    env.reset(seed=seed)
    return np.concatenate([env.state, env._applied_action])


def test_shared_nominal_policy_is_deterministic_and_fingerprinted():
    task = SandboxComparisonTask()
    state = _initial_augmented()
    first = task.nominal_action(state)
    second = task.nominal_action(state.copy())
    assert np.array_equal(first, second)
    assert first.shape == (5,)
    assert np.all(first >= task.action_lower)
    assert np.all(first <= task.action_upper)
    assert len(task.nominal_policy_fingerprint) == 64
    assert len(task.constraint_contract_fingerprint) == 64


def test_point_margins_match_environment_contract():
    env = AirGroundRuntimeEnv()
    env.reset(seed=9)
    augmented = np.concatenate([env.state, env._applied_action])
    observed = SandboxComparisonTask().point_constraint_margins(
        augmented, step_index=env.step_index,
    )
    assert np.allclose(observed.as_array(), env.constraint_margins().as_array())


def test_policy_fingerprint_changes_when_policy_gain_changes():
    default = SandboxComparisonTask()
    changed = SandboxComparisonTask(uav_position_gain=0.9)
    assert default.nominal_policy_fingerprint != changed.nominal_policy_fingerprint
    assert default.constraint_contract_fingerprint == changed.constraint_contract_fingerprint


def test_constraint_fingerprint_changes_when_safety_distance_changes():
    default = SandboxComparisonTask()
    changed = SandboxComparisonTask(minimum_separation=1.2)
    assert default.constraint_contract_fingerprint != changed.constraint_contract_fingerprint


def test_shared_affine_contract_filters_a_near_collision_action():
    task = SandboxComparisonTask()
    state = np.zeros(15)
    state[:3] = [0.505, 0.0, 2.0]
    state[6:8] = [-0.505, 0.0]
    toward = np.array([-2.0, 0.0, 0.0, 2.0, 0.0])
    away = -toward
    constraints = task.next_state_constraints(
        state, separation_reference_action=toward,
    )
    assert np.any(constraints.A @ toward > constraints.b + 1e-9)
    result = AffineSafetyFilter(task.action_lower, task.action_upper).filter(
        toward, constraints.A, constraints.b, away,
    )
    assert result.status == FilterStatus.FILTERED
    assert np.all(constraints.A @ result.action <= constraints.b + 1e-7)
    assert task.postcheck_next_state(state, result.action)


def test_separation_tangent_is_a_sufficient_next_step_condition():
    task = SandboxComparisonTask()
    state = np.zeros(15)
    state[:3] = [0.505, 0.0, 2.0]
    state[6:8] = [-0.505, 0.0]
    reference = np.array([-2.0, 0.0, 0.0, 2.0, 0.0])
    constraints = task.next_state_constraints(
        state, separation_reference_action=reference,
    )
    generator = np.random.default_rng(12)
    feasible_count = 0
    for action in generator.uniform(-2.0, 2.0, size=(500, 5)):
        if np.all(constraints.A @ action <= constraints.b + 1e-10):
            feasible_count += 1
            assert task.postcheck_next_state(state, action)
    assert feasible_count > 0
