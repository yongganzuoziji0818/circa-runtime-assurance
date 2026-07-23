import numpy as np
import pytest

from agc_runtime_assurance.filtering import FilterStatus
from agc_runtime_assurance.sandbox_baselines import (
    SandboxACoFiAdapter,
    SandboxBaselineInfeasible,
    SandboxConformalCBFAdapter,
    SandboxNominalCBFAdapter,
)
from agc_runtime_assurance.sandbox_task import SandboxComparisonTask


def _near_collision_toward_goals():
    state = np.zeros(15)
    state[:3] = [-0.505, 0.0, 2.0]
    state[6:8] = [0.505, 0.0]
    return state


def test_nominal_cbf_adapter_uses_shared_policy_and_contract():
    task = SandboxComparisonTask()
    state = _near_collision_toward_goals()
    fallback = np.array([-2.0, 0.0, 0.0, 2.0, 0.0])
    decision = SandboxNominalCBFAdapter(task).decide(
        state, fallback_action=fallback,
    )
    assert decision.filter_result.status == FilterStatus.FILTERED
    assert decision.nominal_policy_fingerprint == task.nominal_policy_fingerprint
    assert decision.constraint_contract_fingerprint == task.constraint_contract_fingerprint
    assert np.all(
        decision.constraint_bundle.A @ decision.action
        <= decision.constraint_bundle.b + 1e-7
    )
    assert task.postcheck_next_state(state, decision.action)


def test_conformal_cbf_adapter_translates_same_affine_contract():
    task = SandboxComparisonTask()
    state = _near_collision_toward_goals()
    fallback = np.array([-2.0, 0.0, 0.0, 2.0, 0.0])
    decision = SandboxConformalCBFAdapter(
        target_loss=-0.05, learning_rate=0.1, initial_value=0.0, task=task,
    ).decide(state, fallback_action=fallback)
    step = decision.conformal_step
    assert np.all(step.affine_A @ decision.action <= step.affine_b + 1e-7)
    assert np.array_equal(step.affine_A, decision.constraint_bundle.A)
    assert np.array_equal(step.affine_b, decision.constraint_bundle.b)
    assert task.postcheck_next_state(state, decision.action)


def test_negative_conformal_value_tightens_shared_constraints():
    task = SandboxComparisonTask()
    state = _near_collision_toward_goals()
    fallback = np.array([-2.0, 0.0, 0.0, 2.0, 0.0])
    nominal = SandboxNominalCBFAdapter(task).decide(state, fallback_action=fallback)
    conformal = SandboxConformalCBFAdapter(
        target_loss=-0.05, learning_rate=0.1, initial_value=-0.01, task=task,
    ).decide(state, fallback_action=fallback)
    assert conformal.conformal_step.filter_result.intervention_norm >= (
        nominal.filter_result.intervention_norm - 1e-7
    )
    assert np.all(
        conformal.constraint_bundle.A @ conformal.action
        <= conformal.constraint_bundle.b - 0.01 + 1e-7
    )


def test_adapter_refuses_to_execute_when_no_postchecked_action_exists():
    state = np.zeros(15)
    state[2] = 2.0
    with pytest.raises(SandboxBaselineInfeasible, match="postcheck"):
        SandboxNominalCBFAdapter().decide(
            state, fallback_action=np.zeros(5),
        )


def test_acofi_adapter_does_not_hide_unsafe_task_policy_selection():
    task = SandboxComparisonTask()
    state = _near_collision_toward_goals()
    fallback = np.array([-2.0, 0.0, 0.0, 2.0, 0.0])
    adapter = SandboxACoFiAdapter(
        target_alpha=0.1, learning_rate=0.05, gamma=0.9,
        safety_threshold=0.1, task=task,
    )
    decision = adapter.decide(
        state, step_index=0, predicted_task_q=1.0, fallback_action=fallback,
    )
    assert decision.acofi_decision.source == "task_policy"
    assert decision.exact_next_step_postcheck is False
    assert decision.safe_decision.filter_result.status == FilterStatus.FILTERED
    assert task.postcheck_next_state(state, decision.safe_decision.action)


def test_acofi_bad_delayed_feedback_switches_to_shared_safe_policy():
    task = SandboxComparisonTask()
    state = _near_collision_toward_goals()
    fallback = np.array([-2.0, 0.0, 0.0, 2.0, 0.0])
    adapter = SandboxACoFiAdapter(
        target_alpha=0.1, learning_rate=0.05, gamma=0.9,
        safety_threshold=0.1, task=task,
    )
    _, error = adapter.observe_transition(
        previous_predicted_q=1.0,
        previous_local_margin=0.01,
        next_learned_value=-1.0,
    )
    assert error
    decision = adapter.decide(
        state, step_index=0, predicted_task_q=100.0, fallback_action=fallback,
    )
    assert decision.acofi_decision.source == "learned_safe_policy"
    assert np.array_equal(decision.action, decision.safe_decision.action)
    assert decision.exact_next_step_postcheck is True
