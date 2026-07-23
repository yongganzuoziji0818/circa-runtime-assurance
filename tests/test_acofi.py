import numpy as np
import pytest

from agc_runtime_assurance.acofi import ACoFiRuntimeAdapter


def _adapter():
    return ACoFiRuntimeAdapter(
        target_alpha=0.1, learning_rate=0.05, gamma=0.9,
        safety_threshold=0.2,
    )


def test_acofi_initial_decision_uses_paper_switching_threshold():
    adapter = _adapter()
    decision = adapter.decide(
        predicted_task_q=0.31, current_local_margin=1.0,
        task_action=np.array([0.8]), safe_action=np.array([-0.2]),
    )
    # q1=0, gamma*epsilon=0.18, (1-gamma)*l=0.1.
    assert decision.switching_threshold == 0.28
    assert decision.source == "task_policy"
    assert np.allclose(decision.action, [0.8])


def test_bad_delayed_feedback_forces_safe_policy_with_small_history():
    adapter = _adapter()
    target, error = adapter.observe_transition(
        previous_predicted_q=1.0, previous_local_margin=0.5,
        next_learned_value=0.0,
    )
    assert target == pytest.approx(0.05)
    assert error
    decision = adapter.decide(
        predicted_task_q=100.0, current_local_margin=0.5,
        task_action=np.array([0.8]), safe_action=np.array([-0.2]),
    )
    assert decision.source == "learned_safe_policy"
    assert np.allclose(decision.action, [-0.2])
    assert adapter.empirical_error_rate == 1.0


def test_acofi_decision_returns_action_copy():
    adapter = _adapter()
    task = np.array([0.8])
    decision = adapter.decide(
        predicted_task_q=1.0, current_local_margin=0.0,
        task_action=task, safe_action=np.array([-0.2]),
    )
    task[0] = 99.0
    assert decision.action[0] == 0.8
