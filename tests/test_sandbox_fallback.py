import numpy as np

from agc_runtime_assurance.sandbox_fallback import (
    SandboxFallbackSafeMPCAdapter,
    sandbox_backup_equilibrium,
    sandbox_backup_invariant_radius,
)


def test_sandbox_fallback_solution_is_bound_to_same_verified_invariant():
    adapter = SandboxFallbackSafeMPCAdapter()
    state = sandbox_backup_equilibrium()
    decision = adapter.decide(state, horizon=1)
    assert decision.feasible
    assert decision.bound_result is not None
    assert decision.bound_result.feasible
    assert decision.backup_invariant_fingerprint == (
        decision.bound_result.backup_invariant_fingerprint
    )
    assert np.array_equal(decision.action, decision.solution.fallback_inputs[0])
    assert adapter.task.postcheck_next_state(state, decision.action)


def test_sandbox_fallback_accepts_perturbation_inside_backup_box():
    adapter = SandboxFallbackSafeMPCAdapter()
    state = sandbox_backup_equilibrium()
    state = state + 0.1 * sandbox_backup_invariant_radius()
    decision = adapter.decide(state, horizon=1)
    assert decision.feasible
    assert decision.solution.tube_result is not None
    assert decision.solution.tube_result.minimum_slack >= -1e-9


def test_sandbox_fallback_reports_infeasible_far_from_recovery_box():
    adapter = SandboxFallbackSafeMPCAdapter()
    state = np.zeros(15)
    state[2] = 2.0
    decision = adapter.decide(state, horizon=1)
    assert decision.feasible is False
    assert decision.bound_result is None
    assert decision.solution.solver_status.startswith("scipy_slsqp_failed:")


def test_sandbox_fallback_fingerprints_shared_task_contracts():
    adapter = SandboxFallbackSafeMPCAdapter()
    decision = adapter.decide(sandbox_backup_equilibrium(), horizon=0)
    assert decision.nominal_policy_fingerprint == adapter.task.nominal_policy_fingerprint
    assert decision.constraint_contract_fingerprint == (
        adapter.task.constraint_contract_fingerprint
    )
    assert len(decision.backup_invariant_fingerprint) == 64
