import numpy as np

from agc_runtime_assurance.fallback_mpc import (
    LinearBoxFallbackSafeMPC,
    LinearBoxFallbackSafeMPCQP,
    LinearBoxFallbackTubeVerifier,
)


def _verifier(error_radius=0.0, disturbance_radius=0.0):
    return LinearBoxFallbackTubeVerifier(
        A=np.array([[1.0]]), B=np.array([[1.0]]), C=np.array([[1.0]]),
        K=np.array([[-0.5]]), state_lower=np.array([-2.0]),
        state_upper=np.array([2.0]), input_lower=np.array([-1.0]),
        input_upper=np.array([1.0]), disturbance_radius=np.array([disturbance_radius]),
        estimation_error_radius=np.array([error_radius]),
        recovery_lower=np.array([-0.2]), recovery_upper=np.array([0.2]),
    )


def test_fallback_tube_requires_shared_first_input_and_reaches_recovery_set():
    result = _verifier().verify(
        state_estimate=np.array([1.0]),
        fallback_inputs=np.array([[-0.5], [-0.25], [-0.125]]),
        nominal_first_input=np.array([-0.5]),
    )
    assert result.feasible
    assert result.reason == "fallback_tube_feasible"
    assert result.nominal_states[-1, 0] == 0.125


def test_fallback_tube_fails_when_nominal_and_fallback_first_inputs_differ():
    result = _verifier().verify(
        state_estimate=np.array([1.0]), fallback_inputs=np.array([[-0.5]]),
        nominal_first_input=np.array([-0.4]),
    )
    assert not result.feasible
    assert result.reason == "nominal_fallback_first_input_mismatch"


def test_uncertainty_tube_can_destroy_terminal_recoverability():
    result = _verifier(error_radius=0.15, disturbance_radius=0.05).verify(
        state_estimate=np.array([0.0]), fallback_inputs=np.array([[0.0]]),
        nominal_first_input=np.array([0.0]),
    )
    assert not result.feasible
    assert result.reason == "terminal_recovery_set_failed"


def test_error_radius_recursion_matches_box_specialization_of_equation_14():
    radii = _verifier(error_radius=0.1, disturbance_radius=0.05).error_radii(1)
    # |A+BKC| = 0.5; additive = 0.5*0.1 + 0.05 + 0.1 = 0.2.
    assert np.allclose(radii[:, 0], [0.0, 0.2, 0.3])


def test_online_fallback_mpc_modifies_nominal_plan_to_reach_recovery_set():
    solver = LinearBoxFallbackSafeMPC(_verifier())
    solution = solver.solve(
        state_estimate=np.array([1.0]),
        nominal_reference=np.zeros((3, 1)),
        emergency_action=np.array([-1.0]),
    )
    assert solution.feasible
    assert solution.tube_result is not None and solution.tube_result.feasible
    assert solution.action[0] < 0.0
    assert np.allclose(solution.action, solution.fallback_inputs[0])
    assert -0.2 - 1e-7 <= solution.tube_result.nominal_states[-1, 0] <= 0.2 + 1e-7


def test_online_fallback_mpc_fails_closed_when_recovery_set_is_unreachable():
    verifier = LinearBoxFallbackTubeVerifier(
        A=np.array([[1.0]]), B=np.array([[1.0]]), C=np.array([[1.0]]),
        K=np.array([[0.0]]), state_lower=np.array([-2.0]), state_upper=np.array([2.0]),
        input_lower=np.array([-0.1]), input_upper=np.array([0.1]),
        disturbance_radius=np.array([0.0]), estimation_error_radius=np.array([0.0]),
        recovery_lower=np.array([1.5]), recovery_upper=np.array([1.8]),
    )
    solution = LinearBoxFallbackSafeMPC(verifier).solve(
        state_estimate=np.array([0.0]), nominal_reference=np.zeros((1, 1)),
        emergency_action=np.array([-0.1]),
    )
    assert not solution.feasible
    assert np.allclose(solution.action, [-0.1])
    assert solution.solver_status.startswith("scipy_slsqp_failed")


def test_osqp_backend_matches_reference_feasibility_and_postchecks_solution():
    verifier = _verifier()
    reference = np.zeros((3, 1))
    kwargs = {
        "state_estimate": np.array([1.0]),
        "nominal_reference": reference,
        "emergency_action": np.array([-1.0]),
    }
    reference_solution = LinearBoxFallbackSafeMPC(verifier).solve(**kwargs)
    qp_solution = LinearBoxFallbackSafeMPCQP(verifier).solve(**kwargs)
    assert reference_solution.feasible and qp_solution.feasible
    assert qp_solution.solver_status == "cvxpy_osqp"
    assert qp_solution.tube_result is not None and qp_solution.tube_result.feasible
    assert np.allclose(qp_solution.action, qp_solution.fallback_inputs[0])
    assert np.allclose(qp_solution.action, reference_solution.action, atol=1e-5)


def test_osqp_backend_fails_closed_on_infeasible_recovery_problem():
    verifier = LinearBoxFallbackTubeVerifier(
        A=np.array([[1.0]]), B=np.array([[1.0]]), C=np.array([[1.0]]),
        K=np.array([[0.0]]), state_lower=np.array([-2.0]), state_upper=np.array([2.0]),
        input_lower=np.array([-0.1]), input_upper=np.array([0.1]),
        disturbance_radius=np.array([0.0]), estimation_error_radius=np.array([0.0]),
        recovery_lower=np.array([1.5]), recovery_upper=np.array([1.8]),
    )
    solution = LinearBoxFallbackSafeMPCQP(verifier).solve(
        state_estimate=np.array([0.0]), nominal_reference=np.zeros((1, 1)),
        emergency_action=np.array([-0.1]),
    )
    assert not solution.feasible
    assert np.allclose(solution.action, [-0.1])
    assert solution.solver_status == "cvxpy_osqp_status:infeasible"
