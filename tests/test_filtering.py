import numpy as np

from agc_runtime_assurance.filtering import AffineSafetyFilter, FilterStatus


def test_filter_minimally_projects_unsafe_action():
    safety_filter = AffineSafetyFilter(np.array([-2.0]), np.array([2.0]))
    result = safety_filter.filter(
        nominal=np.array([1.5]), A=np.array([[1.0]]), b=np.array([0.5]), fallback=np.array([0.0])
    )
    assert result.status == FilterStatus.FILTERED
    assert np.allclose(result.action, [0.5], atol=1e-4)
    assert result.backend in {"cvxpy_osqp", "scipy_slsqp"}


def test_infeasible_filter_uses_fallback():
    safety_filter = AffineSafetyFilter(np.array([-1.0]), np.array([1.0]))
    result = safety_filter.filter(
        nominal=np.array([0.0]), A=np.array([[1.0], [-1.0]]), b=np.array([-2.0, -2.0]), fallback=np.array([0.25])
    )
    assert result.status == FilterStatus.INFEASIBLE_FALLBACK
    assert np.allclose(result.action, [0.25])


def test_filter_postcondition_holds_for_coupled_constraints():
    safety_filter = AffineSafetyFilter(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    A = np.array([[1.0, 1.0], [-1.0, 0.0]])
    b = np.array([0.25, 0.75])
    result = safety_filter.filter(
        nominal=np.array([0.8, 0.8]), A=A, b=b, fallback=np.zeros(2)
    )
    assert result.status == FilterStatus.FILTERED
    assert np.all(A @ result.action <= b + 1e-7)
    assert np.all(result.action >= -1.0 - 1e-9)
    assert np.all(result.action <= 1.0 + 1e-9)
