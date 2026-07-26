from agc_runtime_assurance.assumption_matched_comparison_v10 import (
    Compatibility,
    ProblemDeclaration,
    assess_methods,
)


def frozen_like_problem(**updates) -> ProblemDeclaration:
    values = {
        "bounded_outcome": True,
        "deterministic_missing_stratum": True,
        "target_includes_missing_stratum": True,
        "registered_interval_witness": True,
        "witness_validated": True,
        "exchangeable_calibration_sample": False,
        "accepted_smoothness_model": False,
    }
    values.update(updates)
    return ProblemDeclaration(**values)


def by_name(problem):
    return {item.method: item for item in assess_methods(problem)}


def test_worst_case_and_valid_circa_target_the_declared_full_population():
    methods = by_name(frozen_like_problem())
    assert (
        methods["Manski worst-case bounds"].compatibility
        is Compatibility.AVAILABLE_FOR_TARGET_ESTIMAND
    )
    assert (
        methods["CIRCA interval-witness bounds"].compatibility
        is Compatibility.AVAILABLE_FOR_TARGET_ESTIMAND
    )


def test_complete_case_is_not_misreported_as_same_estimand():
    methods = by_name(frozen_like_problem())
    assert (
        methods["complete-case mean"].compatibility
        is Compatibility.AVAILABLE_ONLY_FOR_DIFFERENT_ESTIMAND
    )


def test_ipw_aipw_refuse_structural_zero_positivity():
    methods = by_name(frozen_like_problem())
    assert (
        methods["IPW/AIPW"].compatibility
        is Compatibility.UNAVAILABLE_ASSUMPTION_FAILURE
    )
    assert "structural zero" in methods["IPW/AIPW"].refusal_or_limitation


def test_circa_refuses_when_registered_witness_fails():
    methods = by_name(frozen_like_problem(witness_validated=False))
    assert (
        methods["CIRCA interval-witness bounds"].compatibility
        is Compatibility.UNAVAILABLE_ASSUMPTION_FAILURE
    )
    assert "worst-case" in methods[
        "CIRCA interval-witness bounds"
    ].refusal_or_limitation


def test_conformal_and_smoothness_are_not_silently_assumed():
    methods = by_name(frozen_like_problem())
    assert (
        methods["conformal risk control"].compatibility
        is Compatibility.UNAVAILABLE_ASSUMPTION_FAILURE
    )
    assert (
        methods["smoothness/model-based extrapolation"].compatibility
        is Compatibility.UNAVAILABLE_ASSUMPTION_FAILURE
    )


def test_all_declared_methods_are_always_reported():
    names = [item.method for item in assess_methods(frozen_like_problem())]
    assert names == [
        "Manski worst-case bounds",
        "complete-case mean",
        "IPW/AIPW",
        "CIRCA interval-witness bounds",
        "conformal risk control",
        "smoothness/model-based extrapolation",
    ]
