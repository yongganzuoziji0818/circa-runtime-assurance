"""Deterministic assumption audit for missing-outcome comparison methods."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class Compatibility(str, Enum):
    AVAILABLE_FOR_TARGET_ESTIMAND = "AVAILABLE_FOR_TARGET_ESTIMAND"
    AVAILABLE_ONLY_FOR_DIFFERENT_ESTIMAND = "AVAILABLE_ONLY_FOR_DIFFERENT_ESTIMAND"
    UNAVAILABLE_ASSUMPTION_FAILURE = "UNAVAILABLE_ASSUMPTION_FAILURE"
    AVAILABLE_IF_ADDITIONAL_ASSUMPTION_ACCEPTED = (
        "AVAILABLE_IF_ADDITIONAL_ASSUMPTION_ACCEPTED"
    )


@dataclass(frozen=True)
class ProblemDeclaration:
    bounded_outcome: bool
    deterministic_missing_stratum: bool
    target_includes_missing_stratum: bool
    registered_interval_witness: bool
    witness_validated: bool
    exchangeable_calibration_sample: bool
    accepted_smoothness_model: bool


@dataclass(frozen=True)
class MethodAssessment:
    method: str
    compatibility: Compatibility
    target_or_output: str
    decisive_assumption: str
    refusal_or_limitation: str

    def as_dict(self) -> dict:
        result = asdict(self)
        result["compatibility"] = self.compatibility.value
        return result


def assess_methods(problem: ProblemDeclaration) -> list[MethodAssessment]:
    """Assess named alternatives without ranking by favorable output."""

    results: list[MethodAssessment] = []

    if problem.bounded_outcome:
        results.append(
            MethodAssessment(
                "Manski worst-case bounds",
                Compatibility.AVAILABLE_FOR_TARGET_ESTIMAND,
                "partial-identification interval for the full bounded-outcome target",
                "known finite outcome bounds",
                "valid but can remain wide when the missing mass is large",
            )
        )
    else:
        results.append(
            MethodAssessment(
                "Manski worst-case bounds",
                Compatibility.UNAVAILABLE_ASSUMPTION_FAILURE,
                "no finite worst-case interval",
                "known finite outcome bounds",
                "refuse finite endpoint claim",
            )
        )

    if problem.target_includes_missing_stratum:
        results.append(
            MethodAssessment(
                "complete-case mean",
                Compatibility.AVAILABLE_ONLY_FOR_DIFFERENT_ESTIMAND,
                "mean among observed outcomes only",
                "none for the observed-only estimand",
                "does not identify the declared full-population target",
            )
        )
    else:
        results.append(
            MethodAssessment(
                "complete-case mean",
                Compatibility.AVAILABLE_FOR_TARGET_ESTIMAND,
                "mean among observed outcomes",
                "target is explicitly restricted to observed outcomes",
                "does not generalize to missing strata",
            )
        )

    if problem.deterministic_missing_stratum:
        weighting_status = Compatibility.UNAVAILABLE_ASSUMPTION_FAILURE
        weighting_limit = "structural zero observation probability violates positivity"
    else:
        weighting_status = (
            Compatibility.AVAILABLE_IF_ADDITIONAL_ASSUMPTION_ACCEPTED
        )
        weighting_limit = "requires correct observation and/or outcome models"
    results.append(
        MethodAssessment(
            "IPW/AIPW",
            weighting_status,
            "full-population mean under missing-at-random style identification",
            "positivity plus appropriate exchangeability/model conditions",
            weighting_limit,
        )
    )

    if (
        problem.bounded_outcome
        and problem.registered_interval_witness
        and problem.witness_validated
    ):
        circa_status = Compatibility.AVAILABLE_FOR_TARGET_ESTIMAND
        circa_limit = "inherits the declared interval-witness validity scope"
    else:
        circa_status = Compatibility.UNAVAILABLE_ASSUMPTION_FAILURE
        circa_limit = "falls back to worst-case bounds or typed refusal"
    results.append(
        MethodAssessment(
            "CIRCA interval-witness bounds",
            circa_status,
            "partial-identification interval for the full bounded-outcome target",
            "registered, validated interval witnesses for missing groups",
            circa_limit,
        )
    )

    if problem.exchangeable_calibration_sample:
        conformal_status = Compatibility.AVAILABLE_IF_ADDITIONAL_ASSUMPTION_ACCEPTED
        conformal_limit = (
            "controls a calibrated monotone risk; it does not by itself identify "
            "systematically unobserved outcomes"
        )
    else:
        conformal_status = Compatibility.UNAVAILABLE_ASSUMPTION_FAILURE
        conformal_limit = "no exchangeable calibration support for the missing stratum"
    results.append(
        MethodAssessment(
            "conformal risk control",
            conformal_status,
            "calibrated risk control for a declared loss family",
            "exchangeable calibration data and monotone risk construction",
            conformal_limit,
        )
    )

    if problem.accepted_smoothness_model:
        smooth_status = Compatibility.AVAILABLE_IF_ADDITIONAL_ASSUMPTION_ACCEPTED
        smooth_limit = "sensitivity depends on the chosen structural smoothness class"
    else:
        smooth_status = Compatibility.UNAVAILABLE_ASSUMPTION_FAILURE
        smooth_limit = "no registered smoothness bridge across the unobserved region"
    results.append(
        MethodAssessment(
            "smoothness/model-based extrapolation",
            smooth_status,
            "model-dependent point or interval extrapolation",
            "accepted structural model across observed and missing regions",
            smooth_limit,
        )
    )

    return results
