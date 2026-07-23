"""Pre-registered seed-level analysis for authorized development results.

The analysis is descriptive and cannot generate paper claims.  Policy seed is
the inferential unit; episodes are used only to estimate each seed/family rate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import string
from typing import Any

import numpy as np

from .metrics import (
    FamilyCount,
    paired_seed_differences,
    worst_family_rate_per_seed,
    worst_family_upper_bound_per_seed,
)
from .risk import clopper_pearson_upper


class DevelopmentAnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class SeedFamilyResult:
    method: str
    policy_seed: int
    scenario_family: str
    episodes: int
    episodes_with_constraint_violation: int
    failed_runs_without_observed_violation: int
    task_successes: int
    mean_task_return: float
    deadline_event_records: int
    deadline_administrative_censors: int
    deadline_known_covered: int
    deadline_observed_noncoverages: int
    deadline_indeterminate: int
    deadline_intervention_truncations: int
    deadline_invalid_or_missing: int
    actions_total: int
    actions_intervened: int
    actions_rejected: int
    zero_validity_actions: int
    qp_attempts: int
    qp_feasible: int
    late_handover_events: int
    backup_infeasible_events: int
    audit_events_expected: int
    audit_events_complete: int
    solver_p50_ms: float
    solver_p95_ms: float
    solver_p99_ms: float

    @property
    def conservative_safety_failures(self) -> int:
        return (
            self.episodes_with_constraint_violation
            + self.failed_runs_without_observed_violation
        )

    def validate(self) -> None:
        if not self.method or not self.scenario_family:
            raise DevelopmentAnalysisError("method and scenario_family must be non-empty")
        if not isinstance(self.policy_seed, int) or self.episodes <= 0:
            raise DevelopmentAnalysisError("policy_seed must be integer and episodes positive")
        episode_counts = (
            self.episodes_with_constraint_violation,
            self.failed_runs_without_observed_violation,
            self.task_successes,
        )
        if any(not isinstance(value, int) or value < 0 for value in episode_counts):
            raise DevelopmentAnalysisError("episode counts must be non-negative integers")
        if self.conservative_safety_failures > self.episodes:
            raise DevelopmentAnalysisError("disjoint safety-failure counts exceed episodes")
        if self.task_successes > self.episodes:
            raise DevelopmentAnalysisError("task successes exceed episodes")
        if not math.isfinite(self.mean_task_return):
            raise DevelopmentAnalysisError("mean task return must be finite")

        deadline_counts = (
            self.deadline_event_records,
            self.deadline_administrative_censors,
            self.deadline_known_covered,
            self.deadline_observed_noncoverages,
            self.deadline_indeterminate,
            self.deadline_intervention_truncations,
            self.deadline_invalid_or_missing,
        )
        if any(not isinstance(value, int) or value < 0 for value in deadline_counts):
            raise DevelopmentAnalysisError("deadline counts must be non-negative integers")
        valid_deadline = self.deadline_event_records + self.deadline_administrative_censors
        classified_valid = (
            self.deadline_known_covered
            + self.deadline_observed_noncoverages
            + self.deadline_indeterminate
        )
        if classified_valid != valid_deadline:
            raise DevelopmentAnalysisError("valid deadline records are not exhaustively classified")

        action_counts = (
            self.actions_total, self.actions_intervened, self.actions_rejected,
            self.zero_validity_actions, self.qp_attempts, self.qp_feasible,
            self.late_handover_events, self.backup_infeasible_events,
            self.audit_events_expected, self.audit_events_complete,
        )
        if any(not isinstance(value, int) or value < 0 for value in action_counts):
            raise DevelopmentAnalysisError("guardrail counts must be non-negative integers")
        for label, value in (
            ("actions_intervened", self.actions_intervened),
            ("actions_rejected", self.actions_rejected),
            ("zero_validity_actions", self.zero_validity_actions),
            ("late_handover_events", self.late_handover_events),
            ("backup_infeasible_events", self.backup_infeasible_events),
        ):
            if value > self.actions_total:
                raise DevelopmentAnalysisError(f"{label} exceeds actions_total")
        if self.qp_feasible > self.qp_attempts:
            raise DevelopmentAnalysisError("qp_feasible exceeds qp_attempts")
        if self.audit_events_complete > self.audit_events_expected:
            raise DevelopmentAnalysisError("complete audit events exceed expected events")
        solver = np.asarray(
            [self.solver_p50_ms, self.solver_p95_ms, self.solver_p99_ms], dtype=float,
        )
        if not np.all(np.isfinite(solver)) or np.any(solver < 0.0):
            raise DevelopmentAnalysisError("solver quantiles must be finite and non-negative")
        if not self.solver_p50_ms <= self.solver_p95_ms <= self.solver_p99_ms:
            raise DevelopmentAnalysisError("solver quantiles must be monotone")


@dataclass(frozen=True)
class PairedSeedEffect:
    primary_method: str
    primary_baseline: str
    seed_differences: tuple[tuple[int, float], ...]
    mean_difference: float
    median_difference: float
    sample_standard_deviation: float
    bootstrap_confidence_interval: tuple[float, float]
    confidence_level: float
    sesoi_absolute_reduction: float
    interpretation: str


@dataclass(frozen=True)
class DeadlineAggregate:
    method: str
    valid_target_records: int
    known_covered: int
    observed_noncoverages: int
    indeterminate: int
    intervention_truncations: int
    invalid_or_missing: int
    best_case_noncoverage_rate: float | None
    worst_case_noncoverage_rate: float | None
    worst_case_upper_bound: float | None


@dataclass(frozen=True)
class GuardrailAggregate:
    method: str
    intervention_rate: float
    rejection_rate: float
    zero_validity_rate: float
    qp_feasibility_rate: float | None
    late_handover_rate: float
    backup_infeasible_rate: float
    audit_completeness_rate: float | None
    task_success_rate: float
    episode_weighted_mean_return: float
    worst_seed_family_solver_p99_ms: float


@dataclass(frozen=True)
class DevelopmentAnalysisReport:
    primary_effect: PairedSeedEffect
    worst_family_rates: dict[str, dict[int, float]]
    worst_family_upper_bounds: dict[str, dict[int, float]]
    secondary_mean_differences: dict[str, float]
    deadline_aggregates: tuple[DeadlineAggregate, ...]
    guardrail_aggregates: tuple[GuardrailAggregate, ...]
    result_manifest_hash: str
    verification_status: str
    claim_generation_allowed: bool


def analyze_development_results(path: str | Path) -> DevelopmentAnalysisReport:
    """Validate and analyze one explicitly authorized development result manifest."""
    raw = Path(path).read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    _verify_boundaries(manifest)
    methods = _unique_strings(manifest.get("methods"), "methods", minimum=2)
    seeds = _unique_ints(manifest.get("policy_seeds"), "policy_seeds", minimum=3)
    families = _unique_strings(
        manifest.get("scenario_families"), "scenario_families", minimum=2,
    )
    primary = manifest.get("primary_method")
    baseline = manifest.get("primary_baseline")
    if primary not in methods or baseline not in methods or primary == baseline:
        raise DevelopmentAnalysisError("primary method and baseline must be distinct declared methods")
    alpha = manifest.get("alpha")
    if not isinstance(alpha, (int, float)) or not 0.0 < alpha < 1.0:
        raise DevelopmentAnalysisError("alpha must lie strictly between 0 and 1")
    resamples = manifest.get("bootstrap_resamples")
    if not isinstance(resamples, int) or resamples < 1000:
        raise DevelopmentAnalysisError("bootstrap_resamples must be at least 1000")
    bootstrap_seed = manifest.get("bootstrap_seed")
    if not isinstance(bootstrap_seed, int):
        raise DevelopmentAnalysisError("bootstrap_seed must be an integer")
    sesoi = manifest.get("sesoi_absolute_reduction")
    if not isinstance(sesoi, (int, float)) or not 0.0 < sesoi < 1.0:
        raise DevelopmentAnalysisError("SESOI must lie strictly between 0 and 1")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise DevelopmentAnalysisError("provenance must be an object")
    for field in (
        "authorization_manifest_hash", "code_tree_hash", "calibration_split_hash",
        "development_split_hash", "scenario_manifest_hash", "baseline_matrix_hash",
    ):
        _require_digest(provenance.get(field), f"provenance {field}")

    records = manifest.get("rows")
    if not isinstance(records, list):
        raise DevelopmentAnalysisError("rows must be a list")
    rows = tuple(_parse_row(record) for record in records)
    _verify_complete_rectangle(rows, methods, seeds, families)

    safety_rates: dict[str, dict[int, float]] = {}
    safety_bounds: dict[str, dict[int, float]] = {}
    for method in methods:
        family_counts = [
            FamilyCount(
                row.policy_seed, row.scenario_family,
                row.conservative_safety_failures, row.episodes,
            )
            for row in rows if row.method == method
        ]
        safety_rates[method] = worst_family_rate_per_seed(family_counts)
        safety_bounds[method] = worst_family_upper_bound_per_seed(
            family_counts, delta=float(alpha),
        )

    differences = paired_seed_differences(safety_rates[primary], safety_rates[baseline])
    interval = _bootstrap_mean_interval(
        differences, alpha=float(alpha), resamples=resamples, seed=bootstrap_seed,
    )
    primary_effect = PairedSeedEffect(
        str(primary), str(baseline),
        tuple((seed, float(value)) for seed, value in zip(sorted(seeds), differences)),
        float(np.mean(differences)), float(np.median(differences)),
        float(np.std(differences, ddof=1)), interval, 1.0 - float(alpha),
        float(sesoi),
        "development_descriptive_only_not_confirmatory_not_a_paper_claim",
    )
    secondary = {
        method: float(np.mean(paired_seed_differences(
            safety_rates[primary], safety_rates[method],
        )))
        for method in methods if method not in {primary, baseline}
    }
    deadline = tuple(_deadline_aggregate(method, rows, float(alpha)) for method in methods)
    primary_deadline = next(item for item in deadline if item.method == primary)
    if primary_deadline.valid_target_records == 0:
        raise DevelopmentAnalysisError("primary method has no valid deadline records")
    guardrails = tuple(_guardrail_aggregate(method, rows) for method in methods)
    return DevelopmentAnalysisReport(
        primary_effect, safety_rates, safety_bounds, secondary,
        deadline, guardrails, hashlib.sha256(raw).hexdigest(),
        "ANALYZED_DEVELOPMENT_ONLY_NOT_REPRODUCED", False,
    )


def _verify_boundaries(manifest: dict[str, Any]) -> None:
    required = {
        "stage": "development_results",
        "development_authorized": True,
        "execution_exit_code": 0,
        "formal_experiment_run": False,
        "sealed_data_used": False,
        "claim_generation_allowed": False,
        "only_calibration_and_development_splits": True,
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise DevelopmentAnalysisError(f"boundary field {field} must equal {expected!r}")


def _parse_row(record: Any) -> SeedFamilyResult:
    if not isinstance(record, dict):
        raise DevelopmentAnalysisError("result rows must be objects")
    try:
        row = SeedFamilyResult(**record)
    except TypeError as error:
        raise DevelopmentAnalysisError(f"result row schema mismatch: {error}") from error
    row.validate()
    return row


def _verify_complete_rectangle(
    rows: tuple[SeedFamilyResult, ...],
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
    families: tuple[str, ...],
) -> None:
    expected = {(method, seed, family) for method in methods for seed in seeds for family in families}
    observed = [(row.method, row.policy_seed, row.scenario_family) for row in rows]
    if len(set(observed)) != len(observed):
        raise DevelopmentAnalysisError("duplicate method-seed-family result row")
    if set(observed) != expected:
        missing = len(expected - set(observed))
        extra = len(set(observed) - expected)
        raise DevelopmentAnalysisError(
            f"result rectangle is incomplete or contaminated: missing={missing}, extra={extra}"
        )


def _bootstrap_mean_interval(
    differences: np.ndarray, *, alpha: float, resamples: int, seed: int,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size < 3 or not np.all(np.isfinite(values)):
        raise DevelopmentAnalysisError("at least three finite seed differences are required")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(resamples, values.size))
    means = np.mean(values[indices], axis=1)
    lower, upper = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


def _deadline_aggregate(
    method: str, rows: tuple[SeedFamilyResult, ...], alpha: float,
) -> DeadlineAggregate:
    selected = [row for row in rows if row.method == method]
    valid = sum(row.deadline_event_records + row.deadline_administrative_censors for row in selected)
    covered = sum(row.deadline_known_covered for row in selected)
    observed = sum(row.deadline_observed_noncoverages for row in selected)
    indeterminate = sum(row.deadline_indeterminate for row in selected)
    intervention = sum(row.deadline_intervention_truncations for row in selected)
    invalid = sum(row.deadline_invalid_or_missing for row in selected)
    if valid == 0:
        return DeadlineAggregate(
            method, 0, covered, observed, indeterminate, intervention, invalid,
            None, None, None,
        )
    worst = observed + indeterminate
    return DeadlineAggregate(
        method, valid, covered, observed, indeterminate, intervention, invalid,
        observed / valid, worst / valid,
        clopper_pearson_upper(worst, valid, delta=alpha),
    )


def _guardrail_aggregate(
    method: str, rows: tuple[SeedFamilyResult, ...],
) -> GuardrailAggregate:
    selected = [row for row in rows if row.method == method]
    actions = sum(row.actions_total for row in selected)
    episodes = sum(row.episodes for row in selected)
    qp_attempts = sum(row.qp_attempts for row in selected)
    audits = sum(row.audit_events_expected for row in selected)
    if actions <= 0 or episodes <= 0:
        raise DevelopmentAnalysisError(f"method {method} has no actions or episodes")
    weighted_return = sum(row.mean_task_return * row.episodes for row in selected) / episodes
    return GuardrailAggregate(
        method,
        sum(row.actions_intervened for row in selected) / actions,
        sum(row.actions_rejected for row in selected) / actions,
        sum(row.zero_validity_actions for row in selected) / actions,
        (sum(row.qp_feasible for row in selected) / qp_attempts if qp_attempts else None),
        sum(row.late_handover_events for row in selected) / actions,
        sum(row.backup_infeasible_events for row in selected) / actions,
        (sum(row.audit_events_complete for row in selected) / audits if audits else None),
        sum(row.task_successes for row in selected) / episodes,
        float(weighted_return),
        max(row.solver_p99_ms for row in selected),
    )


def _unique_strings(value: Any, label: str, *, minimum: int) -> tuple[str, ...]:
    if (
        not isinstance(value, list) or len(value) < minimum
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise DevelopmentAnalysisError(f"{label} must contain at least {minimum} unique strings")
    return tuple(value)


def _unique_ints(value: Any, label: str, *, minimum: int) -> tuple[int, ...]:
    if (
        not isinstance(value, list) or len(value) < minimum
        or any(not isinstance(item, int) for item in value)
        or len(set(value)) != len(value)
    ):
        raise DevelopmentAnalysisError(f"{label} must contain at least {minimum} unique integers")
    return tuple(value)


def _require_digest(value: Any, label: str) -> None:
    if (
        not isinstance(value, str) or len(value) != 64
        or any(character not in string.hexdigits for character in value)
    ):
        raise DevelopmentAnalysisError(f"{label} must be a 64-character hexadecimal digest")
