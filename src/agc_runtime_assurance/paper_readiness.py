"""Fail-closed evidence audit for P4 paper-package readiness.

The auditor reads only declared unsealed artifacts. It never authorizes or runs an
experiment, and a missing/provisional item can never be promoted by inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import string
from typing import Any

from .baseline_evidence import (
    BaselineEvidenceError,
    verify_baseline_reproduction_manifest,
)
from .development_analysis import (
    DevelopmentAnalysisError,
    analyze_development_results,
)
from .power_planning import PowerPlanningError, plan_seed_power_from_manifest


class ReadinessTier(str, Enum):
    METHOD_PROTOCOL = "method_protocol"
    DEVELOPMENT_RESULTS_DRAFT = "development_results_draft"
    FORMAL_RESULTS = "formal_results"


@dataclass(frozen=True)
class ReadinessFinding:
    requirement_id: str
    passed: bool
    status: str
    detail: str


@dataclass(frozen=True)
class PaperReadinessReport:
    target_tier: ReadinessTier
    target_ready: bool
    highest_ready_tier: ReadinessTier | None
    findings: tuple[ReadinessFinding, ...]
    manifest_hash: str


_METHOD_REQUIREMENTS = (
    "focused_literature_audit",
    "scientific_question_freeze",
    "assurance_contract",
    "local_g0",
    "claim_evidence_map",
)
_DEVELOPMENT_REQUIREMENTS = _METHOD_REQUIREMENTS + (
    "remote_g0",
    "systems_assurance_benchmark_evidence",
    "development_authorization",
    "development_results",
    "statistical_analysis",
    "dynamics_and_invariant_evidence",
    "handover_latency_evidence",
    "formal_experiment_plan",
)
_FORMAL_REQUIREMENTS = _DEVELOPMENT_REQUIREMENTS + (
    "formal_authorization",
    "formal_results",
)
_REQUIREMENTS_BY_TIER = {
    ReadinessTier.METHOD_PROTOCOL: _METHOD_REQUIREMENTS,
    ReadinessTier.DEVELOPMENT_RESULTS_DRAFT: _DEVELOPMENT_REQUIREMENTS,
    ReadinessTier.FORMAL_RESULTS: _FORMAL_REQUIREMENTS,
}
_STATISTICAL_OUTPUTS = {
    "worst_family_rate_per_seed",
    "worst_family_upper_bound_per_seed",
    "paired_effect_estimate",
    "paired_confidence_interval",
    "deadline_coverage",
    "censoring_breakdown",
    "guardrail_metrics",
}


def audit_paper_readiness(
    manifest_path: str | Path, repo_root: str | Path,
) -> PaperReadinessReport:
    """Audit a declarative evidence manifest without touching sealed data."""
    manifest_file = Path(manifest_path)
    raw = manifest_file.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    root = Path(repo_root).resolve()
    try:
        target = ReadinessTier(manifest.get("target_tier"))
    except ValueError as error:
        raise ValueError("unknown paper-readiness target_tier") from error

    requirements = manifest.get("requirements")
    if not isinstance(requirements, dict):
        raise ValueError("requirements must be an object")
    findings: list[ReadinessFinding] = []
    for requirement_id in _REQUIREMENTS_BY_TIER[target]:
        record = requirements.get(requirement_id)
        findings.append(_audit_requirement(requirement_id, record, root))

    boundary_errors = []
    if manifest.get("sealed_data_opened") is not False:
        boundary_errors.append("sealed_data_opened must be false")
    if target != ReadinessTier.FORMAL_RESULTS and manifest.get("formal_experiment_run") is not False:
        boundary_errors.append("formal_experiment_run must be false before formal-results tier")
    if boundary_errors:
        findings.append(ReadinessFinding(
            "authorization_boundary", False, "failed", "; ".join(boundary_errors),
        ))

    finding_map = {finding.requirement_id: finding for finding in findings}
    tier_ready = {
        tier: all(
            finding_map.get(req, ReadinessFinding(req, False, "missing", "missing")).passed
            for req in required
        ) and not boundary_errors
        for tier, required in _REQUIREMENTS_BY_TIER.items()
        if _tier_rank(tier) <= _tier_rank(target)
    }
    highest = None
    for tier in (
        ReadinessTier.METHOD_PROTOCOL,
        ReadinessTier.DEVELOPMENT_RESULTS_DRAFT,
        ReadinessTier.FORMAL_RESULTS,
    ):
        if tier_ready.get(tier):
            highest = tier
    return PaperReadinessReport(
        target, tier_ready.get(target, False), highest, tuple(findings),
        hashlib.sha256(raw).hexdigest(),
    )


def _audit_requirement(
    requirement_id: str, record: Any, root: Path,
) -> ReadinessFinding:
    if not isinstance(record, dict):
        return ReadinessFinding(requirement_id, False, "missing", "requirement record is missing")
    status = str(record.get("status", "missing"))
    if status != "verified":
        return ReadinessFinding(requirement_id, False, status, "status is not verified")
    artifact_error = _verify_artifacts(record.get("artifacts"), root)
    if artifact_error:
        return ReadinessFinding(requirement_id, False, "failed", artifact_error)
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        return ReadinessFinding(requirement_id, False, "failed", "metadata must be an object")
    special_error = _verify_special_requirement(requirement_id, metadata, root)
    if special_error:
        return ReadinessFinding(requirement_id, False, "failed", special_error)
    return ReadinessFinding(requirement_id, True, "verified", "declared artifacts and metadata verified")


def _verify_artifacts(artifacts: Any, root: Path) -> str | None:
    if not isinstance(artifacts, list) or not artifacts:
        return "at least one hashed artifact is required"
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            return "artifact entries must be objects"
        relative = artifact.get("path")
        digest = artifact.get("sha256")
        if not isinstance(relative, str) or not relative:
            return "artifact path is missing"
        if not _is_digest(digest):
            return f"artifact {relative} has an invalid SHA256 digest"
        candidate = (root / relative).resolve()
        if candidate != root and root not in candidate.parents:
            return f"artifact path escapes repo root: {relative}"
        if not candidate.is_file():
            return f"artifact does not exist: {relative}"
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != digest.lower():
            return f"artifact digest mismatch: {relative}"
    return None


def _verify_special_requirement(
    requirement_id: str, metadata: dict[str, Any], root: Path,
) -> str | None:
    if requirement_id in {"local_g0", "remote_g0"}:
        if not _is_digest(metadata.get("repo_tree_hash")):
            return "G0 repo_tree_hash is missing or invalid"
        passed, total = metadata.get("passed"), metadata.get("total")
        if not isinstance(passed, int) or not isinstance(total, int) or total <= 0 or passed != total:
            return "G0 passed and total counts must be equal positive integers"
    elif requirement_id == "strong_baseline_reproduction":
        relative = metadata.get("reproduction_manifest_path")
        expected_hash = metadata.get("reproduction_manifest_hash")
        if not isinstance(relative, str) or not relative:
            return "strong-baseline reproduction manifest path is missing"
        if not _is_digest(expected_hash):
            return "strong-baseline reproduction manifest hash is invalid"
        candidate = (root / relative).resolve()
        if candidate != root and root not in candidate.parents:
            return "strong-baseline reproduction manifest escapes repo root"
        if not candidate.is_file():
            return "strong-baseline reproduction manifest is missing"
        try:
            report = verify_baseline_reproduction_manifest(candidate, root)
        except (BaselineEvidenceError, ValueError, json.JSONDecodeError) as error:
            return f"strong-baseline evidence is invalid: {error}"
        if report.manifest_hash != expected_hash.lower():
            return "strong-baseline reproduction manifest digest mismatch"
        if not report.ready:
            failed = ",".join(f.baseline for f in report.findings if not f.passed)
            return f"strong-baseline task-level evidence is incomplete: {failed}"
    elif requirement_id == "systems_assurance_benchmark_evidence":
        result_relative = metadata.get("benchmark_result_path")
        result_hash = metadata.get("benchmark_result_hash")
        manifest_relative = metadata.get("benchmark_manifest_path")
        manifest_hash = metadata.get("benchmark_manifest_hash")
        route_relative = metadata.get("route_decision_path")
        route_hash = metadata.get("route_decision_hash")
        files = (
            (result_relative, result_hash, "benchmark result"),
            (manifest_relative, manifest_hash, "benchmark manifest"),
            (route_relative, route_hash, "route decision"),
        )
        loaded: dict[str, dict[str, Any]] = {}
        for relative, expected, label in files:
            if not isinstance(relative, str) or not relative:
                return f"{label} path is missing"
            if not _is_digest(expected):
                return f"{label} hash is invalid"
            candidate = (root / relative).resolve()
            if candidate != root and root not in candidate.parents:
                return f"{label} path escapes repo root"
            if not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
                return f"{label} hash mismatch"
            try:
                loaded[label] = json.loads(candidate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                return f"{label} is invalid JSON: {error}"
        route = loaded["route decision"]
        if route.get("status") != "option_A_confirmed":
            return "systems-assurance route A is not confirmed"
        if route.get("options", {}).get("A", {}).get("selected") is not True:
            return "systems-assurance route A selection is missing"
        manifest = loaded["benchmark manifest"]
        if manifest.get("experiment_family") != "assurance_case_corruption_benchmark":
            return "benchmark manifest experiment family is invalid"
        if manifest.get("sealed_data_allowed") is not False or manifest.get("formal_experiment") is not False:
            return "benchmark manifest crosses sealed/formal boundary"
        result = loaded["benchmark result"]
        if result.get("manifest_sha256") != manifest_hash:
            return "benchmark result is not bound to the declared manifest"
        if result.get("sealed_data_used") is not False or result.get("formal_experiment_run") is not False:
            return "benchmark result crosses sealed/formal boundary"
        if result.get("claim_generation_allowed") is not False:
            return "benchmark result improperly allows claim generation"
        if result.get("valid_control_count") != 4 or result.get("fault_case_count") != 24:
            return "benchmark result case counts do not match the frozen design"
        expected_families = {
            "missing_required_field", "malformed_digest", "expired_action",
            "monotonic_time_reversal", "latency_fingerprint_mismatch",
            "backup_invariant_mismatch", "constraint_contract_mismatch",
            "audit_chain_tamper",
        }
        if set(result.get("families", {})) != expected_families:
            return "benchmark result fault families do not match the frozen design"
        summary = result.get("summary", {})
        expected_summary = {
            "valid_bundle_acceptance_rate": 1.0,
            "pre_execution_block_rate": 1.0,
            "reason_localization_accuracy": 1.0,
            "worst_family_undetected_fault_rate": 0.0,
            "all_success_gates_passed": True,
        }
        if summary != expected_summary:
            return "benchmark result does not pass the frozen success gates"
    elif requirement_id == "development_authorization":
        if metadata.get("authorized") is not True:
            return "development execution was not explicitly authorized"
        if set(metadata.get("allowed_splits", [])) != {"calibration", "development"}:
            return "development authorization must expose only calibration and development"
        if metadata.get("sealed_data_authorized") is not False:
            return "sealed-data authority must remain false"
    elif requirement_id == "development_results":
        if metadata.get("exit_code") != 0 or metadata.get("only_development_splits") is not True:
            return "development run is incomplete or split provenance is invalid"
        if metadata.get("claim_generation_allowed") is not False:
            return "development execution must not directly authorize paper claims"
    elif requirement_id == "statistical_analysis":
        if metadata.get("analysis_kind") == "route_a_dynamic_l0":
            files = (
                (metadata.get("analysis_path"), metadata.get("analysis_hash"), "dynamic analysis"),
                (metadata.get("result_path"), metadata.get("result_hash"), "dynamic result"),
                (metadata.get("manifest_path"), metadata.get("manifest_hash"), "dynamic manifest"),
            )
            loaded: dict[str, dict[str, Any]] = {}
            for relative, expected, label in files:
                if not isinstance(relative, str) or not relative or not _is_digest(expected):
                    return f"{label} path/hash is invalid"
                candidate = (root / relative).resolve()
                if candidate != root and root not in candidate.parents:
                    return f"{label} escapes repo root"
                if not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
                    return f"{label} hash mismatch"
                try:
                    loaded[label] = json.loads(candidate.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    return f"{label} is invalid JSON: {error}"
            analysis = loaded["dynamic analysis"]
            result = loaded["dynamic result"]
            manifest = loaded["dynamic manifest"]
            if result.get("manifest_sha256") != metadata.get("manifest_hash"):
                return "dynamic result is not bound to the declared manifest"
            if analysis.get("result_sha256") != metadata.get("result_hash"):
                return "dynamic analysis is not bound to the declared result"
            if analysis.get("manifest_sha256") != metadata.get("manifest_hash"):
                return "dynamic analysis is not bound to the declared manifest"
            if result.get("independent_unit") != "scenario_seed" or analysis.get("independent_unit") != "scenario_seed":
                return "dynamic analysis uses the wrong independent unit"
            if len(result.get("rows", [])) != 500 or len(result.get("schedule", [])) != 500:
                return "dynamic result matrix is incomplete"
            if tuple(manifest.get("methods", ())) != (
                "no_runtime_assurance", "fixed_ttl", "unbound_filter",
                "nominal_cbf", "full_assurance_case",
            ):
                return "dynamic Tier-1 method set differs from the frozen design"
            if result.get("sealed_data_used") is not False or result.get("formal_experiment_run") is not False:
                return "dynamic result crosses sealed/formal boundary"
            if analysis.get("claim_generation_allowed") is not False:
                return "dynamic analysis improperly allows claim generation"
            return None
        if not _STATISTICAL_OUTPUTS.issubset(set(metadata.get("outputs", []))):
            return "required seed-level, interval, coverage, censoring, or guardrail output is missing"
        relative = metadata.get("development_result_manifest_path")
        expected_hash = metadata.get("development_result_manifest_hash")
        if not isinstance(relative, str) or not relative:
            return "development result manifest path is missing"
        if not _is_digest(expected_hash):
            return "development result manifest hash is invalid"
        candidate = (root / relative).resolve()
        if candidate != root and root not in candidate.parents:
            return "development result manifest escapes repo root"
        if not candidate.is_file():
            return "development result manifest is missing"
        try:
            report = analyze_development_results(candidate)
        except (DevelopmentAnalysisError, ValueError, json.JSONDecodeError) as error:
            return f"development statistical evidence is invalid: {error}"
        if report.result_manifest_hash != expected_hash.lower():
            return "development result manifest digest mismatch"
        if report.claim_generation_allowed is not False:
            return "development analysis improperly allows claim generation"
    elif requirement_id == "formal_experiment_plan":
        if metadata.get("frozen") is not True:
            return "formal experiment plan is not frozen"
        if metadata.get("sealed_data_opened") is not False:
            return "formal plan evidence indicates sealed-data access"
        relative = metadata.get("power_plan_manifest_path")
        expected_hash = metadata.get("power_plan_manifest_hash")
        if not isinstance(relative, str) or not relative:
            return "power plan manifest path is missing"
        if not _is_digest(expected_hash):
            return "power plan manifest hash is invalid"
        candidate = (root / relative).resolve()
        if candidate != root and root not in candidate.parents:
            return "power plan manifest escapes repo root"
        if not candidate.is_file():
            return "power plan manifest is missing"
        actual_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual_hash != expected_hash.lower():
            return "power plan manifest digest mismatch"
        try:
            plan = plan_seed_power_from_manifest(candidate)
        except (PowerPlanningError, ValueError, json.JSONDecodeError) as error:
            return f"formal power plan is invalid: {error}"
        if plan.selected_policy_seed_count is None:
            return "formal power plan did not reach target power"
        if not plan.resource_feasible:
            return "formal power plan exceeds the frozen resource budget"
    return None


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in string.hexdigits for character in value)
    )


def _tier_rank(tier: ReadinessTier) -> int:
    return {
        ReadinessTier.METHOD_PROTOCOL: 0,
        ReadinessTier.DEVELOPMENT_RESULTS_DRAFT: 1,
        ReadinessTier.FORMAL_RESULTS: 2,
    }[tier]
