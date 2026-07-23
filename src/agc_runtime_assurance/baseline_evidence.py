"""Fail-closed task-level reproduction evidence for strong baselines.

This module validates receipts; it does not execute baseline code.  Equation-
level kernels, a successful import, or a smoke test are deliberately
insufficient for ``task_level_verified`` status.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import string
from typing import Any

from .sandbox_task import SandboxComparisonTask
from .scenario_manifest import ScenarioManifestError, load_g0_scenario_manifest


class BaselineEvidenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenComparisonBudget:
    training_env_steps: int
    evaluation_episodes_per_seed: int
    seed_list: tuple[int, ...]
    max_solver_calls_per_step: int
    max_runtime_ms_per_step: float
    scenario_manifest_hash: str
    nominal_policy_hash: str
    constraint_contract_hash: str
    fingerprint: str


@dataclass(frozen=True)
class BaselineEvidenceFinding:
    baseline: str
    passed: bool
    status: str
    detail: str


@dataclass(frozen=True)
class BaselineMatrixEvidenceReport:
    ready: bool
    budget: FrozenComparisonBudget
    findings: tuple[BaselineEvidenceFinding, ...]
    manifest_hash: str


_BASELINES = {
    "aoi_cbf", "fallback_safe_mpc", "acofi", "multiagent_conformal_cbf",
}
_LICENSE_STATUSES = {"verified_permissive", "permission_documented"}


def verify_baseline_reproduction_manifest(
    manifest_path: str | Path, evidence_root: str | Path,
) -> BaselineMatrixEvidenceReport:
    """Verify frozen budget and original/adapted task receipts for all baselines."""
    raw = Path(manifest_path).read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    root = Path(evidence_root).resolve()
    budget = _verify_budget(manifest.get("shared_budget"), root)
    records = manifest.get("baselines")
    if not isinstance(records, dict) or set(records) != _BASELINES:
        raise BaselineEvidenceError("manifest must contain exactly the four frozen strong baselines")
    findings = tuple(
        _verify_baseline(name, records[name], budget, root)
        for name in sorted(_BASELINES)
    )
    return BaselineMatrixEvidenceReport(
        all(finding.passed for finding in findings), budget, findings,
        hashlib.sha256(raw).hexdigest(),
    )


def _verify_budget(record: Any, root: Path) -> FrozenComparisonBudget:
    if not isinstance(record, dict):
        raise BaselineEvidenceError("shared_budget must be an object")
    integer_fields = (
        "training_env_steps", "evaluation_episodes_per_seed",
        "max_solver_calls_per_step",
    )
    for field in integer_fields:
        if not isinstance(record.get(field), int) or record[field] <= 0:
            raise BaselineEvidenceError(f"shared_budget {field} must be a positive integer")
    runtime = record.get("max_runtime_ms_per_step")
    if not isinstance(runtime, (int, float)) or runtime <= 0:
        raise BaselineEvidenceError("shared_budget max_runtime_ms_per_step must be positive")
    seeds = record.get("seed_list")
    if (
        not isinstance(seeds, list) or not seeds
        or any(not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise BaselineEvidenceError("shared_budget seed_list must contain unique integers")
    for field in (
        "scenario_manifest_hash", "nominal_policy_hash", "constraint_contract_hash",
    ):
        _require_digest(record.get(field), f"shared_budget {field}")
    scenario_path = record.get("scenario_manifest_path")
    if not isinstance(scenario_path, str) or not scenario_path:
        raise BaselineEvidenceError("shared_budget scenario_manifest_path is missing")
    scenario_file = (root / scenario_path).resolve()
    if scenario_file != root and root not in scenario_file.parents:
        raise BaselineEvidenceError("shared_budget scenario manifest escapes evidence root")
    if not scenario_file.is_file():
        raise BaselineEvidenceError("shared_budget scenario manifest is missing")
    try:
        scenario_manifest = load_g0_scenario_manifest(scenario_file)
    except (ScenarioManifestError, ValueError, json.JSONDecodeError) as error:
        raise BaselineEvidenceError(f"shared_budget scenario manifest is invalid: {error}") from error
    if scenario_manifest.fingerprint != record["scenario_manifest_hash"].lower():
        raise BaselineEvidenceError("shared_budget scenario manifest digest mismatch")
    frozen_task = SandboxComparisonTask()
    if record["nominal_policy_hash"].lower() != frozen_task.nominal_policy_fingerprint:
        raise BaselineEvidenceError("shared_budget nominal policy differs from frozen task")
    if record["constraint_contract_hash"].lower() != frozen_task.constraint_contract_fingerprint:
        raise BaselineEvidenceError("shared_budget constraint contract differs from frozen task")
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return FrozenComparisonBudget(
        record["training_env_steps"], record["evaluation_episodes_per_seed"],
        tuple(seeds), record["max_solver_calls_per_step"], float(runtime),
        record["scenario_manifest_hash"], record["nominal_policy_hash"],
        record["constraint_contract_hash"], hashlib.sha256(canonical).hexdigest(),
    )


def _verify_baseline(
    name: str, record: Any, budget: FrozenComparisonBudget, root: Path,
) -> BaselineEvidenceFinding:
    if not isinstance(record, dict):
        return BaselineEvidenceFinding(name, False, "missing", "baseline record is missing")
    status = str(record.get("status", "missing"))
    if status != "task_level_verified":
        return BaselineEvidenceFinding(name, False, status, "status is not task_level_verified")
    try:
        _verify_provenance(name, record)
        _verify_task_receipt(name, "original_task", record.get("original_task"), budget, root)
        _verify_task_receipt(name, "adapted_1u1g_task", record.get("adapted_1u1g_task"), budget, root)
        _verify_usage(name, record.get("budget_usage"), budget)
    except BaselineEvidenceError as error:
        return BaselineEvidenceFinding(name, False, "failed", str(error))
    return BaselineEvidenceFinding(
        name, True, "task_level_verified",
        "provenance, license, original/adapted receipts, and equal-budget usage verified",
    )


def _verify_provenance(name: str, record: dict[str, Any]) -> None:
    for field in ("paper_url", "upstream_url"):
        value = record.get(field)
        if not isinstance(value, str) or not value.startswith("https://"):
            raise BaselineEvidenceError(f"{name} {field} must be an HTTPS URL")
    if not isinstance(record.get("paper_version"), str) or not record["paper_version"].strip():
        raise BaselineEvidenceError(f"{name} paper_version is missing")
    if record.get("license_status") not in _LICENSE_STATUSES:
        raise BaselineEvidenceError(f"{name} license is not cleared")
    if record.get("reproduction_scope") != "original_and_1u1g_adapted":
        raise BaselineEvidenceError(f"{name} reproduction scope is incomplete")
    for field in (
        "source_hash", "implementation_hash", "environment_lock_hash",
        "formula_map_hash", "difference_report_hash", "hyperparameter_source_hash",
    ):
        _require_digest(record.get(field), f"{name} {field}")


def _verify_task_receipt(
    name: str,
    task_name: str,
    receipt: Any,
    budget: FrozenComparisonBudget,
    root: Path,
) -> None:
    if not isinstance(receipt, dict) or receipt.get("status") != "verified":
        raise BaselineEvidenceError(f"{name} {task_name} receipt is not verified")
    if receipt.get("exit_code") != 0:
        raise BaselineEvidenceError(f"{name} {task_name} did not exit successfully")
    if tuple(receipt.get("seed_list", [])) != budget.seed_list:
        raise BaselineEvidenceError(f"{name} {task_name} seed list differs from frozen budget")
    if receipt.get("evaluation_episodes_per_seed") != budget.evaluation_episodes_per_seed:
        raise BaselineEvidenceError(f"{name} {task_name} evaluation count differs from frozen budget")
    _verify_file_receipt(name, task_name, receipt, root)


def _verify_file_receipt(
    name: str, task_name: str, receipt: dict[str, Any], root: Path,
) -> None:
    relative = receipt.get("result_path")
    digest = receipt.get("result_hash")
    if not isinstance(relative, str) or not relative:
        raise BaselineEvidenceError(f"{name} {task_name} result_path is missing")
    _require_digest(digest, f"{name} {task_name} result_hash")
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise BaselineEvidenceError(f"{name} {task_name} result path escapes evidence root")
    if not candidate.is_file():
        raise BaselineEvidenceError(f"{name} {task_name} result artifact is missing")
    if hashlib.sha256(candidate.read_bytes()).hexdigest() != digest.lower():
        raise BaselineEvidenceError(f"{name} {task_name} result digest mismatch")


def _verify_usage(name: str, usage: Any, budget: FrozenComparisonBudget) -> None:
    if not isinstance(usage, dict):
        raise BaselineEvidenceError(f"{name} budget_usage is missing")
    ceilings = (
        ("training_env_steps", budget.training_env_steps),
        ("max_solver_calls_per_step", budget.max_solver_calls_per_step),
        ("max_runtime_ms_per_step", budget.max_runtime_ms_per_step),
    )
    for field, ceiling in ceilings:
        value = usage.get(field)
        if not isinstance(value, (int, float)) or value < 0 or value > ceiling:
            raise BaselineEvidenceError(f"{name} exceeds or lacks budget field {field}")
    if usage.get("scenario_manifest_hash") != budget.scenario_manifest_hash:
        raise BaselineEvidenceError(f"{name} used a different scenario manifest")
    if usage.get("nominal_policy_hash") != budget.nominal_policy_hash:
        raise BaselineEvidenceError(f"{name} used a different nominal policy")
    if usage.get("constraint_contract_hash") != budget.constraint_contract_hash:
        raise BaselineEvidenceError(f"{name} used a different constraint contract")


def _require_digest(value: Any, label: str) -> None:
    if (
        not isinstance(value, str) or len(value) != 64
        or any(character not in string.hexdigits for character in value)
    ):
        raise BaselineEvidenceError(f"{label} must be a 64-character hexadecimal digest")
