import hashlib
import json

from agc_runtime_assurance.baseline_evidence import (
    verify_baseline_reproduction_manifest,
)
from agc_runtime_assurance.sandbox_task import SandboxComparisonTask


BASELINES = (
    "aoi_cbf", "fallback_safe_mpc", "acofi", "multiagent_conformal_cbf",
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _task_receipt(tmp_path, baseline, task):
    name = f"{baseline}-{task}.json"
    (tmp_path / name).write_text(name, encoding="utf-8")
    return {
        "status": "verified",
        "exit_code": 0,
        "seed_list": [11, 22],
        "evaluation_episodes_per_seed": 5,
        "result_path": name,
        "result_hash": _digest(name),
    }


def _record(tmp_path, baseline, scenario_hash):
    digest = "a" * 64
    return {
        "status": "task_level_verified",
        "paper_url": "https://example.invalid/paper",
        "paper_version": "v1",
        "upstream_url": "https://example.invalid/source",
        "license_status": "verified_permissive",
        "reproduction_scope": "original_and_1u1g_adapted",
        "source_hash": digest,
        "implementation_hash": digest,
        "environment_lock_hash": digest,
        "formula_map_hash": digest,
        "difference_report_hash": digest,
        "hyperparameter_source_hash": digest,
        "original_task": _task_receipt(tmp_path, baseline, "original"),
        "adapted_1u1g_task": _task_receipt(tmp_path, baseline, "adapted"),
        "budget_usage": {
            "training_env_steps": 80,
            "max_solver_calls_per_step": 1,
            "max_runtime_ms_per_step": 5.0,
            "scenario_manifest_hash": scenario_hash,
            "nominal_policy_hash": SandboxComparisonTask().nominal_policy_fingerprint,
            "constraint_contract_hash": SandboxComparisonTask().constraint_contract_fingerprint,
        },
    }


def _manifest(tmp_path):
    scenario = {
        "manifest_id": "test-scenarios",
        "stage": "g0_baseline_compatibility",
        "development_only": True,
        "authorized_execution": False,
        "sealed_data_referenced": False,
        "scenarios": [{
            "name": "nominal", "reset_seed": 1, "horizon": 10,
            "shift": {
                "uav_mass": 1.0, "uav_drag": 0.1, "ugv_friction": 0.1,
                "actuator_lag": 0.0, "sensor_bias": 0.0,
            },
            "runtime_timing": {
                "observation_age_s": 0.0, "communication_delay_s": 0.0,
                "compute_delay_s": 0.0, "actuation_delay_s": 0.0,
            },
        }],
    }
    scenario_path = tmp_path / "scenarios.json"
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
    scenario_hash = hashlib.sha256(scenario_path.read_bytes()).hexdigest()
    task = SandboxComparisonTask()
    return {
        "shared_budget": {
            "training_env_steps": 100,
            "evaluation_episodes_per_seed": 5,
            "seed_list": [11, 22],
            "max_solver_calls_per_step": 2,
            "max_runtime_ms_per_step": 10.0,
            "scenario_manifest_path": "scenarios.json",
            "scenario_manifest_hash": scenario_hash,
            "nominal_policy_hash": task.nominal_policy_fingerprint,
            "constraint_contract_hash": task.constraint_contract_fingerprint,
        },
        "baselines": {
            name: _record(tmp_path, name, scenario_hash) for name in BASELINES
        },
    }


def _audit(tmp_path, manifest):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return verify_baseline_reproduction_manifest(path, tmp_path)


def test_complete_original_and_adapted_receipts_pass(tmp_path):
    report = _audit(tmp_path, _manifest(tmp_path))
    assert report.ready is True
    assert all(finding.passed for finding in report.findings)
    assert len(report.budget.fingerprint) == 64


def test_equation_level_kernel_cannot_be_promoted(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["baselines"]["acofi"]["status"] = "provisional_kernel_only"
    report = _audit(tmp_path, manifest)
    finding = next(f for f in report.findings if f.baseline == "acofi")
    assert finding.passed is False
    assert finding.status == "provisional_kernel_only"


def test_uncleared_license_fails_even_with_successful_receipts(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["baselines"]["aoi_cbf"]["license_status"] = "missing"
    report = _audit(tmp_path, manifest)
    finding = next(f for f in report.findings if f.baseline == "aoi_cbf")
    assert finding.passed is False
    assert "license is not cleared" in finding.detail


def test_budget_overrun_fails_equal_budget_contract(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["baselines"]["fallback_safe_mpc"]["budget_usage"]["training_env_steps"] = 101
    report = _audit(tmp_path, manifest)
    finding = next(f for f in report.findings if f.baseline == "fallback_safe_mpc")
    assert finding.passed is False
    assert "training_env_steps" in finding.detail


def test_receipt_path_cannot_escape_evidence_root(tmp_path):
    outside = tmp_path.parent / "outside-baseline.json"
    outside.write_text("outside", encoding="utf-8")
    manifest = _manifest(tmp_path)
    receipt = manifest["baselines"]["multiagent_conformal_cbf"]["original_task"]
    receipt["result_path"] = "../outside-baseline.json"
    receipt["result_hash"] = _digest("outside")
    report = _audit(tmp_path, manifest)
    finding = next(f for f in report.findings if f.baseline == "multiagent_conformal_cbf")
    assert finding.passed is False
    assert "escapes evidence root" in finding.detail
