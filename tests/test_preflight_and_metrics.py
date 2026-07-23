import json

import numpy as np
import pytest

from agc_runtime_assurance.metrics import (
    FamilyCount, paired_seed_differences, worst_family_rate_per_seed,
    worst_family_upper_bound_per_seed,
)
from agc_runtime_assurance.preflight import PreflightError, verify_development_manifest


def _manifest(authorized: bool):
    digest = "a" * 64
    return {
        "manifest_id": "p4-dev-example", "stage": "development_pilot",
        "authorized": authorized, "formal_experiment_authorized": False,
        "sealed_data_authorized": False, "claim_generation_allowed": False,
        "experiment_family": "contract_smoke",
        "allowed_splits": ["calibration", "development"], "code_hash": digest,
        "split_hashes": {"calibration": digest, "development": "b" * 64},
    }


def test_preflight_refuses_unauthorized_development(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest(False)))
    with pytest.raises(PreflightError, match="not explicitly authorized"):
        verify_development_manifest(path)


def test_preflight_refuses_sealed_split_even_if_authorized(tmp_path):
    manifest = _manifest(True)
    manifest["allowed_splits"].append("sealed")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(PreflightError, match="exactly calibration and development"):
        verify_development_manifest(path)


def test_preflight_accepts_only_bounded_development_authority(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest(True)))
    authorization = verify_development_manifest(path)
    assert authorization.manifest_id == "p4-dev-example"
    assert authorization.experiment_family == "contract_smoke"
    assert len(authorization.manifest_hash) == 64


def test_preflight_refuses_non_hexadecimal_digest(tmp_path):
    manifest = _manifest(True)
    manifest["code_hash"] = "z" * 64
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(PreflightError, match="hexadecimal"):
        verify_development_manifest(path)


def test_preflight_refuses_incomplete_strong_baseline_matrix(tmp_path):
    manifest = _manifest(True)
    manifest["experiment_family"] = "strong_baseline_matrix"
    manifest["baseline_artifacts"] = {}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(PreflightError, match="requires exactly"):
        verify_development_manifest(path)


def test_preflight_accepts_only_verified_hashed_strong_baselines(tmp_path):
    manifest = _manifest(True)
    manifest["experiment_family"] = "strong_baseline_matrix"
    record = {
        "status": "verified", "upstream_url": "https://example.invalid/paper",
        "source_hash": "c" * 64, "implementation_hash": "d" * 64,
        "license_status": "verified_permissive",
        "reproduction_scope": "original_and_1u1g_adapted",
        "original_task_result_hash": "1" * 64,
        "adapted_task_result_hash": "2" * 64,
        "budget_contract_hash": "3" * 64,
    }
    manifest["baseline_artifacts"] = {
        name: dict(record) for name in (
            "aoi_cbf", "fallback_safe_mpc", "acofi", "multiagent_conformal_cbf"
        )
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert verify_development_manifest(path).experiment_family == "strong_baseline_matrix"


def test_preflight_refuses_provisional_strong_baseline(tmp_path):
    manifest = _manifest(True)
    manifest["experiment_family"] = "strong_baseline_matrix"
    record = {
        "status": "verified", "upstream_url": "https://example.invalid/paper",
        "source_hash": "c" * 64, "implementation_hash": "d" * 64,
        "license_status": "verified_permissive",
        "reproduction_scope": "original_and_1u1g_adapted",
        "original_task_result_hash": "1" * 64,
        "adapted_task_result_hash": "2" * 64,
        "budget_contract_hash": "3" * 64,
    }
    manifest["baseline_artifacts"] = {
        name: dict(record) for name in (
            "aoi_cbf", "fallback_safe_mpc", "acofi", "multiagent_conformal_cbf"
        )
    }
    manifest["baseline_artifacts"]["acofi"]["status"] = "provisional"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(PreflightError, match="acofi is not verified"):
        verify_development_manifest(path)


def test_preflight_refuses_uncleared_baseline_license(tmp_path):
    manifest = _manifest(True)
    manifest["experiment_family"] = "strong_baseline_matrix"
    record = {
        "status": "verified", "upstream_url": "https://example.invalid/paper",
        "source_hash": "c" * 64, "implementation_hash": "d" * 64,
        "license_status": "missing",
        "reproduction_scope": "original_and_1u1g_adapted",
        "original_task_result_hash": "1" * 64,
        "adapted_task_result_hash": "2" * 64,
        "budget_contract_hash": "3" * 64,
    }
    manifest["baseline_artifacts"] = {
        name: dict(record) for name in (
            "aoi_cbf", "fallback_safe_mpc", "acofi", "multiagent_conformal_cbf"
        )
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(PreflightError, match="license is not cleared"):
        verify_development_manifest(path)


def test_worst_family_rate_preserves_policy_seed_as_unit():
    rows = [
        FamilyCount(1, "mass", 1, 100), FamilyCount(1, "delay", 5, 100),
        FamilyCount(2, "mass", 2, 100), FamilyCount(2, "delay", 3, 100),
    ]
    assert worst_family_rate_per_seed(rows) == {1: 0.05, 2: 0.03}
    bounds = worst_family_upper_bound_per_seed(rows)
    assert bounds[1] > 0.05 and bounds[2] > 0.03
    differences = paired_seed_differences({1: 0.03, 2: 0.02}, {1: 0.05, 2: 0.03})
    assert np.allclose(differences, [-0.02, -0.01])
