import hashlib
import json
from pathlib import Path

import pytest

from agc_runtime_assurance.assurance_case_benchmark import (
    AssuranceBenchmarkError,
    run_assurance_case_benchmark,
)
from agc_runtime_assurance.g1a_runner import canonical_code_tree_hash


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "docs").mkdir()
    (root / "experiments" / "manifests").mkdir(parents=True)
    (root / "src" / "dummy.py").write_text("x=1\n", encoding="utf-8")
    (root / "tests" / "test_dummy.py").write_text("def test_x(): assert True\n", encoding="utf-8")
    bound = {}
    for name in ("authorization", "route_decision", "protocol"):
        path = root / "docs" / f"{name}.json"
        path.write_text(json.dumps({"name": name}), encoding="utf-8")
        bound[name] = (path.relative_to(root).as_posix(), _digest(path))
    families = {
        name: [0, 1, 2] for name in (
            "missing_required_field", "malformed_digest", "expired_action",
            "monotonic_time_reversal", "latency_fingerprint_mismatch",
            "backup_invariant_mismatch", "constraint_contract_mismatch",
            "audit_chain_tamper",
        )
    }
    manifest = {
        "manifest_id": "test-benchmark",
        "experiment_family": "assurance_case_corruption_benchmark",
        "stage": "unsealed_development",
        "execution_authorized": True,
        "sealed_data_allowed": False,
        "formal_experiment": False,
        "claim_generation_allowed": False,
        "run_count": 1,
        "scientific_failure_retry_allowed": False,
        "code_tree_sha256": canonical_code_tree_hash(root),
        "valid_controls": ["nominal", "filtered", "backup", "recovery"],
        "fault_families": families,
        "randomization_seed": 20260718,
        "budget": {"max_cases": 28, "max_runtime_seconds": 30, "max_output_bytes": 500000},
    }
    for name, (path, digest) in bound.items():
        manifest[f"{name}_path"] = path
        manifest[f"{name}_sha256"] = digest
    path = root / "experiments" / "manifests" / "run.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return root, path


def test_runner_executes_frozen_randomized_schedule_once(tmp_path):
    root, manifest = _fixture(tmp_path)
    output = root / "results" / "result.json"
    result = run_assurance_case_benchmark(manifest, root, output)
    assert result["case_count"] == 28
    assert result["summary"] == {
        "valid_bundle_acceptance_rate": 1.0,
        "pre_execution_block_rate": 1.0,
        "reason_localization_accuracy": 1.0,
        "worst_family_undetected_fault_rate": 0.0,
        "all_success_gates_passed": True,
    }
    assert output.is_file()
    with pytest.raises(AssuranceBenchmarkError, match="output already exists"):
        run_assurance_case_benchmark(manifest, root, output)


def test_runner_refuses_code_hash_drift(tmp_path):
    root, manifest = _fixture(tmp_path)
    (root / "src" / "dummy.py").write_text("x=2\n", encoding="utf-8")
    with pytest.raises(AssuranceBenchmarkError, match="code tree"):
        run_assurance_case_benchmark(manifest, root, root / "result.json")
