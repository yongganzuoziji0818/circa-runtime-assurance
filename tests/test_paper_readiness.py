import hashlib
import json

from agc_runtime_assurance.paper_readiness import (
    ReadinessTier,
    _verify_special_requirement,
    audit_paper_readiness,
)


METHOD_REQUIREMENTS = (
    "focused_literature_audit", "scientific_question_freeze",
    "assurance_contract", "local_g0", "claim_evidence_map",
)
DEVELOPMENT_ONLY = (
    "remote_g0", "systems_assurance_benchmark_evidence", "development_authorization",
    "development_results", "statistical_analysis",
    "dynamics_and_invariant_evidence", "handover_latency_evidence",
    "formal_experiment_plan",
)


def _artifact(tmp_path, name="evidence.txt"):
    path = tmp_path / name
    path.write_text(name, encoding="utf-8")
    return {"path": name, "sha256": hashlib.sha256(name.encode()).hexdigest()}


def _verified_record(artifact, metadata=None):
    return {"status": "verified", "artifacts": [artifact], "metadata": metadata or {}}


def _manifest(tmp_path):
    artifact = _artifact(tmp_path)
    digest = "a" * 64
    requirements = {name: _verified_record(artifact) for name in METHOD_REQUIREMENTS}
    requirements["local_g0"]["metadata"] = {
        "repo_tree_hash": digest, "passed": 132, "total": 132,
    }
    requirements.update({name: {"status": "missing"} for name in DEVELOPMENT_ONLY})
    return {
        "target_tier": "development_results_draft",
        "sealed_data_opened": False,
        "formal_experiment_run": False,
        "requirements": requirements,
    }


def test_current_shape_can_be_method_ready_but_not_results_draft_ready(tmp_path):
    manifest = _manifest(tmp_path)
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = audit_paper_readiness(path, tmp_path)
    assert report.target_ready is False
    assert report.highest_ready_tier == ReadinessTier.METHOD_PROTOCOL
    assert any(
        finding.requirement_id == "remote_g0" and not finding.passed
        for finding in report.findings
    )


def test_artifact_digest_mismatch_fails_closed(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["target_tier"] = "method_protocol"
    manifest["requirements"]["claim_evidence_map"]["artifacts"][0]["sha256"] = "0" * 64
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = audit_paper_readiness(path, tmp_path)
    assert report.target_ready is False
    finding = next(f for f in report.findings if f.requirement_id == "claim_evidence_map")
    assert "digest mismatch" in finding.detail


def test_path_escape_is_refused(tmp_path):
    outside = tmp_path.parent / "outside-evidence.txt"
    outside.write_text("outside", encoding="utf-8")
    manifest = _manifest(tmp_path)
    manifest["target_tier"] = "method_protocol"
    manifest["requirements"]["assurance_contract"]["artifacts"] = [{
        "path": "../outside-evidence.txt",
        "sha256": hashlib.sha256(b"outside").hexdigest(),
    }]
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = audit_paper_readiness(path, tmp_path)
    finding = next(f for f in report.findings if f.requirement_id == "assurance_contract")
    assert finding.passed is False
    assert "escapes repo root" in finding.detail


def test_sealed_access_invalidates_preformal_readiness(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["target_tier"] = "method_protocol"
    manifest["sealed_data_opened"] = True
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = audit_paper_readiness(path, tmp_path)
    assert report.target_ready is False
    assert any(f.requirement_id == "authorization_boundary" for f in report.findings)


def test_statistical_readiness_rechecks_development_boundary(tmp_path):
    result_manifest = tmp_path / "development-results.json"
    result_manifest.write_text(json.dumps({
        "stage": "development_results",
        "development_authorized": False,
        "execution_exit_code": None,
        "formal_experiment_run": False,
        "sealed_data_used": False,
        "claim_generation_allowed": False,
        "only_calibration_and_development_splits": True,
    }), encoding="utf-8")
    digest = hashlib.sha256(result_manifest.read_bytes()).hexdigest()
    manifest = _manifest(tmp_path)
    manifest["requirements"]["statistical_analysis"] = {
        "status": "verified",
        "artifacts": [{"path": result_manifest.name, "sha256": digest}],
        "metadata": {
            "outputs": [
                "worst_family_rate_per_seed",
                "worst_family_upper_bound_per_seed",
                "paired_effect_estimate",
                "paired_confidence_interval",
                "deadline_coverage",
                "censoring_breakdown",
                "guardrail_metrics",
            ],
            "development_result_manifest_path": result_manifest.name,
            "development_result_manifest_hash": digest,
        },
    }
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = audit_paper_readiness(path, tmp_path)
    finding = next(f for f in report.findings if f.requirement_id == "statistical_analysis")
    assert finding.passed is False
    assert "development_authorized" in finding.detail


def test_formal_plan_readiness_rechecks_power_variance_boundary(tmp_path):
    power_manifest = tmp_path / "power-plan.json"
    power_manifest.write_text(json.dumps({
        "stage": "formal_power_planning",
        "development_variance_available": False,
        "formal_experiment_authorized": False,
        "sealed_data_used": False,
    }), encoding="utf-8")
    digest = hashlib.sha256(power_manifest.read_bytes()).hexdigest()
    manifest = _manifest(tmp_path)
    manifest["requirements"]["formal_experiment_plan"] = {
        "status": "verified",
        "artifacts": [{"path": power_manifest.name, "sha256": digest}],
        "metadata": {
            "frozen": True,
            "sealed_data_opened": False,
            "power_plan_manifest_path": power_manifest.name,
            "power_plan_manifest_hash": digest,
        },
    }
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = audit_paper_readiness(path, tmp_path)
    finding = next(f for f in report.findings if f.requirement_id == "formal_experiment_plan")
    assert finding.passed is False
    assert "variance is not available" in finding.detail


def test_systems_assurance_benchmark_rechecks_route_and_success_gates(tmp_path):
    route = {
        "status": "option_A_confirmed",
        "options": {"A": {"selected": True}},
    }
    benchmark_manifest = {
        "experiment_family": "assurance_case_corruption_benchmark",
        "sealed_data_allowed": False,
        "formal_experiment": False,
    }
    route_path = tmp_path / "route.json"
    manifest_path = tmp_path / "benchmark.json"
    route_path.write_text(json.dumps(route), encoding="utf-8")
    manifest_path.write_text(json.dumps(benchmark_manifest), encoding="utf-8")
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    result = {
        "manifest_sha256": manifest_digest,
        "sealed_data_used": False,
        "formal_experiment_run": False,
        "claim_generation_allowed": False,
        "valid_control_count": 4,
        "fault_case_count": 24,
        "families": {name: {} for name in (
            "missing_required_field", "malformed_digest", "expired_action",
            "monotonic_time_reversal", "latency_fingerprint_mismatch",
            "backup_invariant_mismatch", "constraint_contract_mismatch",
            "audit_chain_tamper",
        )},
        "summary": {
            "valid_bundle_acceptance_rate": 1.0,
            "pre_execution_block_rate": 1.0,
            "reason_localization_accuracy": 1.0,
            "worst_family_undetected_fault_rate": 0.0,
            "all_success_gates_passed": True,
        },
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    record = {
        "status": "verified",
        "artifacts": [
            {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (route_path, manifest_path, result_path)
        ],
        "metadata": {
            "benchmark_result_path": result_path.name,
            "benchmark_result_hash": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "benchmark_manifest_path": manifest_path.name,
            "benchmark_manifest_hash": manifest_digest,
            "route_decision_path": route_path.name,
            "route_decision_hash": hashlib.sha256(route_path.read_bytes()).hexdigest(),
        },
    }
    manifest = _manifest(tmp_path)
    manifest["requirements"]["systems_assurance_benchmark_evidence"] = record
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = audit_paper_readiness(readiness_path, tmp_path)
    finding = next(
        value for value in report.findings
        if value.requirement_id == "systems_assurance_benchmark_evidence"
    )
    assert finding.passed

    result["summary"]["worst_family_undetected_fault_rate"] = 0.1
    result_path.write_text(json.dumps(result), encoding="utf-8")
    record["artifacts"][2]["sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    record["metadata"]["benchmark_result_hash"] = record["artifacts"][2]["sha256"]
    readiness_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = audit_paper_readiness(readiness_path, tmp_path)
    finding = next(
        value for value in report.findings
        if value.requirement_id == "systems_assurance_benchmark_evidence"
    )
    assert not finding.passed
    assert "success gates" in finding.detail


def test_route_a_dynamic_analysis_rechecks_hash_chain_and_unit(tmp_path):
    manifest = {"methods": [
        "no_runtime_assurance", "fixed_ttl", "unbound_filter",
        "nominal_cbf", "full_assurance_case",
    ]}
    manifest_path = tmp_path / "dynamic_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    result = {
        "manifest_sha256": manifest_hash, "independent_unit": "scenario_seed",
        "rows": [{}] * 500, "schedule": [{}] * 500,
        "sealed_data_used": False, "formal_experiment_run": False,
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    result_hash = hashlib.sha256(result_path.read_bytes()).hexdigest()
    analysis = {
        "result_sha256": result_hash, "manifest_sha256": manifest_hash,
        "independent_unit": "scenario_seed", "claim_generation_allowed": False,
    }
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    record = {
        "analysis_kind": "route_a_dynamic_l0",
        "analysis_path": analysis_path.name,
        "analysis_hash": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
        "result_path": result_path.name, "result_hash": result_hash,
        "manifest_path": manifest_path.name, "manifest_hash": manifest_hash,
    }
    assert _verify_special_requirement("statistical_analysis", record, tmp_path) is None
    analysis["independent_unit"] = "episode"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    record["analysis_hash"] = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    assert "wrong independent unit" in _verify_special_requirement(
        "statistical_analysis", record, tmp_path,
    )
