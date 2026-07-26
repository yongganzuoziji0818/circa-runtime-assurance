from dataclasses import replace

from agc_runtime_assurance.circa_assurance_case_v10 import (
    ConsumptionRequest,
    EvidenceObject,
    EvidenceStatus,
    assurance_case_record,
    evaluate_evidence,
    resolve_evidence_set,
)


def valid_evidence() -> EvidenceObject:
    return EvidenceObject(
        evidence_id="e-001",
        subject_id="uav-17",
        edge_id="air-ground-link-a",
        issuer="offline-audit-r1",
        model_sha256="1" * 64,
        config_sha256="2" * 64,
        source_sha256="3" * 64,
        scope_id="bounded-loss-demo",
        issued_at="2026-07-26T08:00:00Z",
        valid_from="2026-07-26T08:00:00Z",
        valid_until="2026-07-26T10:00:00Z",
        lower_bound=0.82,
        upper_bound=0.93,
        witness_valid=True,
        outcome_complete=True,
        interference_modeled=True,
    )


def request() -> ConsumptionRequest:
    return ConsumptionRequest(
        subject_id="uav-17",
        edge_id="air-ground-link-a",
        model_sha256="1" * 64,
        config_sha256="2" * 64,
        scope_id="bounded-loss-demo",
        minimum_lower_bound=0.80,
        observed_at="2026-07-26T09:00:00Z",
    )


def test_valid_evidence_is_admissible_but_never_operational_authorization():
    decision = evaluate_evidence(valid_evidence(), request())
    assert decision.status is EvidenceStatus.ADMISSIBLE
    assert decision.operational_authorization is False
    record = assurance_case_record([valid_evidence()], request())
    assert record["all_evidence_admissible"] is True
    assert record["operational_authorization"] is False


def test_scope_or_hash_mismatch_holds():
    decision = evaluate_evidence(
        replace(valid_evidence(), config_sha256="f" * 64), request()
    )
    assert decision.status is EvidenceStatus.HOLD_SCOPE_MISMATCH


def test_expired_evidence_is_stale():
    decision = evaluate_evidence(
        valid_evidence(),
        replace(request(), observed_at="2026-07-26T11:00:00Z"),
    )
    assert decision.status is EvidenceStatus.STALE_EXPIRED


def test_invalid_witness_refuses():
    decision = evaluate_evidence(
        replace(valid_evidence(), witness_valid=False), request()
    )
    assert decision.status is EvidenceStatus.REFUSE_INVALID_WITNESS


def test_incomplete_outcome_refuses():
    decision = evaluate_evidence(
        replace(valid_evidence(), outcome_complete=False), request()
    )
    assert decision.status is EvidenceStatus.REFUSE_INCOMPLETE_OUTCOME


def test_unmodeled_interference_refuses():
    decision = evaluate_evidence(
        replace(valid_evidence(), interference_modeled=False), request()
    )
    assert decision.status is EvidenceStatus.REFUSE_UNMODELED_INTERFERENCE


def test_bound_below_declared_minimum_holds():
    decision = evaluate_evidence(
        replace(valid_evidence(), lower_bound=0.79), request()
    )
    assert decision.status is EvidenceStatus.HOLD_BELOW_REQUIRED_BOUND


def test_conflicting_admissible_evidence_is_not_optimistically_selected():
    first = valid_evidence()
    second = replace(
        first,
        evidence_id="e-002",
        source_sha256="4" * 64,
        lower_bound=0.86,
        upper_bound=0.95,
    )
    decisions = resolve_evidence_set([first, second], request())
    assert [item.status for item in decisions] == [
        EvidenceStatus.SUPERSEDED_CONFLICT,
        EvidenceStatus.SUPERSEDED_CONFLICT,
    ]


def test_canonical_hash_is_order_stable_and_content_sensitive():
    first = valid_evidence()
    assert len(first.content_sha256()) == 64
    assert first.content_sha256() != replace(first, issuer="other").content_sha256()
