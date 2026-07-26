"""Evidence-only assurance-case checks for CIRCA-RESS V10.

This module deliberately does not actuate, authorize, or control a system.  It
turns a bounded-risk evidence object into a typed admissibility decision so
that stale, incomplete, inconsistent, or out-of-scope evidence cannot be
silently consumed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
from typing import Iterable


class EvidenceStatus(str, Enum):
    ADMISSIBLE = "ADMISSIBLE"
    HOLD_SCOPE_MISMATCH = "HOLD_SCOPE_MISMATCH"
    HOLD_BELOW_REQUIRED_BOUND = "HOLD_BELOW_REQUIRED_BOUND"
    REFUSE_INVALID_WITNESS = "REFUSE_INVALID_WITNESS"
    REFUSE_INCOMPLETE_OUTCOME = "REFUSE_INCOMPLETE_OUTCOME"
    REFUSE_UNMODELED_INTERFERENCE = "REFUSE_UNMODELED_INTERFERENCE"
    STALE_EXPIRED = "STALE_EXPIRED"
    SUPERSEDED_CONFLICT = "SUPERSEDED_CONFLICT"


@dataclass(frozen=True)
class EvidenceObject:
    evidence_id: str
    subject_id: str
    edge_id: str
    issuer: str
    model_sha256: str
    config_sha256: str
    source_sha256: str
    scope_id: str
    issued_at: str
    valid_from: str
    valid_until: str
    lower_bound: float
    upper_bound: float
    witness_valid: bool
    outcome_complete: bool
    interference_modeled: bool

    def canonical_payload(self) -> bytes:
        return json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")

    def content_sha256(self) -> str:
        return sha256(self.canonical_payload()).hexdigest()


@dataclass(frozen=True)
class ConsumptionRequest:
    subject_id: str
    edge_id: str
    model_sha256: str
    config_sha256: str
    scope_id: str
    minimum_lower_bound: float
    observed_at: str


@dataclass(frozen=True)
class EvidenceDecision:
    evidence_id: str
    status: EvidenceStatus
    reason: str
    content_sha256: str
    lower_bound: float
    upper_bound: float
    operational_authorization: bool = False

    def as_dict(self) -> dict:
        result = asdict(self)
        result["status"] = self.status.value
        return result


def _instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _base_decision(
    evidence: EvidenceObject,
    status: EvidenceStatus,
    reason: str,
) -> EvidenceDecision:
    return EvidenceDecision(
        evidence_id=evidence.evidence_id,
        status=status,
        reason=reason,
        content_sha256=evidence.content_sha256(),
        lower_bound=evidence.lower_bound,
        upper_bound=evidence.upper_bound,
    )


def evaluate_evidence(
    evidence: EvidenceObject,
    request: ConsumptionRequest,
) -> EvidenceDecision:
    """Evaluate one immutable evidence object against one consumption request."""

    if not evidence.outcome_complete:
        return _base_decision(
            evidence,
            EvidenceStatus.REFUSE_INCOMPLETE_OUTCOME,
            "required outcome fields are incomplete",
        )
    if not evidence.witness_valid:
        return _base_decision(
            evidence,
            EvidenceStatus.REFUSE_INVALID_WITNESS,
            "registered witness validity is false",
        )
    if not evidence.interference_modeled:
        return _base_decision(
            evidence,
            EvidenceStatus.REFUSE_UNMODELED_INTERFERENCE,
            "declared interference assumptions are not satisfied",
        )
    if evidence.lower_bound > evidence.upper_bound:
        return _base_decision(
            evidence,
            EvidenceStatus.REFUSE_INCOMPLETE_OUTCOME,
            "lower bound exceeds upper bound",
        )

    expected_scope = (
        evidence.subject_id == request.subject_id
        and evidence.edge_id == request.edge_id
        and evidence.model_sha256 == request.model_sha256
        and evidence.config_sha256 == request.config_sha256
        and evidence.scope_id == request.scope_id
    )
    if not expected_scope:
        return _base_decision(
            evidence,
            EvidenceStatus.HOLD_SCOPE_MISMATCH,
            "subject, edge, model, configuration, or scope identity differs",
        )

    observed = _instant(request.observed_at)
    if observed < _instant(evidence.valid_from) or observed > _instant(
        evidence.valid_until
    ):
        return _base_decision(
            evidence,
            EvidenceStatus.STALE_EXPIRED,
            "observation time falls outside the evidence validity interval",
        )
    if evidence.lower_bound < request.minimum_lower_bound:
        return _base_decision(
            evidence,
            EvidenceStatus.HOLD_BELOW_REQUIRED_BOUND,
            "bounded evidence is below the request's declared minimum",
        )
    return _base_decision(
        evidence,
        EvidenceStatus.ADMISSIBLE,
        "identity, validity, completeness, witness, and bound checks passed",
    )


def resolve_evidence_set(
    evidence_objects: Iterable[EvidenceObject],
    request: ConsumptionRequest,
) -> list[EvidenceDecision]:
    """Evaluate a set and fail closed when admissible objects conflict.

    Two individually admissible objects conflict when they bind the same
    subject/edge/scope but carry different source hashes or bound intervals.
    Conflicting objects are all marked ``SUPERSEDED_CONFLICT``; this function
    never chooses the most favorable object.
    """

    objects = list(evidence_objects)
    decisions = [evaluate_evidence(item, request) for item in objects]
    admissible_indices = [
        index
        for index, decision in enumerate(decisions)
        if decision.status is EvidenceStatus.ADMISSIBLE
    ]
    if len(admissible_indices) <= 1:
        return decisions

    signatures = {
        (
            objects[index].source_sha256,
            objects[index].lower_bound,
            objects[index].upper_bound,
        )
        for index in admissible_indices
    }
    if len(signatures) == 1:
        return decisions

    for index in admissible_indices:
        decisions[index] = _base_decision(
            objects[index],
            EvidenceStatus.SUPERSEDED_CONFLICT,
            "multiple otherwise-admissible evidence objects conflict; no winner selected",
        )
    return decisions


def assurance_case_record(
    evidence_objects: Iterable[EvidenceObject],
    request: ConsumptionRequest,
) -> dict:
    """Return a deterministic, evidence-only assurance-case record."""

    decisions = resolve_evidence_set(evidence_objects, request)
    return {
        "schema": "circa_assurance_case_record/v10",
        "claim": "bounded evidence is admissible for the declared scope",
        "decisions": [decision.as_dict() for decision in decisions],
        "all_evidence_admissible": bool(decisions)
        and all(
            decision.status is EvidenceStatus.ADMISSIBLE
            for decision in decisions
        ),
        "operational_authorization": False,
        "boundary": (
            "This record supports evidence review only. It is not permission "
            "to actuate, deploy, or operate a controller."
        ),
    }
