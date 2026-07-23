"""Tamper-evident, fail-closed assurance-case bundle verification.

The bundle certifies the internal consistency and traceability of one runtime
decision.  Passing this verifier is not a physical-system safety certificate.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import string
from typing import Any, Mapping


SCHEMA_VERSION = "p4-assurance-case-v1"
DECISION_MODES = {"nominal", "filtered", "backup", "recovery"}
REASON_CODES = {
    "accepted",
    "missing_required_field",
    "malformed_digest",
    "expired_action",
    "monotonic_time_reversal",
    "latency_fingerprint_mismatch",
    "backup_invariant_mismatch",
    "constraint_contract_mismatch",
    "audit_chain_tamper",
    "invalid_schema",
    "invalid_decision_semantics",
    "action_digest_mismatch",
}


@dataclass(frozen=True)
class AssuranceCaseVerification:
    accepted: bool
    reason_code: str
    pre_execution_blocked: bool
    bundle_sha256: str


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in string.hexdigits for character in value)
    )


def _required(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    return all(key in mapping for key in keys)


def _reject(bundle: Any, reason: str) -> AssuranceCaseVerification:
    if reason not in REASON_CODES:
        raise ValueError(f"unknown assurance-case reason: {reason}")
    try:
        digest = _sha256(bundle)
    except (TypeError, ValueError):
        digest = hashlib.sha256(repr(bundle).encode("utf-8")).hexdigest()
    return AssuranceCaseVerification(False, reason, True, digest)


def assurance_audit_payload(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact decision payload bound by the bundle's audit entry."""

    action = bundle["action"]
    evidence = bundle["evidence"]
    return {
        "schema_version": bundle["schema_version"],
        "case_id": bundle["case_id"],
        "decision_mode": bundle["decision_mode"],
        "action_sha256": action["action_sha256"],
        "issued_at": action["issued_at"],
        "valid_until": action["valid_until"],
        "consumed_at": action["consumed_at"],
        "source": action["source"],
        "evidence": dict(evidence),
    }


def verify_assurance_case(bundle: Any) -> AssuranceCaseVerification:
    """Verify one bundle and reject the first mechanically localizable defect."""

    if not isinstance(bundle, Mapping):
        return _reject(bundle, "missing_required_field")
    if not _required(
        bundle,
        ("schema_version", "case_id", "decision_mode", "action", "evidence", "audit"),
    ):
        return _reject(bundle, "missing_required_field")
    if bundle["schema_version"] != SCHEMA_VERSION:
        return _reject(bundle, "invalid_schema")
    if not isinstance(bundle["case_id"], str) or not bundle["case_id"]:
        return _reject(bundle, "missing_required_field")
    if bundle["decision_mode"] not in DECISION_MODES:
        return _reject(bundle, "invalid_decision_semantics")

    action, evidence, audit = bundle["action"], bundle["evidence"], bundle["audit"]
    if not all(isinstance(item, Mapping) for item in (action, evidence, audit)):
        return _reject(bundle, "missing_required_field")
    if not _required(
        action,
        ("values", "action_sha256", "issued_at", "valid_until", "consumed_at", "source"),
    ):
        return _reject(bundle, "missing_required_field")
    evidence_keys = (
        "code_sha256", "config_sha256", "model_sha256", "predictor_sha256",
        "certificate_sha256", "constraint_contract_sha256",
        "filter_constraint_sha256", "latency_sha256",
        "recoverability_latency_sha256", "backup_invariant_sha256",
        "fallback_backup_invariant_sha256",
    )
    if not _required(evidence, evidence_keys):
        return _reject(bundle, "missing_required_field")
    if not _required(audit, ("previous_hash", "payload", "hash")):
        return _reject(bundle, "missing_required_field")

    digests = [action["action_sha256"], audit["previous_hash"], audit["hash"]]
    digests.extend(evidence[key] for key in evidence_keys)
    if not all(_is_digest(value) for value in digests):
        return _reject(bundle, "malformed_digest")

    values = action["values"]
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
    ):
        return _reject(bundle, "invalid_decision_semantics")
    if _sha256(values) != action["action_sha256"]:
        return _reject(bundle, "action_digest_mismatch")

    times = (action["issued_at"], action["valid_until"], action["consumed_at"])
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in times):
        return _reject(bundle, "invalid_decision_semantics")
    issued_at, valid_until, consumed_at = map(float, times)
    if issued_at < 0.0 or consumed_at < issued_at:
        return _reject(bundle, "monotonic_time_reversal")
    if valid_until <= issued_at or consumed_at > valid_until:
        return _reject(bundle, "expired_action")

    expected_sources = {
        "nominal": "nominal",
        "filtered": "safety_filter",
        "backup": "verified_backup",
        "recovery": "verified_backup",
    }
    if action["source"] != expected_sources[bundle["decision_mode"]]:
        return _reject(bundle, "invalid_decision_semantics")
    if evidence["latency_sha256"] != evidence["recoverability_latency_sha256"]:
        return _reject(bundle, "latency_fingerprint_mismatch")
    if evidence["backup_invariant_sha256"] != evidence["fallback_backup_invariant_sha256"]:
        return _reject(bundle, "backup_invariant_mismatch")
    if evidence["constraint_contract_sha256"] != evidence["filter_constraint_sha256"]:
        return _reject(bundle, "constraint_contract_mismatch")

    expected_payload = assurance_audit_payload(bundle)
    if audit["payload"] != expected_payload:
        return _reject(bundle, "audit_chain_tamper")
    expected_audit_hash = hashlib.sha256(
        (audit["previous_hash"] + _canonical_bytes(expected_payload).decode("utf-8")).encode("utf-8")
    ).hexdigest()
    if audit["hash"] != expected_audit_hash:
        return _reject(bundle, "audit_chain_tamper")
    return AssuranceCaseVerification(True, "accepted", False, _sha256(bundle))


def build_valid_assurance_case(
    *, case_id: str, decision_mode: str = "nominal", issued_at: float = 10.0,
) -> dict[str, Any]:
    """Build a deterministic valid fixture for development fault injection."""

    if decision_mode not in DECISION_MODES:
        raise ValueError("unknown decision_mode")
    digest = lambda label: hashlib.sha256(label.encode("utf-8")).hexdigest()
    sources = {
        "nominal": "nominal", "filtered": "safety_filter",
        "backup": "verified_backup", "recovery": "verified_backup",
    }
    values = [0.1, -0.2, 0.3, -0.4, 0.5]
    latency = digest("latency-v1")
    invariant = digest("backup-invariant-v1")
    constraint = digest("constraint-contract-v1")
    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "decision_mode": decision_mode,
        "action": {
            "values": values,
            "action_sha256": _sha256(values),
            "issued_at": float(issued_at),
            "valid_until": float(issued_at + 1.0),
            "consumed_at": float(issued_at + 0.5),
            "source": sources[decision_mode],
        },
        "evidence": {
            "code_sha256": digest("code-v1"),
            "config_sha256": digest("config-v1"),
            "model_sha256": digest("model-v1"),
            "predictor_sha256": digest("predictor-v1"),
            "certificate_sha256": digest("certificate-v1"),
            "constraint_contract_sha256": constraint,
            "filter_constraint_sha256": constraint,
            "latency_sha256": latency,
            "recoverability_latency_sha256": latency,
            "backup_invariant_sha256": invariant,
            "fallback_backup_invariant_sha256": invariant,
        },
        "audit": {"previous_hash": "0" * 64},
    }
    payload = assurance_audit_payload(bundle)
    bundle["audit"]["payload"] = payload
    body = _canonical_bytes(payload).decode("utf-8")
    bundle["audit"]["hash"] = hashlib.sha256(("0" * 64 + body).encode("utf-8")).hexdigest()
    return bundle


def inject_assurance_fault(
    bundle: Mapping[str, Any], *, family: str, variant: int,
) -> dict[str, Any]:
    """Inject exactly one preregistered corruption without repairing the audit."""

    if not isinstance(variant, int) or variant < 0:
        raise ValueError("variant must be a non-negative integer")
    damaged = deepcopy(dict(bundle))
    if family == "missing_required_field":
        targets = (("action", "valid_until"), ("evidence", "code_sha256"), ("audit", "hash"))
        parent, key = targets[variant % len(targets)]
        del damaged[parent][key]
    elif family == "malformed_digest":
        targets = (
            ("action", "action_sha256"), ("evidence", "code_sha256"),
            ("evidence", "certificate_sha256"), ("audit", "hash"),
        )
        parent, key = targets[variant % len(targets)]
        damaged[parent][key] = "not-a-sha256"
    elif family == "expired_action":
        damaged["action"]["consumed_at"] = damaged["action"]["valid_until"] + 0.01 * (variant + 1)
    elif family == "monotonic_time_reversal":
        damaged["action"]["consumed_at"] = damaged["action"]["issued_at"] - 0.01 * (variant + 1)
    elif family == "latency_fingerprint_mismatch":
        damaged["evidence"]["recoverability_latency_sha256"] = hashlib.sha256(
            f"wrong-latency-{variant}".encode("utf-8")
        ).hexdigest()
    elif family == "backup_invariant_mismatch":
        damaged["evidence"]["fallback_backup_invariant_sha256"] = hashlib.sha256(
            f"wrong-invariant-{variant}".encode("utf-8")
        ).hexdigest()
    elif family == "constraint_contract_mismatch":
        damaged["evidence"]["filter_constraint_sha256"] = hashlib.sha256(
            f"wrong-constraint-{variant}".encode("utf-8")
        ).hexdigest()
    elif family == "audit_chain_tamper":
        damaged["audit"]["payload"] = dict(damaged["audit"]["payload"])
        damaged["audit"]["payload"]["case_id"] = f"tampered-{variant}"
    else:
        raise ValueError(f"unknown fault family: {family}")
    return damaged
