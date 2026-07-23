import pytest

from agc_runtime_assurance.assurance_case import (
    build_valid_assurance_case,
    inject_assurance_fault,
    verify_assurance_case,
)


FAULT_FAMILIES = (
    "missing_required_field",
    "malformed_digest",
    "expired_action",
    "monotonic_time_reversal",
    "latency_fingerprint_mismatch",
    "backup_invariant_mismatch",
    "constraint_contract_mismatch",
    "audit_chain_tamper",
)


@pytest.mark.parametrize("mode", ["nominal", "filtered", "backup", "recovery"])
def test_valid_assurance_case_is_accepted(mode):
    result = verify_assurance_case(build_valid_assurance_case(case_id=mode, decision_mode=mode))
    assert result.accepted
    assert result.reason_code == "accepted"
    assert not result.pre_execution_blocked
    assert len(result.bundle_sha256) == 64


@pytest.mark.parametrize("family", FAULT_FAMILIES)
@pytest.mark.parametrize("variant", [0, 1, 2])
def test_each_preregistered_fault_is_blocked_and_localized(family, variant):
    bundle = build_valid_assurance_case(case_id=f"{family}-{variant}")
    result = verify_assurance_case(
        inject_assurance_fault(bundle, family=family, variant=variant)
    )
    assert not result.accepted
    assert result.pre_execution_blocked
    assert result.reason_code == family


def test_unknown_fault_family_is_rejected_by_fixture_builder():
    with pytest.raises(ValueError, match="unknown fault family"):
        inject_assurance_fault(
            build_valid_assurance_case(case_id="x"), family="not_registered", variant=0,
        )
