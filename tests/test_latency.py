import pytest

from agc_runtime_assurance.latency import HandoverLatencyCertificate


def _certificate():
    return HandoverLatencyCertificate(
        observation_age_bound=0.20, communication_bound=0.03,
        computation_bound=0.04, actuation_bound=0.05,
        dispatch_jitter_bound=0.01, guard_bound=0.02,
        source_fingerprint="a" * 64,
    )


def test_handover_latency_certificate_sums_every_frozen_component():
    certificate = _certificate()
    assert certificate.handover_total_bound == pytest.approx(0.35)
    assert certificate.execution_guard_bound == pytest.approx(0.03)
    assert len(certificate.fingerprint()) == 64


def test_observed_age_must_remain_within_certified_bound():
    certificate = _certificate()
    assert certificate.covers_observation_age(0.20)
    assert not certificate.covers_observation_age(0.200001)


def test_negative_component_bound_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        HandoverLatencyCertificate(
            observation_age_bound=0.1, communication_bound=-0.01,
            computation_bound=0.1, actuation_bound=0.1,
            dispatch_jitter_bound=0.0, guard_bound=0.0,
            source_fingerprint="b" * 64,
        )


def test_untraceable_source_digest_is_rejected():
    with pytest.raises(ValueError, match="64-character hexadecimal"):
        HandoverLatencyCertificate(
            observation_age_bound=0.1, communication_bound=0.1,
            computation_bound=0.1, actuation_bound=0.1,
            dispatch_jitter_bound=0.0, guard_bound=0.0,
            source_fingerprint="not-a-digest",
        )
