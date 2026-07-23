from __future__ import annotations

from agc_runtime_assurance.circa_isaac_gz1_v10_schema import (
    DEFAULT_CAP_BYTES,
    DEFAULT_HORIZON,
    DEFAULT_ROLLOUTS,
    allocate_tiny_fixture,
    array_schema,
    capacity_projection,
    validate_tiny_fixture,
)


def test_tiny_fixture_contains_only_seed_sentinels() -> None:
    fixture = allocate_tiny_fixture()
    validate_tiny_fixture(fixture)
    assert set(fixture["scenario_seed"].tolist()) == {-1}
    assert all(not value.dtype.hasobject for value in fixture.values())


def test_schema_dimensions_match_frozen_design() -> None:
    schema = array_schema(DEFAULT_ROLLOUTS, DEFAULT_HORIZON)
    assert len(schema) == 58
    assert schema["isaac_body_state"][1] == (15_360, 80, 2, 13)
    assert schema["scenario_seed"][0].str == "<i8"


def test_capacity_proposal_is_not_mislabeled_as_audit_pass() -> None:
    result = capacity_projection()
    assert result["status"] == "CAPACITY_PROPOSAL_ONLY_NOT_AN_AUDIT_PASS"
    assert result["raw_array_bytes"] == 949_974_512
    assert result["conservative_projected_bytes"] == 1_049_166_268
    assert result["proposed_cap_bytes"] == DEFAULT_CAP_BYTES
    assert result["proposal_headroom_bytes"] == 561_446_468
    assert result["projection_within_proposed_cap"] is True
    assert result["scientific_seed_material_generated"] is False
    assert result["scientific_run_executed"] is False
