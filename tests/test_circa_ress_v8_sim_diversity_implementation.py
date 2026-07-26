from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from agc_runtime_assurance.circa_ress_v8_sim_diversity_adapter import (
    DiversityAdapterError,
    DiversityRuntimeAdapter,
    FROZEN_FAMILIES,
    load_frozen_design,
    scenario_from_design,
    validate_world_semantics,
    world_patch_for,
)
from agc_runtime_assurance.circa_ress_v8_sim_diversity_schema import (
    DEFAULT_HORIZON,
    DEFAULT_ROLLOUTS,
    DiversitySchemaError,
    allocate_schema_sentinel,
    array_schema,
    schema_metadata,
    validate_schema_arrays,
)
from agc_runtime_assurance.contracts import ActionEnvelope


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "experiments/manifests/circa_ress_v8_sim_diversity_r1_DESIGN_ONLY.json"
WORLD = ROOT / "sim/gazebo/circa_ress_v8_sim_diversity_r1.sdf"
ADAPTER = ROOT / "src/agc_runtime_assurance/circa_ress_v8_sim_diversity_adapter.py"
FACADE = ROOT / "src/agc_runtime_assurance/public_timestamp_aligned_filter_primitives.py"


def test_world_is_self_contained_and_has_exactly_two_dynamic_agents() -> None:
    audit = validate_world_semantics(WORLD)
    assert audit["dynamic_agents"] == ["uav", "ugv"]
    assert audit["self_contained"] is True
    assert audit["simulator_invoked"] is False
    assert len(audit["patchable_models"]) == 6


def test_frozen_design_and_all_candidate_patches_are_complete() -> None:
    design = load_frozen_design(DESIGN)
    assert tuple(item["id"] for item in design["families"]) == FROZEN_FAMILIES
    for family in FROZEN_FAMILIES:
        for candidate in ("A", "B"):
            scenario = scenario_from_design(design, family, candidate)
            patch = world_patch_for(scenario)
            assert patch.scenario.family_id == family
            assert len(patch.model_patches) == 6
            assert all(item.model_name for item in patch.model_patches)


def test_adapter_import_closure_excludes_every_prior_scientific_runner() -> None:
    tree = ast.parse(ADAPTER.read_text(encoding="utf-8"))
    imported = "\n".join(
        ast.unparse(node)
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    )
    forbidden = (
        "circa_gazebo_gz0_v7_runner",
        "circa_gazebo_gz0_v8_runner",
        "circa_gazebo_gz0_v9_runner",
        "circa_ress_v6_wp3_pybullet",
        "circa_isaac_gz1_v10_runner",
        "gz.sim",
        "gazebo",
        "subprocess",
        "random",
    )
    assert not any(token in imported for token in forbidden)
    assert ".contracts" in imported
    assert ".public_timestamp_aligned_filter_primitives" in imported
    facade_imports = "\n".join(
        ast.unparse(node)
        for node in ast.parse(FACADE.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    )
    assert "_runner" not in facade_imports


def test_adapter_refuses_expired_and_invalid_provenance_without_filter_execution() -> None:
    adapter = DiversityRuntimeAdapter()
    expired = ActionEnvelope(
        np.ones(5), issued_at=1.0, valid_until=1.1, source="fixture"
    )
    refusal = adapter.decide(
        state_history=[],
        applied_action_history=[],
        nominal_envelope=expired,
        now=1.2,
        observation_delay_steps=0,
        communication_delay_steps=0,
    )
    assert refusal.fail_closed and refusal.refusal_code == "expired_action"
    invalid = adapter.decide(
        state_history=[],
        applied_action_history=[],
        nominal_envelope=expired,
        now=1.05,
        observation_delay_steps=0,
        communication_delay_steps=0,
        provenance_valid=False,
    )
    assert invalid.fail_closed and invalid.refusal_code == "invalid_provenance"
    assert len(refusal.decision_digest) == 64


def test_adapter_valid_path_is_in_memory_and_emits_a_typed_decision() -> None:
    adapter = DiversityRuntimeAdapter()
    state = np.array([-2.0, 0.0, 0.0, -0.1, 0.0, 0.0, 2.0, 0.0, 0.1, 0.0])
    decision = adapter.decide(
        state_history=[state, state],
        applied_action_history=[np.zeros(5), np.zeros(5)],
        nominal_envelope=ActionEnvelope(
            np.zeros(5), issued_at=0.0, valid_until=2.0, source="fixture"
        ),
        now=1.0,
        observation_delay_steps=0,
        communication_delay_steps=0,
    )
    assert decision.refusal_code in {"none", "invalid_alignment_evidence"}
    assert decision.envelope.action.shape == (5,)
    assert len(decision.decision_digest) == 64


def test_nonrunning_schema_has_no_object_members_or_seed_material() -> None:
    schema = array_schema(DEFAULT_ROLLOUTS, DEFAULT_HORIZON)
    assert len(schema) >= 40
    assert all(not dtype.hasobject for dtype, _ in schema.values())
    assert schema["future_seed_sentinel"][0] == np.dtype("<i8")
    assert schema["selected_action"][1] == (6400, 80, 5)
    metadata = schema_metadata()
    assert metadata["schema_only"] is True
    assert metadata["runnable"] is False
    assert metadata["scientific_seed_material_generated"] is False
    assert metadata["capacity_audit_executed"] is False


def test_small_schema_sentinel_validates_and_rejects_seed_or_untyped_failure() -> None:
    arrays = allocate_schema_sentinel(3, 4)
    validate_schema_arrays(arrays, 3, 4)
    arrays["future_seed_sentinel"][0] = 123
    with pytest.raises(DiversitySchemaError, match="seed material"):
        validate_schema_arrays(arrays, 3, 4)
    arrays = allocate_schema_sentinel(3, 4)
    arrays["completed_step_mask"][0, 0] = True
    arrays["fail_closed"][0, 0] = True
    with pytest.raises(DiversitySchemaError, match="typed refusal"):
        validate_schema_arrays(arrays, 3, 4)


def test_sources_contain_no_scientific_entry_or_output_writer() -> None:
    schema_body = (
        ROOT / "src/agc_runtime_assurance/circa_ress_v8_sim_diversity_schema.py"
    ).read_text(encoding="utf-8")
    adapter_body = ADAPTER.read_text(encoding="utf-8")
    forbidden = (
        "derive_scenario_seed",
        "compile_schedule",
        "np.save",
        "open(",
        "subprocess",
        "Popen",
        "gz sim",
    )
    for token in forbidden:
        assert token not in schema_body
        assert token not in adapter_body


def test_design_remains_nonrunnable() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    authorization = design["authorization"]
    assert authorization["scientific_seed_generation_authorized"] is False
    assert authorization["scientific_output_creation_authorized"] is False
    assert authorization["simulator_or_scientific_runner_authorized"] is False
