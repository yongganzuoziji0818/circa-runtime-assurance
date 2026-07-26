from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

from agc_runtime_assurance.circa_ress_v8_sim_diversity_runner import (
    HORIZON,
    ROLLOUTS,
    _operational_margin,
    _stress_plan,
    compile_schedule,
    derive_future_seed,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = json.loads(
    (
        ROOT
        / "experiments/manifests/circa_ress_v8_sim_diversity_r1_DESIGN_ONLY.json"
    ).read_text(encoding="utf-8")
)
OPERATIONALIZATION = json.loads(
    (
        ROOT
        / "experiments/manifests/circa_ress_v8_sim_diversity_r1_OPERATIONALIZATION_FROZEN.json"
    ).read_text(encoding="utf-8")
)
REGISTRY = json.loads(
    (
        ROOT
        / "sim/gazebo/circa_ress_v8_sim_diversity_r1_worlds/WORLD_VARIANT_REGISTRY.json"
    ).read_text(encoding="utf-8")
)
RUNNER = (
    ROOT
    / "src/agc_runtime_assurance/circa_ress_v8_sim_diversity_runner.py"
)


def test_fixture_schedule_has_exact_frozen_pairing_and_split_sizes() -> None:
    schedule = compile_schedule({"master_seed": 12345, "schedule_seed": 67890})
    assert len(schedule) == ROLLOUTS
    assert len({item.pair_id for item in schedule}) == 960
    assert len({item.future_seed for item in schedule}) == 960
    validation = [item for item in schedule if item.split_index == 0]
    evaluation = [item for item in schedule if item.split_index == 1]
    assert len(validation) == 1280
    assert len(evaluation) == 5120
    assert {item.method_index for item in validation} == {0, 3}
    assert {item.method_index for item in evaluation} == {0, 1, 2, 3}


def test_seed_derivation_is_namespaced_and_deterministic() -> None:
    left = derive_future_seed(7, "validation", "SDF1", "A", 0)
    assert left == derive_future_seed(7, "validation", "SDF1", "A", 0)
    assert left != derive_future_seed(7, "evaluation", "SDF1", "A", 0)
    assert left != derive_future_seed(7, "validation", "SDF1", "B", 0)


def test_all_initial_operational_margins_meet_frozen_floor() -> None:
    initial = np.asarray(OPERATIONALIZATION["task"]["initial_state"], dtype=float)
    floor = DESIGN["common_contract"]["minimum_initial_operational_margin_m"]
    for family in DESIGN["families"]:
        for candidate in ("A", "B"):
            operational, hard = _operational_margin(
                initial,
                family["id"],
                family[f"candidate_{candidate}"],
                OPERATIONALIZATION,
            )
            assert operational >= floor - 1e-12
            assert hard >= floor - 1e-12


def test_counter_based_stress_is_shared_and_bounded() -> None:
    entry = next(
        value
        for value in REGISTRY["variants"]
        if value["family"] == "SDF3" and value["candidate"] == "B"
    )
    first = _stress_plan(123, "SDF3", "B", entry["world_patch"])
    second = _stress_plan(123, "SDF3", "B", entry["world_patch"])
    assert first == second
    jitters, streaks, losses = first
    assert len(jitters) == len(streaks) == len(losses) == HORIZON
    assert set(jitters) == {0}
    assert all(value >= 0 for value in streaks)


def test_runner_imports_no_prior_scientific_runner_or_result_module() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
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
        "experiments.results",
        "frozen_analysis",
    )
    assert not any(token in imported for token in forbidden)


def test_registry_has_exactly_one_self_contained_world_per_candidate() -> None:
    assert REGISTRY["variant_count"] == 10
    keys = {(item["family"], item["candidate"]) for item in REGISTRY["variants"]}
    assert len(keys) == 10
    for item in REGISTRY["variants"]:
        body = (ROOT / item["path"]).read_text(encoding="utf-8").lower()
        assert not any(
            token in body for token in ("http://", "https://", "fuel://", "model://")
        )
