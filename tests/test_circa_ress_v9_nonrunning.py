from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from agc_runtime_assurance.circa_ress_v9_initial_domain import (
    INITIAL_MARGIN_FLOOR_M,
    initial_state_from_future_seed,
    validate_complete_support,
)
from agc_runtime_assurance.circa_ress_v9_runner import (
    INDEPENDENT_UNITS,
    ROLLOUTS,
    compile_schedule,
)
from agc_runtime_assurance.circa_ress_v9_schema import (
    DEFAULT_HORIZON,
    DEFAULT_ROLLOUTS,
    allocate_schema_sentinel,
    array_schema,
    schema_metadata,
    validate_schema_arrays,
)


ROOT = Path(__file__).resolve().parents[1]


def test_complete_initial_support_is_analytically_feasible() -> None:
    margins = validate_complete_support()
    assert min(margins.values()) >= INITIAL_MARGIN_FLOOR_M
    assert np.isclose(margins["SDF1_B_ugv_corridor"], INITIAL_MARGIN_FLOOR_M)
    assert np.isclose(margins["SDF4_B_ugv_corridor"], INITIAL_MARGIN_FLOOR_M)


def test_future_seed_mapping_is_bounded_deterministic_and_keeps_ugv_y_zero() -> None:
    first = initial_state_from_future_seed(123456)
    second = initial_state_from_future_seed(123456)
    assert np.array_equal(first, second)
    assert -2.53 <= first[0] <= -2.47
    assert -0.03 <= first[1] <= 0.03
    assert 2.47 <= first[6] <= 2.53
    assert first[7] == 0.0


def test_v9_schedule_dimensions_without_materializing_a_receipt() -> None:
    schedule = compile_schedule({"master_seed": 1, "schedule_seed": 2})
    assert len(schedule) == ROLLOUTS == 7424
    assert len({row.pair_id for row in schedule}) == INDEPENDENT_UNITS == 1088
    positive_evaluation = [
        row
        for row in schedule
        if row.split_index == 1 and row.family_index < 4
    ]
    sdf5_evaluation = [
        row
        for row in schedule
        if row.split_index == 1 and row.family_index == 4
    ]
    assert len(positive_evaluation) == 5120
    assert len(sdf5_evaluation) == 1024


def test_nonrunning_v9_schema_is_seedless_and_has_frozen_dimensions() -> None:
    schema = array_schema()
    assert DEFAULT_ROLLOUTS == 7424
    assert DEFAULT_HORIZON == 80
    assert schema["selected_action"][1] == (7424, 80, 5)
    assert all(not dtype.hasobject for dtype, _ in schema.values())
    arrays = allocate_schema_sentinel(3, 4)
    validate_schema_arrays(arrays, 3, 4)
    assert np.all(arrays["future_seed_sentinel"] == -1)
    metadata = schema_metadata()
    assert metadata["schema_only"] is True
    assert metadata["runnable"] is False
    assert metadata["capacity_audit_executed"] is False


def test_v9_modules_have_no_import_time_execution_or_seed_literals() -> None:
    for relative in (
        "src/agc_runtime_assurance/circa_ress_v9_initial_domain.py",
        "src/agc_runtime_assurance/circa_ress_v9_schema.py",
        "src/agc_runtime_assurance/circa_ress_v9_runner.py",
    ):
        body = (ROOT / relative).read_text(encoding="utf-8")
        tree = ast.parse(body)
        assert "__main__" not in body
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"mkdir", "write_bytes", "write_text"}
            for node in ast.walk(tree)
        )
