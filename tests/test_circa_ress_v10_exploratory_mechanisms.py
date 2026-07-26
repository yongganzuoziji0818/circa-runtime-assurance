import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts" / "analyze_circa_ress_v10_v9_mechanisms_exploratory.py"
)
SPEC = importlib.util.spec_from_file_location("v10_mechanisms", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def fixture_arrays():
    rows, horizon = 20, 2
    arrays = {
        "split_index": np.concatenate(
            [
                np.ones(16, dtype=np.uint8),
                np.zeros(4, dtype=np.uint8),
            ]
        ),
        "family_index": np.zeros(rows, dtype=np.uint8),
        "method_index": np.concatenate(
            [
                np.repeat(np.arange(4, dtype=np.uint8), 4),
                np.full(4, MODULE.PRIMARY, dtype=np.uint8),
            ]
        ),
        "driver_index": np.tile(np.array([0, 1], dtype=np.uint8), 10),
        "pair_id": np.concatenate(
            [
                np.tile(np.repeat(np.array([10, 11]), 2), 4),
                np.repeat(np.array([20, 21]), 2),
            ]
        ).astype(
            np.int32
        ),
        "operational_first_violation": np.zeros(rows, dtype=bool),
        "hard_first_violation": np.zeros(rows, dtype=bool),
        "applied_intervention": np.zeros(rows, dtype=bool),
        "typed_refusal": np.zeros(rows, dtype=bool),
        "refusal_code": np.zeros((rows, horizon), dtype=np.uint8),
        "backup_tube_feasible": np.ones((rows, horizon), dtype=np.int8),
        "terminal_reachability": np.ones((rows, horizon), dtype=np.int8),
        "operational_margin_m": np.ones((rows, horizon), dtype=float),
    }
    return arrays


def test_grouped_boolean_uses_pair_level_or_across_drivers():
    arrays = fixture_arrays()
    arrays["operational_first_violation"][0] = True
    mask = MODULE.selected_rows(
        arrays, split=1, family=0, method=0
    )
    ids, values = MODULE.grouped_boolean(
        arrays, mask, "operational_first_violation"
    )
    assert ids.tolist() == [10, 11]
    assert values.tolist() == [True, False]


def test_family_summary_reports_all_methods_and_descriptive_tags():
    arrays = fixture_arrays()
    summary = MODULE.family_summary(arrays, 0)
    assert summary["shadow_opportunity_rate"] == 0.0
    assert summary["primary_residual_operational_risk"] == 0.0
    assert set(summary["methods"]) == set(MODULE.METHODS)
    assert "OPPORTUNITY_ZERO_IN_FROZEN_SHADOW" in summary[
        "descriptive_boundary_tags"
    ]
    assert "REDUCTION_BELOW_ORIGINAL_FROZEN_MINIMUM" in summary[
        "descriptive_boundary_tags"
    ]


def test_validation_error_is_grouped_by_pair_not_step():
    arrays = fixture_arrays()
    arrays["split_index"].fill(0)
    arrays["method_index"].fill(MODULE.PRIMARY)
    arrays["operational_margin_m"][0, 0] = -0.1
    result = MODULE.validation_witness_summary(arrays, 0)
    assert result["negative_margin_certificate_steps"] == 1
    assert result["witness_error_units"] == 1
