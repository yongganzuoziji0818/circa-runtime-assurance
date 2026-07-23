import json

import pytest

from agc_runtime_assurance.dynamic_l0 import DynamicL0Error
from agc_runtime_assurance.dynamic_l0_analysis import _upper_cp, analyze_dynamic_l0


def test_exact_upper_bound_is_not_zero_for_zero_events():
    assert 0.0 < _upper_cp(0, 20) < 1.0
    assert _upper_cp(20, 20) == 1.0


def test_analysis_fails_closed_on_incomplete_result(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"scenario_seeds": list(range(20))}), encoding="utf-8")
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"manifest_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(DynamicL0Error, match="not bound"):
        analyze_dynamic_l0(result, manifest, tmp_path / "analysis.json")
