import json
from pathlib import Path

import numpy as np
import pytest

from agc_runtime_assurance.aoi_cbf_paired_analysis import (
    PairedAnalysisError,
    analyze_pair,
    exact_mcnemar_two_sided,
    paired_bootstrap_interval,
    sha256_file,
    wilson_interval,
)


def test_exact_mcnemar_matches_frozen_contingency():
    assert exact_mcnemar_two_sided(4, 36) == pytest.approx(1.8570244719740003e-07)
    assert exact_mcnemar_two_sided(0, 0) == 1.0


def test_wilson_interval_contains_observed_rate():
    lower, upper = wilson_interval(30, 100)
    assert lower < 0.30 < upper


def test_paired_bootstrap_is_deterministic():
    differences = np.asarray([-1] * 4 + [1] * 36 + [0] * 60)
    first = paired_bootstrap_interval(differences, replicates=1000, seed=17)
    second = paired_bootstrap_interval(differences, replicates=1000, seed=17)
    assert first == second
    assert first[0] < 0.32 < first[1]


def _receipt(path: Path, *, delay_aware: bool, safe: list[bool]) -> str:
    value = {
        "status": "completed",
        "claim_generation_allowed": False,
        "sealed_data_used": False,
        "formal_or_g2": False,
        "checkpoint_sha256": {"actor.pkl": "a"},
        "evaluation": {
            "delay_aware": delay_aware,
            "mean_reward": 0.0,
            "mean_error": 0.0,
            "mean_length": 1.0,
        },
        "episodes": [
            {"episode_seed": index, "safe": outcome}
            for index, outcome in enumerate(safe)
        ],
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return sha256_file(path)


def test_analysis_refuses_unpaired_receipts(tmp_path):
    control = tmp_path / "control.json"
    delay = tmp_path / "delay.json"
    control_hash = _receipt(control, delay_aware=False, safe=[False] * 100)
    delay_hash = _receipt(delay, delay_aware=True, safe=[True] * 99)
    with pytest.raises(PairedAnalysisError, match="paired 100-seed"):
        analyze_pair(
            control,
            delay,
            expected_control_sha256=control_hash,
            expected_delay_sha256=delay_hash,
            bootstrap_replicates=100,
        )


def test_analysis_refuses_hash_mismatch(tmp_path):
    control = tmp_path / "control.json"
    delay = tmp_path / "delay.json"
    _receipt(control, delay_aware=False, safe=[False] * 100)
    delay_hash = _receipt(delay, delay_aware=True, safe=[True] * 100)
    with pytest.raises(PairedAnalysisError, match="control receipt hash"):
        analyze_pair(
            control,
            delay,
            expected_control_sha256="0" * 64,
            expected_delay_sha256=delay_hash,
            bootstrap_replicates=100,
        )
