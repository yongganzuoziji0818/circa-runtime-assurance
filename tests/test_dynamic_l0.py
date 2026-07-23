import numpy as np

from agc_runtime_assurance.dynamic_l0 import (
    FAMILIES, METHODS, _assurance_bundle, _inject_runtime_fault,
    run_rollout,
)
from agc_runtime_assurance.assurance_case import verify_assurance_case


def test_runtime_action_mutation_is_blocked_by_assurance_case():
    bundle = _assurance_bundle(np.array([.1, .2, .3, .4, .5]), 1.0, .5, "case")
    _inject_runtime_fault(bundle, "action_digest_mismatch", 7)
    result = verify_assurance_case(bundle)
    assert not result.accepted
    assert result.pre_execution_blocked
    assert result.reason_code == "action_digest_mismatch"


def test_each_tier1_method_runs_one_paired_sandbox_rollout():
    rows = [run_rollout(method, "communication_shift", 901, horizon=8) for method in METHODS]
    assert {row.method for row in rows} == set(METHODS)
    assert all(row.scenario_seed == 901 for row in rows)
    assert all(row.scenario_family in FAMILIES for row in rows)
    assert all(1 <= row.action_count <= 8 for row in rows)
    full = next(row for row in rows if row.method == "full_assurance_case")
    assert full.pre_execution_block_count >= 1
