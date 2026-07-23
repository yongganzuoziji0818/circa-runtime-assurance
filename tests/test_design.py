from collections import defaultdict

from agc_runtime_assurance.design import blocked_run_schedule


def test_randomized_complete_blocks_are_balanced_and_deterministic():
    kwargs = dict(
        methods=["tavh", "fixed_ttl", "aoi_cbf"], policy_seeds=[1, 2],
        scenario_ids=["mass", "delay"], order_seed=17,
    )
    first = blocked_run_schedule(**kwargs)
    second = blocked_run_schedule(**kwargs)
    assert first == second
    groups = defaultdict(list)
    for run in first:
        groups[run.block_id].append(run.method)
    assert len(groups) == 4
    assert all(set(methods) == set(kwargs["methods"]) for methods in groups.values())


def test_run_order_seed_changes_order_not_treatment_content():
    common = dict(
        methods=["tavh", "baseline"], policy_seeds=[1, 2, 3],
        scenario_ids=["a", "b"],
    )
    a = blocked_run_schedule(**common, order_seed=1)
    b = blocked_run_schedule(**common, order_seed=2)
    assert a != b
    content_a = {(x.policy_seed, x.scenario_id, x.method) for x in a}
    content_b = {(x.policy_seed, x.scenario_id, x.method) for x in b}
    assert content_a == content_b
