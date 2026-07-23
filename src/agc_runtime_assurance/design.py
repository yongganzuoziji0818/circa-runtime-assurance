"""Seeded randomized-complete-block schedules for fair method comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RunAssignment:
    order_index: int
    block_id: str
    policy_seed: int
    scenario_id: str
    method: str


def blocked_run_schedule(
    *,
    methods: list[str],
    policy_seeds: list[int],
    scenario_ids: list[str],
    order_seed: int,
) -> list[RunAssignment]:
    """Place every method once in each policy-seed x scenario block."""

    if len(methods) < 2 or len(set(methods)) != len(methods):
        raise ValueError("methods must contain at least two unique names")
    if not policy_seeds or len(set(policy_seeds)) != len(policy_seeds):
        raise ValueError("policy_seeds must be non-empty and unique")
    if not scenario_ids or len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("scenario_ids must be non-empty and unique")
    rng = np.random.default_rng(int(order_seed))
    blocks = [(int(seed), scenario) for seed in policy_seeds for scenario in scenario_ids]
    rng.shuffle(blocks)
    schedule: list[RunAssignment] = []
    for seed, scenario in blocks:
        method_order = list(methods)
        rng.shuffle(method_order)
        block_id = f"policy-{seed}__scenario-{scenario}"
        for method in method_order:
            schedule.append(
                RunAssignment(len(schedule), block_id, seed, scenario, method)
            )
    return schedule
