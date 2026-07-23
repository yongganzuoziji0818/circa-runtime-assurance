"""Pre-registered safety metrics; policy seed remains the inferential unit."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .risk import clopper_pearson_upper


@dataclass(frozen=True)
class FamilyCount:
    policy_seed: int
    scenario_family: str
    violations: int
    episodes: int

    def validate(self) -> None:
        if not self.scenario_family:
            raise ValueError("scenario_family must be non-empty")
        if self.episodes <= 0 or not 0 <= self.violations <= self.episodes:
            raise ValueError("require 0 <= violations <= episodes and episodes > 0")


def worst_family_rate_per_seed(rows: list[FamilyCount]) -> dict[int, float]:
    """Return max family violation rate for each independent policy seed."""

    totals: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        row.validate()
        key = (int(row.policy_seed), row.scenario_family)
        totals[key][0] += int(row.violations)
        totals[key][1] += int(row.episodes)
    if not totals:
        raise ValueError("at least one family count is required")
    per_seed: dict[int, list[float]] = defaultdict(list)
    for (seed, _), (violations, episodes) in totals.items():
        per_seed[seed].append(violations / episodes)
    return {seed: max(rates) for seed, rates in per_seed.items()}


def worst_family_upper_bound_per_seed(
    rows: list[FamilyCount], *, delta: float = 0.05
) -> dict[int, float]:
    """Conservative maximum of family-wise exact one-sided rate bounds."""

    totals: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        row.validate()
        key = (int(row.policy_seed), row.scenario_family)
        totals[key][0] += int(row.violations)
        totals[key][1] += int(row.episodes)
    if not totals:
        raise ValueError("at least one family count is required")
    families_per_seed: dict[int, int] = defaultdict(int)
    for seed, _ in totals:
        families_per_seed[seed] += 1
    per_seed: dict[int, list[float]] = defaultdict(list)
    for (seed, _), (violations, episodes) in totals.items():
        adjusted_delta = delta / families_per_seed[seed]
        per_seed[seed].append(clopper_pearson_upper(violations, episodes, adjusted_delta))
    return {seed: max(bounds) for seed, bounds in per_seed.items()}


def paired_seed_differences(
    treatment: dict[int, float], baseline: dict[int, float]
) -> np.ndarray:
    if set(treatment) != set(baseline) or not treatment:
        raise ValueError("treatment and baseline must share a non-empty policy-seed set")
    seeds = sorted(treatment)
    return np.asarray([treatment[s] - baseline[s] for s in seeds], dtype=float)
