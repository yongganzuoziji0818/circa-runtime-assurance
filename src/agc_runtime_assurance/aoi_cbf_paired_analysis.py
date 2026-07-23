"""Fail-closed development analysis for the paired AoI-CBF adapter diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


class PairedAnalysisError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_mcnemar_two_sided(control_only: int, delay_only: int) -> float:
    if min(control_only, delay_only) < 0:
        raise PairedAnalysisError("discordant counts must be non-negative")
    discordant = control_only + delay_only
    if discordant == 0:
        return 1.0
    lower = min(control_only, delay_only)
    tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / 2**discordant
    return min(1.0, 2.0 * tail)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise PairedAnalysisError("invalid binomial count")
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2)
    ) / denominator
    return center - radius, center + radius


def paired_bootstrap_interval(
    differences: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise PairedAnalysisError("paired differences must be a non-empty vector")
    if replicates <= 0:
        raise PairedAnalysisError("bootstrap replicates must be positive")
    generator = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    batch = 10_000
    for start in range(0, replicates, batch):
        stop = min(start + batch, replicates)
        indices = generator.integers(0, values.size, size=(stop - start, values.size))
        bootstrap[start:stop] = values[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975], method="linear")
    return float(lower), float(upper)


def analyze_pair(
    control_path: str | Path,
    delay_path: str | Path,
    *,
    expected_control_sha256: str,
    expected_delay_sha256: str,
    bootstrap_replicates: int = 100_000,
    bootstrap_seed: int = 20_260_718,
) -> dict[str, Any]:
    control_file = Path(control_path)
    delay_file = Path(delay_path)
    if sha256_file(control_file) != expected_control_sha256:
        raise PairedAnalysisError("control receipt hash mismatch")
    if sha256_file(delay_file) != expected_delay_sha256:
        raise PairedAnalysisError("delay receipt hash mismatch")
    control = json.loads(control_file.read_text(encoding="utf-8"))
    delay = json.loads(delay_file.read_text(encoding="utf-8"))
    for label, receipt in (("control", control), ("delay", delay)):
        if receipt.get("status") != "completed":
            raise PairedAnalysisError(f"{label} receipt is not completed")
        if receipt.get("claim_generation_allowed") is not False:
            raise PairedAnalysisError(f"{label} receipt permits claim generation")
        if receipt.get("sealed_data_used") is not False or receipt.get("formal_or_g2") is not False:
            raise PairedAnalysisError(f"{label} receipt crosses the evidence boundary")

    control_records = control.get("episodes")
    delay_records = delay.get("episodes")
    if not isinstance(control_records, list) or not isinstance(delay_records, list):
        raise PairedAnalysisError("episode records are missing")
    control_seeds = [record.get("episode_seed") for record in control_records]
    delay_seeds = [record.get("episode_seed") for record in delay_records]
    if control_seeds != delay_seeds or len(control_seeds) != 100:
        raise PairedAnalysisError("receipts do not contain the exact paired 100-seed stream")
    if control.get("checkpoint_sha256") != delay.get("checkpoint_sha256"):
        raise PairedAnalysisError("checkpoint hashes differ across arms")
    if control["evaluation"].get("delay_aware") is not False:
        raise PairedAnalysisError("control arm semantics mismatch")
    if delay["evaluation"].get("delay_aware") is not True:
        raise PairedAnalysisError("delay arm semantics mismatch")

    control_safe = np.asarray([bool(record.get("safe")) for record in control_records])
    delay_safe = np.asarray([bool(record.get("safe")) for record in delay_records])
    both_safe = int(np.sum(control_safe & delay_safe))
    control_only = int(np.sum(control_safe & ~delay_safe))
    delay_only = int(np.sum(~control_safe & delay_safe))
    both_unsafe = int(np.sum(~control_safe & ~delay_safe))
    differences = delay_safe.astype(np.int8) - control_safe.astype(np.int8)
    risk_difference = float(np.mean(differences))
    bootstrap_interval = paired_bootstrap_interval(
        differences,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    control_successes = int(control_safe.sum())
    delay_successes = int(delay_safe.sum())

    return {
        "analysis_status": "completed_development_only",
        "claim_generation_allowed": False,
        "sealed_data_used": False,
        "formal_or_g2": False,
        "input_sha256": {
            "control_receipt.json": expected_control_sha256,
            "delay_receipt.json": expected_delay_sha256,
        },
        "sample_hierarchy": {
            "trained_policy_seeds": 1,
            "paired_episode_seeds": 100,
            "training_seed_uncertainty_estimable": False,
            "inference_scope": "conditional_on_the_single_trained_policy_seed",
        },
        "safe_outcome": {
            "control": {
                "safe": control_successes,
                "total": 100,
                "rate": control_successes / 100,
                "wilson_95_ci": list(wilson_interval(control_successes, 100)),
            },
            "delay_aware": {
                "safe": delay_successes,
                "total": 100,
                "rate": delay_successes / 100,
                "wilson_95_ci": list(wilson_interval(delay_successes, 100)),
            },
            "paired_contingency": {
                "both_safe": both_safe,
                "control_only_safe": control_only,
                "delay_only_safe": delay_only,
                "both_unsafe": both_unsafe,
            },
            "paired_risk_difference_delay_minus_control": risk_difference,
            "paired_bootstrap_95_ci": list(bootstrap_interval),
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "mcnemar_exact_two_sided_p": exact_mcnemar_two_sided(
                control_only,
                delay_only,
            ),
        },
        "descriptive_only": {
            "mean_reward": {
                "control": control["evaluation"]["mean_reward"],
                "delay_aware": delay["evaluation"]["mean_reward"],
            },
            "mean_error": {
                "control": control["evaluation"]["mean_error"],
                "delay_aware": delay["evaluation"]["mean_error"],
            },
            "mean_length": {
                "control": control["evaluation"]["mean_length"],
                "delay_aware": delay["evaluation"]["mean_length"],
            },
        },
        "interpretation_boundary": (
            "The episode-paired signal is exploratory development evidence conditional on one "
            "trained policy seed. It does not estimate training-seed variability and cannot "
            "support superiority, certification, or confirmatory claims."
        ),
    }
