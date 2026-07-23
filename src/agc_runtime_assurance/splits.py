"""Fail-closed split registry; sealed values are represented only by hashes."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Iterable


def _hash_values(values: Iterable[int]) -> str:
    canonical = json.dumps(sorted(int(v) for v in values), separators=(",", ":"))
    return sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class SplitManifest:
    train_hash: str
    calibration_hash: str
    development_hash: str
    sealed_hash: str
    counts: dict[str, int]


def build_split_manifest(
    *,
    train: Iterable[int],
    calibration: Iterable[int],
    development: Iterable[int],
    sealed: Iterable[int],
) -> SplitManifest:
    groups = {
        "train": {int(v) for v in train},
        "calibration": {int(v) for v in calibration},
        "development": {int(v) for v in development},
        "sealed": {int(v) for v in sealed},
    }
    names = list(groups)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = groups[left] & groups[right]
            if overlap:
                raise ValueError(f"split leakage between {left} and {right}: {sorted(overlap)[:5]}")
    return SplitManifest(
        train_hash=_hash_values(groups["train"]),
        calibration_hash=_hash_values(groups["calibration"]),
        development_hash=_hash_values(groups["development"]),
        sealed_hash=_hash_values(groups["sealed"]),
        counts={name: len(values) for name, values in groups.items()},
    )
