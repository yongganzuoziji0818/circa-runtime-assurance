"""Traceable deterministic latency budget for backup handover."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import string

import numpy as np


@dataclass(frozen=True)
class HandoverLatencyCertificate:
    """Frozen componentwise upper bounds used by validity and recovery gates.

    This G0 object accepts only a declared deterministic/specification bound.
    A sample maximum or empirical percentile must not be relabelled as a
    worst-case certificate; statistical tolerance bounds require a separate,
    pre-registered evidence contract.
    """

    observation_age_bound: float
    communication_bound: float
    computation_bound: float
    actuation_bound: float
    dispatch_jitter_bound: float
    guard_bound: float
    source_fingerprint: str
    evidence_kind: str = "deterministic_or_specification_bound"

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.observation_age_bound, self.communication_bound,
             self.computation_bound, self.actuation_bound,
             self.dispatch_jitter_bound, self.guard_bound], dtype=float,
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("latency component bounds must be finite and non-negative")
        if self.evidence_kind != "deterministic_or_specification_bound":
            raise ValueError("G0 handover certificate requires deterministic/specification evidence")
        if len(self.source_fingerprint) != 64 or any(
            character not in string.hexdigits for character in self.source_fingerprint
        ):
            raise ValueError("source_fingerprint must be a 64-character hexadecimal digest")

    @property
    def handover_total_bound(self) -> float:
        return float(
            self.observation_age_bound + self.communication_bound
            + self.computation_bound + self.actuation_bound
            + self.dispatch_jitter_bound + self.guard_bound
        )

    @property
    def execution_guard_bound(self) -> float:
        return float(self.dispatch_jitter_bound + self.guard_bound)

    def covers_observation_age(self, observed_age: float) -> bool:
        if not np.isfinite(observed_age) or observed_age < 0.0:
            raise ValueError("observed age must be finite and non-negative")
        return bool(float(observed_age) <= self.observation_age_bound)

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "observation_age_bound": self.observation_age_bound,
                "communication_bound": self.communication_bound,
                "computation_bound": self.computation_bound,
                "actuation_bound": self.actuation_bound,
                "dispatch_jitter_bound": self.dispatch_jitter_bound,
                "guard_bound": self.guard_bound,
                "source_fingerprint": self.source_fingerprint.lower(),
                "evidence_kind": self.evidence_kind,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(payload).hexdigest()
