"""Conservative mechanical gate for handing control to a verified backup."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backup_invariant import BackupInvariantResult
from .latency import HandoverLatencyCertificate
from .risk import ConstraintMargins


@dataclass(frozen=True)
class RecoverabilityResult:
    recoverable: bool
    current_margins: np.ndarray
    required_margins: np.ndarray
    critical_constraint: int


@dataclass(frozen=True)
class AssuredRecoverabilityResult:
    """Composition evidence binding a margin gate to a verified backup set."""

    recoverable: bool
    reason: str
    current_margins: np.ndarray
    required_margins: np.ndarray
    critical_constraint: int
    backup_invariant_fingerprint: str
    latency_fingerprint: str
    handover_total_bound: float


class MarginErosionRecoverabilityGate:
    """Check that margins survive worst-case handover plus backup settling.

    ``erosion_rate_bounds[j]`` is a verified upper bound on how fast constraint
    margin j can decrease while the advanced action is revoked and the backup
    reaches its invariant regime.  This is a sufficient G0 gate, not a learned
    predictor and not a replacement for reachability verification.
    """

    def __init__(
        self,
        erosion_rate_bounds: np.ndarray,
        margin_guards: np.ndarray,
        *,
        backup_settle_time: float,
    ):
        rates = np.asarray(erosion_rate_bounds, dtype=float).reshape(-1)
        guards = np.asarray(margin_guards, dtype=float).reshape(-1)
        if rates.shape != (4,) or guards.shape != (4,):
            raise ValueError("recoverability gate requires four constraint groups")
        if not np.all(np.isfinite(rates)) or not np.all(np.isfinite(guards)):
            raise ValueError("recoverability parameters must be finite")
        if np.any(rates < 0.0) or np.any(guards < 0.0) or backup_settle_time < 0.0:
            raise ValueError("rates, guards, and settle time must be non-negative")
        self.erosion_rate_bounds = rates
        self.margin_guards = guards
        self.backup_settle_time = float(backup_settle_time)

    def evaluate(
        self, margins: ConstraintMargins, *, handover_delay: float
    ) -> RecoverabilityResult:
        if not np.isfinite(handover_delay) or handover_delay < 0.0:
            raise ValueError("handover_delay must be finite and non-negative")
        current = margins.as_array()
        required = self.margin_guards + self.erosion_rate_bounds * (
            float(handover_delay) + self.backup_settle_time
        )
        slack = current - required
        critical = int(np.argmin(slack))
        return RecoverabilityResult(
            bool(np.all(slack >= 0.0)), current.copy(), required, critical
        )


class VerifiedBackupRecoverabilityGate:
    """Fail closed unless both backup invariance and handover margin are verified."""

    def __init__(
        self,
        *,
        backup_invariant: BackupInvariantResult,
        margin_gate: MarginErosionRecoverabilityGate,
    ):
        if not isinstance(backup_invariant, BackupInvariantResult):
            raise TypeError("a mechanically evaluated BackupInvariantResult is required")
        if not isinstance(margin_gate, MarginErosionRecoverabilityGate):
            raise TypeError("a MarginErosionRecoverabilityGate is required")
        self.backup_invariant = backup_invariant
        self.margin_gate = margin_gate

    def evaluate(
        self,
        margins: ConstraintMargins,
        *,
        latency_certificate: HandoverLatencyCertificate,
    ) -> AssuredRecoverabilityResult:
        if not isinstance(latency_certificate, HandoverLatencyCertificate):
            raise TypeError("a HandoverLatencyCertificate is required")
        margin_result = self.margin_gate.evaluate(
            margins, handover_delay=latency_certificate.handover_total_bound
        )
        if not self.backup_invariant.verified:
            recoverable = False
            reason = f"backup_invariant_not_verified:{self.backup_invariant.reason}"
        elif not margin_result.recoverable:
            recoverable = False
            reason = "handover_margin_gate_failed"
        else:
            recoverable = True
            reason = "verified_backup_recoverable"
        return AssuredRecoverabilityResult(
            recoverable=recoverable,
            reason=reason,
            current_margins=margin_result.current_margins,
            required_margins=margin_result.required_margins,
            critical_constraint=margin_result.critical_constraint,
            backup_invariant_fingerprint=self.backup_invariant.fingerprint,
            latency_fingerprint=latency_certificate.fingerprint(),
            handover_total_bound=latency_certificate.handover_total_bound,
        )
