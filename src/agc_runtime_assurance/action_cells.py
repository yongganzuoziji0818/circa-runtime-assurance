"""Prototype continuous-action-cell first-passage validity certificates.

This module implements only TAVH-v2's deterministic transport and per-cell
split-conformal layer.  Its semantics are cell-conditional marginal coverage,
not coverage conditional on the action being accepted/executed.  A selective
conformal layer is still required before any selected-action claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable

import numpy as np

from .contracts import ActionEnvelope
from .validity import ActionValidityCertificate


@dataclass(frozen=True)
class FirstPassageCellSpec:
    cell_id: Hashable
    action_radius: float
    state_radius: float
    transversality_kappa: float
    action_barrier_sensitivity: float
    state_barrier_sensitivity: float
    proof_reference: str

    def validate(self) -> None:
        values = np.asarray([
            self.action_radius, self.state_radius, self.transversality_kappa,
            self.action_barrier_sensitivity, self.state_barrier_sensitivity,
        ], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("cell sensitivity constants must be finite")
        if self.action_radius < 0.0 or self.state_radius < 0.0:
            raise ValueError("cell radii must be non-negative")
        if self.transversality_kappa <= 0.0:
            raise ValueError("transversality_kappa must be positive")
        if self.action_barrier_sensitivity < 0.0 or self.state_barrier_sensitivity < 0.0:
            raise ValueError("barrier sensitivities must be non-negative")
        if not self.proof_reference:
            raise ValueError("a proof or bound provenance reference is required")

    @property
    def action_time_sensitivity(self) -> float:
        return self.action_barrier_sensitivity / self.transversality_kappa

    @property
    def state_time_sensitivity(self) -> float:
        return self.state_barrier_sensitivity / self.transversality_kappa


@dataclass(frozen=True)
class CellCertificateRecord:
    spec: FirstPassageCellSpec
    certificate: ActionValidityCertificate | None
    calibration_count: int
    status: str


class CellConditionalValidityBank:
    """Fail-closed bank of per-cell certificates and deterministic transport."""

    coverage_semantics = "cell_conditional_marginal_not_selection_conditional"

    def __init__(self, records: dict[Hashable, CellCertificateRecord], alpha: float):
        self.records = dict(records)
        self.alpha = float(alpha)

    @classmethod
    def fit(
        cls,
        *,
        cell_ids: Iterable[Hashable],
        predicted_representative_horizons: np.ndarray,
        realized_representative_horizons: np.ndarray,
        specs: Iterable[FirstPassageCellSpec],
        alpha: float,
        minimum_cell_samples: int,
    ) -> "CellConditionalValidityBank":
        labels = np.asarray(list(cell_ids), dtype=object).reshape(-1)
        predicted = np.asarray(predicted_representative_horizons, dtype=float).reshape(-1)
        realized = np.asarray(realized_representative_horizons, dtype=float).reshape(-1)
        if labels.size == 0 or labels.shape != predicted.shape or predicted.shape != realized.shape:
            raise ValueError("cell labels and horizons must be non-empty and aligned")
        if not isinstance(minimum_cell_samples, int) or minimum_cell_samples <= 0:
            raise ValueError("minimum_cell_samples must be a positive integer")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")

        records: dict[Hashable, CellCertificateRecord] = {}
        for spec in specs:
            spec.validate()
            if spec.cell_id in records:
                raise ValueError("cell specs must have unique identifiers")
            mask = labels == spec.cell_id
            count = int(np.count_nonzero(mask))
            certificate = None
            status = "insufficient_cell_calibration"
            if count >= minimum_cell_samples:
                certificate = ActionValidityCertificate.fit(
                    predicted[mask], realized[mask], alpha=alpha
                )
                status = "cell_marginal_certificate_ready"
            records[spec.cell_id] = CellCertificateRecord(
                spec, certificate, count, status
            )
        return cls(records, alpha)

    def certified_duration(
        self,
        *,
        cell_id: Hashable,
        predicted_representative_horizon: float,
        action_deviation: float,
        state_uncertainty: float,
        observation_age: float,
        compute_delay: float,
        communication_delay: float,
        actuation_delay: float,
        guard_time: float = 0.0,
    ) -> float:
        record = self.records.get(cell_id)
        values = np.asarray([
            predicted_representative_horizon, action_deviation, state_uncertainty,
            observation_age, compute_delay, communication_delay, actuation_delay,
            guard_time,
        ], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("horizons, deviations, uncertainty, and debits must be non-negative")
        if record is None or record.certificate is None:
            return 0.0
        spec = record.spec
        if action_deviation > spec.action_radius or state_uncertainty > spec.state_radius:
            return 0.0
        transport = (
            spec.action_time_sensitivity * action_deviation
            + spec.state_time_sensitivity * state_uncertainty
        )
        timing = observation_age + compute_delay + communication_delay + actuation_delay + guard_time
        duration = (
            predicted_representative_horizon
            - record.certificate.optimism_correction
            - transport
            - timing
        )
        return max(0.0, float(duration))

    def issue(
        self,
        action: np.ndarray,
        *,
        issued_at: float,
        cell_id: Hashable,
        predicted_representative_horizon: float,
        action_deviation: float,
        state_uncertainty: float,
        observation_age: float,
        compute_delay: float,
        communication_delay: float,
        actuation_delay: float,
        guard_time: float = 0.0,
    ) -> ActionEnvelope:
        duration = self.certified_duration(
            cell_id=cell_id,
            predicted_representative_horizon=predicted_representative_horizon,
            action_deviation=action_deviation,
            state_uncertainty=state_uncertainty,
            observation_age=observation_age,
            compute_delay=compute_delay,
            communication_delay=communication_delay,
            actuation_delay=actuation_delay,
            guard_time=guard_time,
        )
        return ActionEnvelope(
            np.asarray(action, dtype=float), float(issued_at), float(issued_at) + duration,
            "cell_first_passage_candidate",
            "cell_marginal_not_selection_conditional" if duration > 0.0 else "reject_zero_horizon",
        )
