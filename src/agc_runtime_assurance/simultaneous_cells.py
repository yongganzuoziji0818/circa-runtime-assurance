"""Selection-robust marginal calibration over a frozen finite action-cell set.

The calibration score for one exchangeable context is the maximum optimism
error over every pre-registered action cell.  Therefore the split-conformal
event is simultaneous across cells and survives an arbitrary downstream cell
selection rule.  It is still marginal over contexts: it is not
action-conditional coverage, accepted-action conditional risk control, an
anytime guarantee, or protection against arbitrary distribution drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Hashable, Iterable

import numpy as np

from .transversality import TeamTransportResult


@dataclass(frozen=True)
class SimultaneousCellCertificate:
    alpha: float
    optimism_correction: float
    calibration_contexts: int
    cell_ids: tuple[Hashable, ...]
    calibration_hash: str

    coverage_semantics = (
        "simultaneous_over_frozen_cells_marginal_over_exchangeable_contexts_"
        "not_action_or_acceptance_conditional_not_anytime"
    )

    @classmethod
    def fit(
        cls,
        *,
        cell_ids: Iterable[Hashable],
        predicted_horizons: np.ndarray,
        realized_horizons: np.ndarray,
        alpha: float,
    ) -> "SimultaneousCellCertificate":
        cells = tuple(cell_ids)
        predicted = np.asarray(predicted_horizons, dtype=float)
        realized = np.asarray(realized_horizons, dtype=float)
        if (
            predicted.ndim != 2
            or predicted.shape[0] == 0
            or predicted.shape[1] == 0
            or predicted.shape != realized.shape
        ):
            raise ValueError(
                "predicted and realized horizons must be aligned non-empty matrices"
            )
        if len(cells) != predicted.shape[1] or len(set(cells)) != len(cells):
            raise ValueError("cell_ids must be unique and match matrix columns")
        if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(realized)):
            raise ValueError("horizon matrices must be finite")
        if np.any(predicted < 0.0) or np.any(realized < 0.0):
            raise ValueError("horizons must be non-negative")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")

        context_scores = np.max(predicted - realized, axis=1)
        ordered = np.sort(context_scores)
        rank = math.ceil((ordered.size + 1) * (1.0 - alpha))
        correction = (
            math.inf if rank > ordered.size else float(ordered[rank - 1])
        )

        payload = np.concatenate((predicted, realized), axis=1)
        order = np.lexsort(tuple(payload[:, index] for index in reversed(
            range(payload.shape[1])
        )))
        digest = sha256()
        digest.update(repr(cells).encode("utf-8"))
        digest.update(payload[order].tobytes())
        return cls(
            alpha=float(alpha),
            optimism_correction=correction,
            calibration_contexts=int(predicted.shape[0]),
            cell_ids=cells,
            calibration_hash=digest.hexdigest(),
        )

    def certified_duration(
        self,
        *,
        cell_id: Hashable,
        predicted_representative_horizon: float,
        deterministic_transport_debit: float,
        observation_age: float,
        compute_delay: float,
        communication_delay: float,
        actuation_delay: float,
        guard_time: float = 0.0,
    ) -> float:
        values = np.asarray(
            [
                predicted_representative_horizon,
                deterministic_transport_debit,
                observation_age,
                compute_delay,
                communication_delay,
                actuation_delay,
                guard_time,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("horizon and all debits must be finite and non-negative")
        if cell_id not in self.cell_ids or not math.isfinite(
            self.optimism_correction
        ):
            return 0.0
        duration = (
            predicted_representative_horizon
            - self.optimism_correction
            - float(np.sum(values[1:]))
        )
        return max(0.0, float(duration))

    def certified_duration_from_team_transport(
        self,
        *,
        cell_id: Hashable,
        predicted_representative_horizon: float,
        team_transport: TeamTransportResult,
        observation_age: float,
        compute_delay: float,
        communication_delay: float,
        actuation_delay: float,
        guard_time: float = 0.0,
    ) -> float:
        """Compose simultaneous calibration with a verified T1 transport result.

        The statistical and deterministic horizons are reliability bounds, so
        composition takes their minimum.  A caller cannot replace an invalid
        team transport with an arbitrary numeric debit.
        """

        timing = np.asarray(
            [
                predicted_representative_horizon,
                observation_age,
                compute_delay,
                communication_delay,
                actuation_delay,
                guard_time,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(timing)) or np.any(timing < 0.0):
            raise ValueError("horizon and timing debits must be finite and non-negative")
        if (
            cell_id not in self.cell_ids
            or not team_transport.valid
            or not math.isfinite(self.optimism_correction)
        ):
            return 0.0
        statistical_horizon = (
            predicted_representative_horizon - self.optimism_correction
        )
        composed_horizon = min(
            statistical_horizon, team_transport.transported_team_horizon
        )
        return max(0.0, float(composed_horizon - np.sum(timing[1:])))
