"""Equation-level multi-agent conformal-CBF baseline adapter.

Implements Huriot and Sibai (arXiv:2409.18862v4), Algorithm 1 and Eq. (5),
for already-linearized pairwise CBF constraints.  Trajectory prediction and the
task-specific barrier derivatives remain external, which keeps the comparison
honest and allows all methods to share the same predictor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baselines import ConformalDecisionVariable
from .filtering import AffineSafetyFilter, FilterResult


def arctan_signed_transform(value: np.ndarray | float) -> np.ndarray:
    """Paper example s(r)=atan(r)/pi with range (-1/2, 1/2)."""

    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("loss-transform inputs must be finite")
    return np.arctan(array) / np.pi


def conformal_cbf_interval_loss(
    predicted_constraint_terms: np.ndarray,
    true_constraint_terms: np.ndarray,
    *,
    conformal_value: float,
) -> float:
    """Worst signed predicted-vs-true CBF gap over agents and interval samples."""

    predicted = np.asarray(predicted_constraint_terms, dtype=float)
    truth = np.asarray(true_constraint_terms, dtype=float)
    if predicted.size == 0 or predicted.shape != truth.shape:
        raise ValueError("predicted and true CBF terms must be non-empty and aligned")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(truth)):
        raise ValueError("CBF terms must be finite")
    if not np.isfinite(conformal_value):
        raise ValueError("conformal value must be finite")
    gaps = predicted + float(conformal_value) - truth
    return float(np.max(arctan_signed_transform(gaps)))


@dataclass(frozen=True)
class ConformalCBFStep:
    filter_result: FilterResult
    conformal_value: float
    affine_A: np.ndarray
    affine_b: np.ndarray


class MultiAgentConformalCBF:
    """Minimal-deviation QP with periodic worst-gap conformal adaptation."""

    def __init__(
        self,
        *,
        safety_filter: AffineSafetyFilter,
        target_loss: float,
        learning_rate: float,
        initial_value: float,
    ):
        self.safety_filter = safety_filter
        self.variable = ConformalDecisionVariable(
            target_loss=target_loss,
            learning_rate=learning_rate,
            value=initial_value,
            loss_lower=-0.5,
            loss_upper=0.5,
        )

    def filter_action(
        self,
        *,
        nominal_action: np.ndarray,
        control_coefficients: np.ndarray,
        predicted_offsets: np.ndarray,
        fallback_action: np.ndarray,
    ) -> ConformalCBFStep:
        """Solve min ||u-u_ref||² subject to G u+d+lambda >= 0."""

        coefficients = np.asarray(control_coefficients, dtype=float)
        offsets = np.asarray(predicted_offsets, dtype=float).reshape(-1)
        if coefficients.ndim != 2 or offsets.shape != (coefficients.shape[0],):
            raise ValueError("control coefficients and predicted offsets must align")
        if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(offsets)):
            raise ValueError("conformal CBF constraints must be finite")
        # G u + d + lambda >= 0  <=>  -G u <= d + lambda.
        A = -coefficients
        b = offsets + self.variable.value
        result = self.safety_filter.filter(nominal_action, A, b, fallback_action)
        return ConformalCBFStep(result, float(self.variable.value), A.copy(), b.copy())

    def observe_interval(
        self,
        *,
        predicted_constraint_terms: np.ndarray,
        true_constraint_terms: np.ndarray,
    ) -> float:
        """Update lambda after delayed ground-truth trajectories arrive."""

        loss = conformal_cbf_interval_loss(
            predicted_constraint_terms,
            true_constraint_terms,
            conformal_value=self.variable.value,
        )
        self.variable.update(loss)
        return loss
