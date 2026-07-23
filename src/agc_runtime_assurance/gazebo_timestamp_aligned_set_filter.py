"""Timestamp-aligned set-propagation backup filter for CIRCA-GZ0-v8.

This module is a claim-ineligible validation instrument.  It never generates
scientific seeds or outputs.  The safety object is a center-radius state set
propagated from asynchronous source timestamps to the current decision time
using recorded *applied* actions and prospectively frozen global error bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Sequence

import numpy as np

from .gazebo_robust_backup_filter import (
    BackupPlanEvaluation,
    GazeboPlanarPlant,
    RobustBackupConfig,
    RobustBackupSafetyFilter,
)
from .gazebo_second_system import UAV_QUADRATIC_DRAG, UGV_QUADRATIC_DRAG


class TimestampAlignmentError(ValueError):
    """Raised when the registered timestamp/provenance contract is invalid."""


@dataclass(frozen=True)
class TimestampAlignmentConfig:
    per_agent_position_error_bound_m: float = 0.02
    per_agent_velocity_error_bound_mps: float = 0.02
    relative_acceleration_error_bound_mps2: float = 0.05
    common_mode_position_bias_bound_m: float = 0.15
    maximum_observation_age_steps: int = 3
    maximum_additional_neighbor_communication_age_steps: int = 3
    tolerance: float = 1e-12

    def validate(self) -> None:
        values = np.asarray(
            [
                self.per_agent_position_error_bound_m,
                self.per_agent_velocity_error_bound_mps,
                self.relative_acceleration_error_bound_mps2,
                self.common_mode_position_bias_bound_m,
                self.tolerance,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("alignment bounds must be finite and non-negative")
        if not isinstance(self.maximum_observation_age_steps, int) or not isinstance(
            self.maximum_additional_neighbor_communication_age_steps, int
        ):
            raise ValueError("maximum ages must be integers")
        if self.maximum_observation_age_steps < 0 or (
            self.maximum_additional_neighbor_communication_age_steps < 0
        ):
            raise ValueError("maximum ages must be non-negative")

    def registry_payload(self) -> dict[str, Any]:
        return {
            "per_agent_position_error_bound_m": self.per_agent_position_error_bound_m,
            "per_agent_velocity_error_bound_mps": self.per_agent_velocity_error_bound_mps,
            "relative_acceleration_error_bound_mps2": self.relative_acceleration_error_bound_mps2,
            "common_mode_position_bias_bound_m": self.common_mode_position_bias_bound_m,
            "maximum_observation_age_steps": self.maximum_observation_age_steps,
            "maximum_additional_neighbor_communication_age_steps": (
                self.maximum_additional_neighbor_communication_age_steps
            ),
        }


@dataclass(frozen=True)
class AlignedStateSet:
    center: np.ndarray
    radius: np.ndarray
    local_source_step: int
    neighbor_source_step: int
    decision_step: int
    local_age_steps: int
    neighbor_age_steps: int
    common_mode_position_bias_bound_m: float
    applied_action_history_digest: str
    uncertainty_registry_hash: str
    provenance_hash: str

    def validate(self) -> None:
        center = np.asarray(self.center, dtype=float).reshape(-1)
        radius = np.asarray(self.radius, dtype=float).reshape(-1)
        if center.shape != (10,) or radius.shape != (10,):
            raise TimestampAlignmentError("aligned center/radius must have shape (10,)")
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(radius)):
            raise TimestampAlignmentError("aligned center/radius must be finite")
        if np.any(radius < 0.0):
            raise TimestampAlignmentError("aligned radii must be non-negative")
        if not 0 <= self.local_source_step <= self.decision_step:
            raise TimestampAlignmentError("invalid local source step")
        if not 0 <= self.neighbor_source_step <= self.decision_step:
            raise TimestampAlignmentError("invalid neighbor source step")
        if self.local_age_steps != self.decision_step - self.local_source_step:
            raise TimestampAlignmentError("local age/source mismatch")
        if self.neighbor_age_steps != self.decision_step - self.neighbor_source_step:
            raise TimestampAlignmentError("neighbor age/source mismatch")
        for digest in (
            self.applied_action_history_digest,
            self.uncertainty_registry_hash,
            self.provenance_hash,
        ):
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise TimestampAlignmentError("invalid evidence digest")


@dataclass(frozen=True)
class SetBackupDecision:
    action: np.ndarray
    intervened: bool
    fail_closed: bool
    evidence_valid: bool
    certificate_emitted: bool
    reason: str
    aligned_state: AlignedStateSet | None
    nominal_plan: BackupPlanEvaluation | None
    backup_plan: BackupPlanEvaluation | None
    certificate: dict[str, Any] | None


def _canonical_hash(payload: Any) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _as_state_history(values: Sequence[np.ndarray]) -> list[np.ndarray]:
    output = [np.asarray(value, dtype=float).reshape(-1).copy() for value in values]
    if not output or any(value.shape != (10,) for value in output):
        raise TimestampAlignmentError("state history must contain shape-(10,) states")
    if any(not np.all(np.isfinite(value)) for value in output):
        raise TimestampAlignmentError("state history must be finite")
    return output


def _as_action_history(values: Sequence[np.ndarray], expected: int) -> list[np.ndarray]:
    output = [np.asarray(value, dtype=float).reshape(-1).copy() for value in values]
    if len(output) != expected or any(value.shape != (5,) for value in output):
        raise TimestampAlignmentError(
            "applied-action history must align one-to-one with state history"
        )
    if any(not np.all(np.isfinite(value)) for value in output):
        raise TimestampAlignmentError("applied-action history must be finite")
    return output


def _project_planar(vector: np.ndarray, limit: float | None) -> np.ndarray:
    output = np.asarray(vector, dtype=float).reshape(2).copy()
    if limit is not None:
        norm = float(np.linalg.norm(output))
        if norm > limit:
            output *= limit / norm
    return output


def _propagate_agent_with_applied_action(
    position: np.ndarray,
    velocity: np.ndarray,
    position_radius: np.ndarray,
    velocity_radius: np.ndarray,
    applied_action: np.ndarray,
    plant: GazeboPlanarPlant,
    alignment: TimestampAlignmentConfig,
    *,
    agent: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Conservatively propagate one planar agent's center-radius set one step."""

    plant.validate()
    alignment.validate()
    p = np.asarray(position, dtype=float).reshape(2)
    v = np.asarray(velocity, dtype=float).reshape(2)
    rp = np.asarray(position_radius, dtype=float).reshape(2)
    rv = np.asarray(velocity_radius, dtype=float).reshape(2)
    action = np.asarray(applied_action, dtype=float).reshape(2)
    if not all(np.all(np.isfinite(x)) for x in (p, v, rp, rv, action)):
        raise TimestampAlignmentError("agent set and applied action must be finite")
    if np.any(rp < 0.0) or np.any(rv < 0.0):
        raise TimestampAlignmentError("agent radii must be non-negative")

    if agent == "uav":
        acceleration = (
            action
            - plant.uav_drag * v
            - UAV_QUADRATIC_DRAG * np.linalg.norm(v) * v
        ) / plant.uav_mass
        linear_drag = plant.uav_drag / plant.uav_mass
        quadratic_drag = UAV_QUADRATIC_DRAG / plant.uav_mass
    elif agent == "ugv":
        acceleration = action / (1.0 + plant.ugv_friction) - (
            UGV_QUADRATIC_DRAG * np.linalg.norm(v) * v
        )
        linear_drag = 0.0
        quadratic_drag = UGV_QUADRATIC_DRAG
    else:
        raise ValueError("agent must be 'uav' or 'ugv'")

    unprojected_velocity = v + plant.dt * acceleration
    next_velocity = _project_planar(unprojected_velocity, plant.speed_limit_mps)

    delta_v_2 = float(np.linalg.norm(rv))
    maximum_velocity_2 = float(np.linalg.norm(v)) + delta_v_2
    lipschitz = 1.0 + plant.dt * (
        linear_drag + 2.0 * quadratic_drag * maximum_velocity_2
    )
    per_agent_acceleration_error = 0.5 * (
        alignment.relative_acceleration_error_bound_mps2
    )
    velocity_error_2 = lipschitz * delta_v_2 + (
        plant.dt * per_agent_acceleration_error
    )
    # Euclidean projection onto a speed ball is non-expansive.  A 2-norm error
    # bound is therefore a valid per-axis box radius after projection.
    next_velocity_radius = np.full(2, velocity_error_2, dtype=float)
    next_position = p + plant.dt * next_velocity
    next_position_radius = rp + plant.dt * next_velocity_radius
    return next_position, next_velocity, next_position_radius, next_velocity_radius


def _propagate_component_to_step(
    center: np.ndarray,
    radius: np.ndarray,
    applied: Sequence[np.ndarray],
    source_step: int,
    decision_step: int,
    plant: GazeboPlanarPlant,
    alignment: TimestampAlignmentConfig,
    *,
    agent: str,
) -> None:
    if agent == "uav":
        pos, vel, action_slice = slice(0, 2), slice(3, 5), slice(0, 2)
    elif agent == "ugv":
        pos, vel, action_slice = slice(6, 8), slice(8, 10), slice(3, 5)
    else:
        raise ValueError("unknown component")
    for step in range(source_step + 1, decision_step + 1):
        p, v, rp, rv = _propagate_agent_with_applied_action(
            center[pos],
            center[vel],
            radius[pos],
            radius[vel],
            applied[step][action_slice],
            plant,
            alignment,
            agent=agent,
        )
        center[pos], center[vel] = p, v
        radius[pos], radius[vel] = rp, rv


def align_async_state_history(
    state_history: Sequence[np.ndarray],
    applied_action_history: Sequence[np.ndarray],
    observation_delay_steps: int,
    communication_delay_steps: int,
    plant: GazeboPlanarPlant,
    alignment: TimestampAlignmentConfig,
    *,
    observed_common_mode_bias_m: float = 0.0,
) -> AlignedStateSet:
    """Align local and neighbor observations to the latest history step.

    The supplied histories are fixture/runtime buffers, not scientific data.  The
    action at index ``j`` is the action actually applied on the transition into
    state ``j``.  Early-episode source indices are clamped to zero and the emitted
    age records the resulting effective age.
    """

    plant.validate()
    alignment.validate()
    states = _as_state_history(state_history)
    actions = _as_action_history(applied_action_history, len(states))
    for name, value, maximum in (
        ("observation", observation_delay_steps, alignment.maximum_observation_age_steps),
        (
            "communication",
            communication_delay_steps,
            alignment.maximum_additional_neighbor_communication_age_steps,
        ),
    ):
        if not isinstance(value, int) or value < 0 or value > maximum:
            raise TimestampAlignmentError(f"{name} age exceeds registered bounds")
    if not np.isfinite(observed_common_mode_bias_m) or abs(observed_common_mode_bias_m) > (
        alignment.common_mode_position_bias_bound_m + alignment.tolerance
    ):
        raise TimestampAlignmentError("observed common-mode bias exceeds registry")

    decision_step = len(states) - 1
    local_source = max(0, decision_step - observation_delay_steps)
    neighbor_source = max(
        0, decision_step - observation_delay_steps - communication_delay_steps
    )
    center = states[local_source].copy()
    center[6:10] = states[neighbor_source][6:10]
    center[[0, 1, 6, 7]] += observed_common_mode_bias_m
    radius = np.zeros(10, dtype=float)
    radius[[0, 1, 6, 7]] = alignment.per_agent_position_error_bound_m
    radius[[3, 4, 8, 9]] = alignment.per_agent_velocity_error_bound_mps

    _propagate_component_to_step(
        center,
        radius,
        actions,
        local_source,
        decision_step,
        plant,
        alignment,
        agent="uav",
    )
    _propagate_component_to_step(
        center,
        radius,
        actions,
        neighbor_source,
        decision_step,
        plant,
        alignment,
        agent="ugv",
    )

    history_start = min(local_source, neighbor_source) + 1
    action_payload = [actions[index].tolist() for index in range(history_start, decision_step + 1)]
    action_digest = _canonical_hash(
        {"start_step": history_start, "decision_step": decision_step, "actions": action_payload}
    )
    registry_hash = _canonical_hash(alignment.registry_payload())
    provenance = {
        "local_source_step": local_source,
        "neighbor_source_step": neighbor_source,
        "decision_step": decision_step,
        "center": center.tolist(),
        "radius": radius.tolist(),
        "applied_action_history_digest": action_digest,
        "uncertainty_registry_hash": registry_hash,
    }
    output = AlignedStateSet(
        center=center,
        radius=radius,
        local_source_step=local_source,
        neighbor_source_step=neighbor_source,
        decision_step=decision_step,
        local_age_steps=decision_step - local_source,
        neighbor_age_steps=decision_step - neighbor_source,
        common_mode_position_bias_bound_m=alignment.common_mode_position_bias_bound_m,
        applied_action_history_digest=action_digest,
        uncertainty_registry_hash=registry_hash,
        provenance_hash=_canonical_hash(provenance),
    )
    output.validate()
    return output


class TimestampAlignedSetBackupFilter:
    """Predictive backup filter evaluated on a common-time center-radius set."""

    def __init__(
        self,
        plant: GazeboPlanarPlant,
        backup: RobustBackupConfig,
        alignment: TimestampAlignmentConfig,
    ):
        plant.validate()
        backup.validate()
        alignment.validate()
        self.plant = plant
        self.backup = backup
        self.alignment = alignment
        self._point_filter = RobustBackupSafetyFilter(plant, backup)

    def backup_action(
        self, center: np.ndarray, nominal_action: np.ndarray | None = None
    ) -> np.ndarray:
        return self._point_filter.backup_action(center, nominal_action)

    def _set_margin(self, center: np.ndarray, radius: np.ndarray) -> float:
        relative_center = center[:2] - center[6:8]
        relative_radius = radius[:2] + radius[6:8]
        separation_lower = max(
            0.0, float(np.linalg.norm(relative_center) - np.linalg.norm(relative_radius))
        )
        return separation_lower - self.backup.operational_separation_m

    def _radial_rate_lower(self, center: np.ndarray, radius: np.ndarray) -> float:
        relative_position = center[:2] - center[6:8]
        relative_position_radius = radius[:2] + radius[6:8]
        relative_velocity = center[3:5] - center[8:10]
        relative_velocity_radius = radius[3:5] + radius[8:10]
        center_separation = float(np.linalg.norm(relative_position))
        position_error = float(np.linalg.norm(relative_position_radius))
        separation_lower = center_separation - position_error
        if center_separation <= self.backup.tolerance or separation_lower <= 0.0:
            return float("-inf")
        center_rate = float(
            relative_position @ relative_velocity / center_separation
        )
        velocity_error = float(np.linalg.norm(relative_velocity_radius))
        direction_error = min(2.0, 2.0 * position_error / separation_lower)
        return center_rate - velocity_error - direction_error * (
            float(np.linalg.norm(relative_velocity)) + velocity_error
        )

    def _propagate_requested(
        self,
        center: np.ndarray,
        radius: np.ndarray,
        previous_applied: np.ndarray,
        requested: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        command = np.clip(
            np.asarray(requested, dtype=float).reshape(5),
            -self.backup.action_limit,
            self.backup.action_limit,
        )
        prior = np.asarray(previous_applied, dtype=float).reshape(5)
        applied = self.plant.actuator_lag * prior + (
            1.0 - self.plant.actuator_lag
        ) * command
        next_center = np.asarray(center, dtype=float).reshape(10).copy()
        next_radius = np.asarray(radius, dtype=float).reshape(10).copy()
        for agent, pos, vel, action_slice in (
            ("uav", slice(0, 2), slice(3, 5), slice(0, 2)),
            ("ugv", slice(6, 8), slice(8, 10), slice(3, 5)),
        ):
            p, v, rp, rv = _propagate_agent_with_applied_action(
                next_center[pos],
                next_center[vel],
                next_radius[pos],
                next_radius[vel],
                applied[action_slice],
                self.plant,
                self.alignment,
                agent=agent,
            )
            next_center[pos], next_center[vel] = p, v
            next_radius[pos], next_radius[vel] = rp, rv
        return next_center, next_radius, applied

    def evaluate_plan(
        self,
        aligned: AlignedStateSet,
        previous_applied: np.ndarray,
        first_action: np.ndarray,
        *,
        point_ablation: bool = False,
    ) -> BackupPlanEvaluation:
        aligned.validate()
        center = np.asarray(aligned.center, dtype=float).copy()
        radius = np.zeros(10, dtype=float) if point_ablation else np.asarray(
            aligned.radius, dtype=float
        ).copy()
        applied = np.asarray(previous_applied, dtype=float).reshape(-1).copy()
        first = np.asarray(first_action, dtype=float).reshape(-1).copy()
        if applied.shape != (5,) or first.shape != (5,):
            raise ValueError("plan actions must have shape (5,)")
        if not np.all(np.isfinite(applied)) or not np.all(np.isfinite(first)):
            raise ValueError("plan actions must be finite")

        previous_margin = self._set_margin(center, radius)
        minimum_margin = previous_margin
        minimum_residual = float("inf")
        if previous_margin < -self.backup.tolerance:
            return BackupPlanEvaluation(
                False,
                False,
                previous_margin,
                float("-inf"),
                None,
                0,
                "initial_aligned_set_operational_constraint_violated",
            )
        for step in range(self.backup.horizon_steps):
            command = first if step == 0 else self.backup_action(center)
            center, radius, applied = self._propagate_requested(
                center, radius, applied, command
            )
            margin = self._set_margin(center, radius)
            residual = margin - self.backup.barrier_retention * previous_margin
            minimum_margin = min(minimum_margin, margin)
            minimum_residual = min(minimum_residual, residual)
            if margin < -self.backup.tolerance:
                return BackupPlanEvaluation(
                    False,
                    False,
                    minimum_margin,
                    minimum_residual,
                    None,
                    step + 1,
                    f"aligned_set_operational_constraint_failed_at_step_{step + 1}",
                )
            if residual < -self.backup.tolerance:
                return BackupPlanEvaluation(
                    False,
                    False,
                    minimum_margin,
                    minimum_residual,
                    None,
                    step + 1,
                    f"aligned_set_discrete_barrier_failed_at_step_{step + 1}",
                )
            if (
                margin >= self.backup.terminal_margin_m
                and self._radial_rate_lower(center, radius) >= 0.0
            ):
                return BackupPlanEvaluation(
                    True,
                    True,
                    minimum_margin,
                    minimum_residual,
                    step + 1,
                    step + 1,
                    "terminal_robust_nonclosing_set_reached",
                )
            previous_margin = margin
        return BackupPlanEvaluation(
            False,
            False,
            minimum_margin,
            minimum_residual,
            None,
            self.backup.horizon_steps,
            "terminal_set_not_reached_within_horizon",
        )

    def decide(
        self,
        aligned: AlignedStateSet,
        previous_applied: np.ndarray,
        nominal_action: np.ndarray,
        *,
        point_ablation: bool = False,
    ) -> SetBackupDecision:
        aligned.validate()
        nominal = np.asarray(nominal_action, dtype=float).reshape(-1)
        if nominal.shape != (5,) or not np.all(np.isfinite(nominal)):
            raise ValueError("nominal action must be finite with shape (5,)")
        nominal = np.clip(nominal, -self.backup.action_limit, self.backup.action_limit)
        backup = self.backup_action(aligned.center, nominal)
        nominal_plan = self.evaluate_plan(
            aligned, previous_applied, nominal, point_ablation=point_ablation
        )
        backup_plan = self.evaluate_plan(
            aligned, previous_applied, backup, point_ablation=point_ablation
        )
        if nominal_plan.feasible:
            action, intervened, fail_closed, reason = (
                nominal,
                False,
                False,
                "nominal_with_verified_aligned_set_backup_tube",
            )
        elif backup_plan.feasible:
            action, intervened, fail_closed, reason = (
                backup,
                True,
                False,
                "aligned_set_backup_filter_intervention",
            )
        else:
            action, intervened, fail_closed, reason = (
                backup,
                True,
                True,
                "aligned_set_backup_tube_infeasible_fail_closed",
            )
        certificate = {
            "source_steps": {
                "local": aligned.local_source_step,
                "neighbor": aligned.neighbor_source_step,
                "decision": aligned.decision_step,
            },
            "ages": {
                "local": aligned.local_age_steps,
                "neighbor": aligned.neighbor_age_steps,
            },
            "applied_action_history_digest": aligned.applied_action_history_digest,
            "uncertainty_registry_hash": aligned.uncertainty_registry_hash,
            "provenance_hash": aligned.provenance_hash,
            "point_ablation": point_ablation,
            "nominal_feasible": nominal_plan.feasible,
            "backup_feasible": backup_plan.feasible,
            "nominal_minimum_margin_m": nominal_plan.minimum_tightened_margin_m,
            "backup_minimum_margin_m": backup_plan.minimum_tightened_margin_m,
            "reason": reason,
            "validity_interval_steps": [aligned.decision_step, aligned.decision_step],
        }
        return SetBackupDecision(
            action=action,
            intervened=intervened,
            fail_closed=fail_closed,
            evidence_valid=True,
            certificate_emitted=True,
            reason=reason,
            aligned_state=aligned,
            nominal_plan=nominal_plan,
            backup_plan=backup_plan,
            certificate=certificate,
        )

    def align_and_decide(
        self,
        state_history: Sequence[np.ndarray],
        applied_action_history: Sequence[np.ndarray],
        observation_delay_steps: int,
        communication_delay_steps: int,
        nominal_action: np.ndarray,
        *,
        observed_common_mode_bias_m: float = 0.0,
        point_ablation: bool = False,
    ) -> SetBackupDecision:
        try:
            aligned = align_async_state_history(
                state_history,
                applied_action_history,
                observation_delay_steps,
                communication_delay_steps,
                self.plant,
                self.alignment,
                observed_common_mode_bias_m=observed_common_mode_bias_m,
            )
        except (TimestampAlignmentError, ValueError) as error:
            # Malformed or absent state evidence cannot safely parameterize a
            # model-based backup.  Use a bounded zero-action emergency stop as
            # the deterministic fail-closed fallback; valid-but-stale evidence
            # may still use the state-dependent backup controller.
            try:
                latest = _as_state_history(state_history)[-1]
                nominal = np.asarray(nominal_action, dtype=float).reshape(-1)
                backup = self.backup_action(
                    latest,
                    nominal
                    if nominal.shape == (5,) and np.all(np.isfinite(nominal))
                    else None,
                )
            except (TimestampAlignmentError, ValueError):
                backup = np.zeros(5, dtype=float)
            return SetBackupDecision(
                action=backup,
                intervened=True,
                fail_closed=True,
                evidence_valid=False,
                certificate_emitted=False,
                reason=f"alignment_evidence_refused:{type(error).__name__}",
                aligned_state=None,
                nominal_plan=None,
                backup_plan=None,
                certificate=None,
            )
        previous_applied = np.asarray(applied_action_history[-1], dtype=float)
        return self.decide(
            aligned,
            previous_applied,
            nominal_action,
            point_ablation=point_ablation,
        )


__all__ = [
    "AlignedStateSet",
    "SetBackupDecision",
    "TimestampAlignedSetBackupFilter",
    "TimestampAlignmentConfig",
    "TimestampAlignmentError",
    "align_async_state_history",
]
