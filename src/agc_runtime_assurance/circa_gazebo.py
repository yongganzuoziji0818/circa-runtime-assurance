"""Fail-closed primitives for prospective CIRCA Gazebo validation.

This module contains no experiment launcher.  It separates the observable
active-assurance record from the shadow/no-override oracle record, derives
registered witnesses without reading the oracle outcome, and compiles a
family-level CIRCA certificate only when the provenance contract is complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

try:  # package import in production
    from .circa import (
        INEVITABLE_VIOLATION,
        INVALID_PROVENANCE,
        INVALID_WITNESS,
        NOMINAL_SAFETY,
        NO_WITNESS,
        OBSERVED_CONSISTENCY,
        VALID,
        CircaError,
        identification_interval,
        manski_bounds,
        simultaneous_confidence_interval,
        structured_bounds,
        verify_evidence_contract,
    )
except ImportError:  # direct-module import for dependency-light G0 checks
    from circa import (  # type: ignore
        INEVITABLE_VIOLATION,
        INVALID_PROVENANCE,
        INVALID_WITNESS,
        NOMINAL_SAFETY,
        NO_WITNESS,
        OBSERVED_CONSISTENCY,
        VALID,
        CircaError,
        identification_interval,
        manski_bounds,
        simultaneous_confidence_interval,
        structured_bounds,
        verify_evidence_contract,
    )


class CircaGazeboError(ValueError):
    """Raised when a Gazebo validation object violates its frozen schema."""


@dataclass(frozen=True)
class GazeboWitnessInput:
    """Verifier-only information used to pin an intervened shadow outcome.

    ``reference_first_violation_seconds`` and
    ``reference_safe_minimum_margin`` are mutually exclusive.  The object
    intentionally has no ``Y0`` field: the oracle shadow outcome is forbidden
    from witness construction.
    """

    intervened: int
    active_first_violation: int
    horizon_seconds: float
    trajectory_state_deviation_bound: float
    constraint_lipschitz: float
    reference_first_violation_seconds: float | None = None
    reference_safe_minimum_margin: float | None = None
    transversality_kappa: float | None = None
    crossing_tube_width: float | None = None

    def validate(self) -> None:
        if self.intervened not in (0, 1) or self.active_first_violation not in (0, 1):
            raise CircaGazeboError("intervention and active outcome must be binary")
        finite_nonnegative = (
            self.horizon_seconds,
            self.trajectory_state_deviation_bound,
            self.constraint_lipschitz,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in finite_nonnegative):
            raise CircaGazeboError("witness horizon and deviation constants must be finite and non-negative")
        if self.horizon_seconds <= 0.0:
            raise CircaGazeboError("witness horizon must be positive")
        crossing = self.reference_first_violation_seconds
        safe_margin = self.reference_safe_minimum_margin
        if crossing is not None and safe_margin is not None:
            raise CircaGazeboError("crossing and safe-censor evidence are mutually exclusive")
        if crossing is not None and (not math.isfinite(crossing) or crossing < 0.0):
            raise CircaGazeboError("reference crossing time must be finite and non-negative")
        if safe_margin is not None and (not math.isfinite(safe_margin) or safe_margin <= 0.0):
            raise CircaGazeboError("safe-censor margin must be positive and finite")


@dataclass(frozen=True)
class WitnessDecision:
    code: int
    reason: str
    barrier_deviation: float
    crossing_time_debit: float | None


def derive_registered_witness(evidence: GazeboWitnessInput) -> WitnessDecision:
    """Derive a sound conditional witness without consulting shadow ``Y0``.

    A crossing witness is issued only when a frozen state-error tube and a
    positive transversality constant imply that the latest possible crossing
    remains inside the horizon.  A nominal-safety witness is issued only when
    the full-horizon reference margin strictly dominates the transported
    barrier deviation.  All incomplete cases retain ``NO_WITNESS``.
    """

    evidence.validate()
    barrier_deviation = evidence.constraint_lipschitz * evidence.trajectory_state_deviation_bound
    if evidence.intervened == 0:
        return WitnessDecision(OBSERVED_CONSISTENCY, "factual_consistency", barrier_deviation, None)

    crossing = evidence.reference_first_violation_seconds
    if crossing is not None:
        kappa = evidence.transversality_kappa
        tube = evidence.crossing_tube_width
        if (
            kappa is None
            or tube is None
            or not math.isfinite(kappa)
            or not math.isfinite(tube)
            or kappa <= 0.0
            or tube <= 0.0
        ):
            return WitnessDecision(NO_WITNESS, "crossing_proof_incomplete", barrier_deviation, None)
        debit = barrier_deviation / kappa
        if debit <= tube and crossing + debit <= evidence.horizon_seconds:
            return WitnessDecision(INEVITABLE_VIOLATION, "robust_crossing_within_horizon", barrier_deviation, debit)
        return WitnessDecision(NO_WITNESS, "crossing_not_robust_within_horizon", barrier_deviation, debit)

    margin = evidence.reference_safe_minimum_margin
    if margin is not None and barrier_deviation < margin:
        return WitnessDecision(NOMINAL_SAFETY, "robustly_safe_through_horizon", barrier_deviation, None)
    return WitnessDecision(NO_WITNESS, "no_sound_registered_witness", barrier_deviation, None)


@dataclass(frozen=True)
class GazeboObservedRecord:
    """Deployable-side record; it deliberately excludes the shadow outcome."""

    scenario_seed: int
    family_id: str
    intervened: int
    active_first_violation: int
    witness_code: int
    outcome_complete: bool
    interference_registered: bool
    policy_sha256: str
    constraint_sha256: str
    witness_model_sha256: str
    interaction_graph_sha256: str
    horizon_steps: int


@dataclass(frozen=True)
class GazeboOracleRecord:
    """Evaluation-only shadow outcome stored outside the CIRCA input table."""

    scenario_seed: int
    family_id: str
    shadow_first_violation: int


@dataclass(frozen=True)
class GazeboRegimeTrace:
    """Complete trace from one member of an active/shadow scenario pair."""

    scenario_seed: int
    family_id: str
    horizon_steps: int
    constraint_margins: tuple[tuple[float, ...], ...]
    applied_actions: tuple[tuple[float, ...], ...]
    outcome_complete: bool

    def validate(self) -> None:
        if not isinstance(self.scenario_seed, int) or self.scenario_seed < 0 or not self.family_id:
            raise CircaGazeboError("trace seed and family must be valid")
        if self.horizon_steps <= 0 or len(self.constraint_margins) != self.horizon_steps:
            raise CircaGazeboError("trace must contain the complete registered horizon")
        if len(self.applied_actions) != self.horizon_steps:
            raise CircaGazeboError("action trace must contain the complete registered horizon")
        first_violation_from_margins(self.constraint_margins)
        actions = np.asarray(self.applied_actions, dtype=float)
        if actions.ndim != 2 or actions.shape[1] == 0 or not np.all(np.isfinite(actions)):
            raise CircaGazeboError("applied action trace must be a finite matrix")


def compile_observed_scenario(
    active_trace: GazeboRegimeTrace,
    nominal_actions: Iterable[Iterable[float]],
    witness_input: GazeboWitnessInput,
    *,
    policy_sha256: str,
    constraint_sha256: str,
    witness_model_sha256: str,
    interaction_graph_sha256: str,
    interference_registered: bool = True,
) -> GazeboObservedRecord:
    """Compile only the deployable-side record; no shadow trace is accepted."""

    active_trace.validate()
    intervention = intervention_from_actions(nominal_actions, active_trace.applied_actions)
    if witness_input.intervened != intervention:
        raise CircaGazeboError("witness intervention flag differs from the active trace")
    y1 = first_violation_from_margins(active_trace.constraint_margins)
    if witness_input.active_first_violation != y1:
        raise CircaGazeboError("witness active outcome differs from the complete trace")
    witness = derive_registered_witness(witness_input)
    return GazeboObservedRecord(
        scenario_seed=active_trace.scenario_seed,
        family_id=active_trace.family_id,
        intervened=intervention,
        active_first_violation=y1,
        witness_code=witness.code,
        outcome_complete=active_trace.outcome_complete,
        interference_registered=interference_registered,
        policy_sha256=policy_sha256,
        constraint_sha256=constraint_sha256,
        witness_model_sha256=witness_model_sha256,
        interaction_graph_sha256=interaction_graph_sha256,
        horizon_steps=active_trace.horizon_steps,
    )


def compile_oracle_scenario(shadow_trace: GazeboRegimeTrace) -> GazeboOracleRecord:
    """Compile the evaluation-only shadow table in a separate call path."""

    shadow_trace.validate()
    if not shadow_trace.outcome_complete:
        raise CircaGazeboError("incomplete shadow trace cannot be used as an oracle outcome")
    return GazeboOracleRecord(
        scenario_seed=shadow_trace.scenario_seed,
        family_id=shadow_trace.family_id,
        shadow_first_violation=first_violation_from_margins(shadow_trace.constraint_margins),
    )


@dataclass(frozen=True)
class GazeboEvidenceContract:
    family_ids: tuple[str, ...]
    policy_sha256: str
    constraint_sha256: str
    witness_model_sha256: str
    interaction_graph_sha256: str
    horizon_steps: int
    alpha: float = 0.05
    endpoint_count: int = 12

    def validate(self) -> None:
        if not self.family_ids or len(set(self.family_ids)) != len(self.family_ids):
            raise CircaGazeboError("registered family IDs must be non-empty and unique")
        hashes = (
            self.policy_sha256,
            self.constraint_sha256,
            self.witness_model_sha256,
            self.interaction_graph_sha256,
        )
        if any(len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()) for value in hashes):
            raise CircaGazeboError("contract hashes must be lowercase SHA-256 strings")
        if self.horizon_steps <= 0 or not (0.0 < self.alpha < 1.0) or self.endpoint_count <= 0:
            raise CircaGazeboError("invalid horizon or confidence configuration")


def _refusal(status: str, family_id: str, n: int) -> dict[str, object]:
    return {
        "family_id": family_id,
        "n": int(n),
        "status": status,
        "numeric_certificate_allowed": False,
        "manski_identification_interval": None,
        "circa_identification_interval": None,
        "circa_simultaneous_confidence_interval": None,
    }


def evaluate_observed_family(
    records: Sequence[GazeboObservedRecord],
    contract: GazeboEvidenceContract,
) -> dict[str, object]:
    """Compile a family certificate or return a typed non-numeric refusal."""

    contract.validate()
    if not records:
        raise CircaGazeboError("a family requires at least one independent scenario")
    family_id = records[0].family_id
    if family_id not in contract.family_ids or any(record.family_id != family_id for record in records):
        return _refusal(INVALID_PROVENANCE, family_id, len(records))
    seeds = [record.scenario_seed for record in records]
    if any(not isinstance(seed, int) or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        return _refusal(INVALID_PROVENANCE, family_id, len(records))

    for record in records:
        check = verify_evidence_contract(
            interference_registered=record.interference_registered
            and record.interaction_graph_sha256 == contract.interaction_graph_sha256,
            outcome_complete=record.outcome_complete,
            witness_hash_matches=record.witness_model_sha256 == contract.witness_model_sha256,
            policy_hash_matches=record.policy_sha256 == contract.policy_sha256,
            constraint_hash_matches=record.constraint_sha256 == contract.constraint_sha256,
            horizon_matches=record.horizon_steps == contract.horizon_steps,
        )
        if not check.numeric_certificate_allowed:
            return _refusal(check.status, family_id, len(records))

    r = np.asarray([record.intervened for record in records], dtype=np.int8)
    y1 = np.asarray([record.active_first_violation for record in records], dtype=np.int8)
    witness = np.asarray([record.witness_code for record in records], dtype=np.int8)
    try:
        manski_lower_y0, manski_upper_y0 = manski_bounds(r, y1)
        circa_lower_y0, circa_upper_y0 = structured_bounds(r, y1, witness)
        manski_interval = identification_interval(manski_lower_y0, manski_upper_y0, y1)
        circa_interval = identification_interval(circa_lower_y0, circa_upper_y0, y1)
        confidence = simultaneous_confidence_interval(
            circa_interval[0],
            circa_interval[1],
            len(records),
            contract.alpha,
            contract.endpoint_count,
        )
    except (CircaError, ValueError):
        return _refusal(INVALID_WITNESS, family_id, len(records))

    manski_width = manski_interval[1] - manski_interval[0]
    circa_width = circa_interval[1] - circa_interval[0]
    return {
        "family_id": family_id,
        "n": len(records),
        "status": VALID,
        "numeric_certificate_allowed": True,
        "manski_identification_interval": list(manski_interval),
        "circa_identification_interval": list(circa_interval),
        "circa_simultaneous_confidence_interval": list(confidence),
        "identification_width_ratio": None if manski_width == 0.0 else circa_width / manski_width,
        "intervention_rate": float(np.mean(r)),
        "active_first_violation_rate": float(np.mean(y1)),
        "witness_pinning_rate_among_interventions": (
            float(np.mean(witness[r == 1] != NO_WITNESS)) if np.any(r == 1) else 0.0
        ),
    }


def attach_oracle_audit(
    observed: Sequence[GazeboObservedRecord],
    oracle: Sequence[GazeboOracleRecord],
) -> dict[str, object]:
    """Audit witness soundness and paired sample truth outside estimation."""

    observed_keys = [(item.family_id, item.scenario_seed) for item in observed]
    oracle_keys = [(item.family_id, item.scenario_seed) for item in oracle]
    if len(set(observed_keys)) != len(observed_keys) or observed_keys != oracle_keys:
        raise CircaGazeboError("observed and oracle records must be uniquely paired in identical order")
    y0 = np.asarray([item.shadow_first_violation for item in oracle], dtype=np.int8)
    y1 = np.asarray([item.active_first_violation for item in observed], dtype=np.int8)
    if not np.all(np.isin(y0, [0, 1])):
        raise CircaGazeboError("oracle shadow outcomes must be binary")
    witness = np.asarray([item.witness_code for item in observed], dtype=np.int8)
    inevitable_sound = bool(np.all(y0[witness == INEVITABLE_VIOLATION] == 1))
    nominal_safe_sound = bool(np.all(y0[witness == NOMINAL_SAFETY] == 0))
    return {
        "paired_oracle_sample_delta": float(np.mean(y0 - y1)),
        "inevitable_witness_count": int(np.sum(witness == INEVITABLE_VIOLATION)),
        "nominal_safety_witness_count": int(np.sum(witness == NOMINAL_SAFETY)),
        "inevitable_witness_sound": inevitable_sound,
        "nominal_safety_witness_sound": nominal_safe_sound,
        "all_registered_witnesses_sound_on_paired_gazebo_sample": inevitable_sound and nominal_safe_sound,
    }


def first_violation_from_margins(margins: Iterable[Iterable[float]]) -> int:
    values = np.asarray(margins, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0 or not np.all(np.isfinite(values)):
        raise CircaGazeboError("constraint-margin trace must be a non-empty finite matrix")
    return int(np.any(values < 0.0))


def intervention_from_actions(
    nominal_actions: Iterable[Iterable[float]],
    active_actions: Iterable[Iterable[float]],
    *,
    atol: float = 1e-12,
) -> int:
    nominal = np.asarray(nominal_actions, dtype=float)
    active = np.asarray(active_actions, dtype=float)
    if nominal.shape != active.shape or nominal.ndim != 2 or nominal.shape[0] == 0:
        raise CircaGazeboError("nominal and active action traces must be non-empty and shape matched")
    if not np.all(np.isfinite(nominal)) or not np.all(np.isfinite(active)) or atol < 0.0:
        raise CircaGazeboError("action traces and tolerance must be finite")
    return int(np.any(np.abs(nominal - active) > atol))


def derive_scenario_seed(master_seed: int, family_id: str, index: int) -> int:
    if not isinstance(master_seed, int) or master_seed < 0 or not family_id or index < 0:
        raise CircaGazeboError("invalid deterministic seed derivation input")
    digest = sha256(f"circa-gz1|{master_seed}|{family_id}|{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def schema_capacity_summary(
    *,
    families: Sequence[str],
    scenarios_per_family: int,
    endpoint_count: int,
    alpha: float,
    estimated_bytes_per_paired_scenario: int,
) -> dict[str, object]:
    if not families or len(set(families)) != len(families):
        raise CircaGazeboError("families must be non-empty and unique")
    if scenarios_per_family <= 0 or endpoint_count <= 0 or estimated_bytes_per_paired_scenario <= 0:
        raise CircaGazeboError("capacity counts must be positive")
    if not 0.0 < alpha < 1.0:
        raise CircaGazeboError("alpha must lie in (0, 1)")
    radius = 2.0 * math.sqrt(math.log(endpoint_count / alpha) / (2.0 * scenarios_per_family))
    paired_scenarios = len(families) * scenarios_per_family
    return {
        "families": len(families),
        "scenarios_per_family": scenarios_per_family,
        "independent_paired_scenarios": paired_scenarios,
        "gazebo_regime_rollouts": 2 * paired_scenarios,
        "simultaneous_hoeffding_radius": radius,
        "estimated_output_bytes": paired_scenarios * estimated_bytes_per_paired_scenario,
    }
