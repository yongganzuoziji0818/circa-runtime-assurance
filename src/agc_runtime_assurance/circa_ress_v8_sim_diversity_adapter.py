"""Independent, non-running adapter for CIRCA-RESS-V8-SIM-DIVERSITY-R1.

The module performs frozen-design validation, returns deterministic world patch
data, and translates contract-valid histories into timestamp-aligned decisions.
It deliberately has no Gazebo import, process launch, random generator, seed
derivation, scientific schedule, or output writer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from .contracts import ActionEnvelope, ContractError, ExpiredActionError
from .public_timestamp_aligned_filter_primitives import (
    GazeboPlanarPlant,
    RobustBackupConfig,
    SetBackupDecision,
    TimestampAlignedSetBackupFilter,
    TimestampAlignmentConfig,
)


FROZEN_ROUTE_ID = "CIRCA-RESS-V8-SIM-DIVERSITY-R1"
FROZEN_DESIGN_SHA256 = "f84325ab6d901d2f03f37f2e6b34ebba570d513c8f8b8f29a4581bf9d363aaa9"
FROZEN_FAMILIES = ("SDF1", "SDF2", "SDF3", "SDF4", "SDF5")
FROZEN_CANDIDATES = ("A", "B")
REQUIRED_PATCHABLE_MODELS = (
    "sdf1_occluder",
    "sdf1_corridor_left",
    "sdf1_corridor_right",
    "sdf2_ramp",
    "sdf4_corridor_left",
    "sdf4_corridor_right",
)
PARKED_POSE = (0.0, 0.0, -1000.0, 0.0, 0.0, 0.0)


class DiversityAdapterError(RuntimeError):
    """Raised when a frozen design, world, or adapter input is invalid."""


@dataclass(frozen=True)
class ScenarioSpec:
    family_id: str
    candidate_label: str
    role: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class ModelPatch:
    model_name: str
    pose: tuple[float, float, float, float, float, float]
    size: tuple[float, float, float] | None = None
    friction: float | None = None


@dataclass(frozen=True)
class WorldPatch:
    scenario: ScenarioSpec
    model_patches: tuple[ModelPatch, ...]
    observation_delay_steps: int
    communication_delay_steps: int
    packet_loss: float
    sensor_bias_m: float
    timestamp_jitter_steps: int
    actuation_lag_steps: int
    lateral_disturbance_mps2: float
    registered_faults: tuple[str, ...]


@dataclass(frozen=True)
class AdapterDecision:
    envelope: ActionEnvelope
    intervened: bool
    fail_closed: bool
    evidence_valid: bool
    certificate_emitted: bool
    refusal_code: str
    reason: str
    decision_digest: str


def _canonical_digest(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_design(path: str | Path) -> dict[str, Any]:
    design_path = Path(path)
    if _file_sha256(design_path) != FROZEN_DESIGN_SHA256:
        raise DiversityAdapterError("frozen design hash mismatch")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if design.get("route_id") != FROZEN_ROUTE_ID:
        raise DiversityAdapterError("frozen route id mismatch")
    if design.get("status") != "FROZEN_DESIGN_PROPOSAL_AWAITING_HUMAN_CONFIRMATION":
        raise DiversityAdapterError("unexpected frozen design status")
    families = tuple(item.get("id") for item in design.get("families", ()))
    if families != FROZEN_FAMILIES:
        raise DiversityAdapterError("family order drifted")
    if design.get("authorization", {}).get("simulator_or_scientific_runner_authorized"):
        raise DiversityAdapterError("design-only manifest cannot authorize execution")
    if design.get("authorization", {}).get("scientific_seed_generation_authorized"):
        raise DiversityAdapterError("design-only manifest cannot contain seed authority")
    return design


def scenario_from_design(
    design: Mapping[str, Any], family_id: str, candidate_label: str
) -> ScenarioSpec:
    if family_id not in FROZEN_FAMILIES or candidate_label not in FROZEN_CANDIDATES:
        raise DiversityAdapterError("unknown frozen family or candidate")
    family = next(item for item in design["families"] if item["id"] == family_id)
    parameters = family[f"candidate_{candidate_label}"]
    return ScenarioSpec(
        family_id=family_id,
        candidate_label=candidate_label,
        role=str(family["role"]),
        parameters=dict(parameters),
    )


def validate_world_semantics(path: str | Path) -> dict[str, Any]:
    world_path = Path(path)
    body = world_path.read_text(encoding="utf-8")
    lowered = body.lower()
    if any(token in lowered for token in ("http://", "https://", "fuel://", "model://")):
        raise DiversityAdapterError("world must be self-contained")
    root = ET.fromstring(body)
    world = root.find("world")
    if world is None or world.attrib.get("name") != "circa_ress_v8_sim_diversity_r1":
        raise DiversityAdapterError("world name drifted")
    gravity = (world.findtext("gravity") or "").strip()
    if gravity != "0 0 0":
        raise DiversityAdapterError("world gravity must remain zero")
    step = world.findtext("./physics/max_step_size")
    if step is None or abs(float(step) - 0.01) > 1e-12:
        raise DiversityAdapterError("world physics step drifted")
    models = {node.attrib.get("name"): node for node in world.findall("model")}
    missing = {"uav", "ugv", "floor", *REQUIRED_PATCHABLE_MODELS} - set(models)
    if missing:
        raise DiversityAdapterError(f"world is missing required models: {sorted(missing)}")
    dynamic = tuple(
        name
        for name, model in models.items()
        if (model.findtext("static") or "false").strip().lower() != "true"
    )
    if dynamic != ("uav", "ugv"):
        raise DiversityAdapterError("world must contain exactly the frozen dynamic agents")
    for name in REQUIRED_PATCHABLE_MODELS:
        if (models[name].findtext("static") or "").strip().lower() != "true":
            raise DiversityAdapterError(f"patchable fixture is not static: {name}")
        pose = tuple(float(value) for value in (models[name].findtext("pose") or "").split())
        if pose != PARKED_POSE:
            raise DiversityAdapterError(f"patchable fixture is not parked: {name}")
    return {
        "world_sha256": _file_sha256(world_path),
        "dynamic_agents": list(dynamic),
        "static_models": sorted(set(models) - set(dynamic)),
        "patchable_models": list(REQUIRED_PATCHABLE_MODELS),
        "self_contained": True,
        "simulator_invoked": False,
    }


def world_patch_for(scenario: ScenarioSpec) -> WorldPatch:
    params = scenario.parameters
    patches = [
        ModelPatch(name, PARKED_POSE) for name in REQUIRED_PATCHABLE_MODELS
    ]
    observation_delay = int(params.get("observation_delay_steps", 0))
    communication_delay = int(params.get("communication_delay_steps", 0))
    packet_loss = float(params.get("packet_loss", 0.0))
    sensor_bias = float(params.get("sensor_bias_m", 0.0))
    jitter = int(params.get("timestamp_jitter_steps", 0))
    lag = int(params.get("actuation_lag_steps", 0))
    disturbance = float(params.get("lateral_disturbance_mps2", 0.0))
    faults = tuple(str(value) for value in params.get("faults", ()))

    def replace(patch: ModelPatch) -> None:
        index = next(
            i for i, current in enumerate(patches) if current.model_name == patch.model_name
        )
        patches[index] = patch

    if scenario.family_id == "SDF1":
        aisle = float(params["aisle_width_m"])
        length = float(params["occluder_length_m"])
        replace(ModelPatch("sdf1_occluder", (0, 0, 1, 0, 0, 0), (length, 0.20, 2)))
        replace(ModelPatch("sdf1_corridor_left", (0, aisle / 2, 0.60, 0, 0, 0)))
        replace(ModelPatch("sdf1_corridor_right", (0, -aisle / 2, 0.60, 0, 0, 0)))
    elif scenario.family_id == "SDF2":
        slope = np.deg2rad(float(params["slope_degrees"]))
        friction = float(params["friction_transition"][1])
        replace(ModelPatch("sdf2_ramp", (0, 0, 0, 0, slope, 0), friction=friction))
    elif scenario.family_id == "SDF4":
        aisle = float(params["aisle_width_m"])
        replace(ModelPatch("sdf4_corridor_left", (0, aisle / 2, 0.60, 0, 0, 0)))
        replace(ModelPatch("sdf4_corridor_right", (0, -aisle / 2, 0.60, 0, 0, 0)))

    if observation_delay < 0 or communication_delay < 0 or not 0 <= packet_loss <= 1:
        raise DiversityAdapterError("invalid frozen delay or packet-loss value")
    if lag < 0 or jitter < 0 or sensor_bias < 0 or disturbance < 0:
        raise DiversityAdapterError("invalid frozen non-negative stress value")
    return WorldPatch(
        scenario=scenario,
        model_patches=tuple(patches),
        observation_delay_steps=observation_delay,
        communication_delay_steps=communication_delay,
        packet_loss=packet_loss,
        sensor_bias_m=sensor_bias,
        timestamp_jitter_steps=jitter,
        actuation_lag_steps=lag,
        lateral_disturbance_mps2=disturbance,
        registered_faults=faults,
    )


class DiversityRuntimeAdapter:
    """Contract and filter bridge with deterministic typed refusal."""

    def __init__(self, *, actuator_lag: float = 0.0, friction: float = 0.50):
        plant = GazeboPlanarPlant(
            uav_mass=1.0,
            uav_drag=0.1,
            ugv_friction=friction,
            actuator_lag=actuator_lag,
            dt=0.1,
            speed_limit_mps=0.5,
        )
        backup = RobustBackupConfig(
            operational_separation_m=1.0,
            action_limit=0.5,
            horizon_steps=40,
            terminal_margin_m=0.10,
        )
        alignment = TimestampAlignmentConfig()
        self._filter = TimestampAlignedSetBackupFilter(plant, backup, alignment)

    @staticmethod
    def _refusal(
        *,
        issued_at: float,
        now: float,
        reason: str,
        refusal_code: str,
    ) -> AdapterDecision:
        safe_issued_at = max(0.0, float(issued_at))
        safe_now = max(safe_issued_at, float(now))
        envelope = ActionEnvelope(
            action=np.zeros(5, dtype=float),
            issued_at=safe_issued_at,
            valid_until=max(safe_issued_at + 1e-9, safe_now),
            source="circa_v8_fail_closed_refusal",
            constraint_state="refused",
        )
        payload = {
            "action": envelope.action.tolist(),
            "issued_at": envelope.issued_at,
            "valid_until": envelope.valid_until,
            "source": envelope.source,
            "reason": reason,
            "refusal_code": refusal_code,
        }
        return AdapterDecision(
            envelope=envelope,
            intervened=True,
            fail_closed=True,
            evidence_valid=False,
            certificate_emitted=False,
            refusal_code=refusal_code,
            reason=reason,
            decision_digest=_canonical_digest(payload),
        )

    def decide(
        self,
        *,
        state_history: Sequence[np.ndarray],
        applied_action_history: Sequence[np.ndarray],
        nominal_envelope: ActionEnvelope,
        now: float,
        observation_delay_steps: int,
        communication_delay_steps: int,
        provenance_valid: bool = True,
        monotonic_time_valid: bool = True,
        observed_common_mode_bias_m: float = 0.0,
    ) -> AdapterDecision:
        if not provenance_valid:
            return self._refusal(
                issued_at=nominal_envelope.issued_at,
                now=now,
                reason="evidence_provenance_hash_mismatch",
                refusal_code="invalid_provenance",
            )
        if not monotonic_time_valid:
            return self._refusal(
                issued_at=nominal_envelope.issued_at,
                now=now,
                reason="monotonic_time_reversal",
                refusal_code="invalid_monotonic_time",
            )
        try:
            nominal = nominal_envelope.checked_action(now)
        except ExpiredActionError:
            return self._refusal(
                issued_at=nominal_envelope.issued_at,
                now=now,
                reason="action_expired",
                refusal_code="expired_action",
            )
        except ContractError:
            return self._refusal(
                issued_at=max(0.0, nominal_envelope.issued_at),
                now=max(0.0, now),
                reason="invalid_action_envelope",
                refusal_code="invalid_action_contract",
            )
        decision: SetBackupDecision = self._filter.align_and_decide(
            state_history,
            applied_action_history,
            observation_delay_steps,
            communication_delay_steps,
            nominal,
            observed_common_mode_bias_m=observed_common_mode_bias_m,
        )
        validity = decision.certificate.get("validity_interval_steps") if decision.certificate else None
        valid_until = (
            nominal_envelope.valid_until
            if validity is not None and decision.evidence_valid
            else max(nominal_envelope.issued_at + 1e-9, now)
        )
        envelope = ActionEnvelope(
            action=np.asarray(decision.action, dtype=float),
            issued_at=nominal_envelope.issued_at,
            valid_until=valid_until,
            source="circa_v8_timestamp_aligned_set_backup",
            constraint_state="fail_closed" if decision.fail_closed else "checked",
        )
        refusal_code = "none" if decision.evidence_valid else "invalid_alignment_evidence"
        payload = {
            "action": envelope.action.tolist(),
            "issued_at": envelope.issued_at,
            "valid_until": envelope.valid_until,
            "source": envelope.source,
            "intervened": decision.intervened,
            "fail_closed": decision.fail_closed,
            "evidence_valid": decision.evidence_valid,
            "certificate_emitted": decision.certificate_emitted,
            "refusal_code": refusal_code,
            "reason": decision.reason,
            "certificate": decision.certificate,
        }
        return AdapterDecision(
            envelope=envelope,
            intervened=decision.intervened,
            fail_closed=decision.fail_closed,
            evidence_valid=decision.evidence_valid,
            certificate_emitted=decision.certificate_emitted,
            refusal_code=refusal_code,
            reason=decision.reason,
            decision_digest=_canonical_digest(payload),
        )


__all__ = [
    "AdapterDecision",
    "DiversityAdapterError",
    "DiversityRuntimeAdapter",
    "FROZEN_DESIGN_SHA256",
    "FROZEN_FAMILIES",
    "ModelPatch",
    "ScenarioSpec",
    "WorldPatch",
    "load_frozen_design",
    "scenario_from_design",
    "validate_world_semantics",
    "world_patch_for",
]
