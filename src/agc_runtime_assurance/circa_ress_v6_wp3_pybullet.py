"""CIRCA-RESS-V6 WP3 low-resource PyBullet confirmation.

Importing this module never imports PyBullet and never starts a simulator.
The scientific entry point requires a future seed manifest, an absent output
directory, and a launch manifest whose hashes are checked by the supervisor.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct
from typing import Any, Iterable


ROUTE_ID = "CIRCA-RESS-V6-WP3-PYBULLET-LR1"
ENGINE_VERSION = "3.2.7"
FAMILIES = tuple(f"PBF{i}" for i in range(1, 7))
POSITIVE_FAMILIES = FAMILIES[:5]
ALPHA_S = 0.025
ALPHA_BETA = 0.025
DELTA_STAR = 0.10
CAPACITY_BYTES = 64 * 1024 * 1024


class WP3Error(RuntimeError):
    """Fail-closed route error."""


@dataclass(frozen=True)
class ScenarioResult:
    phase: str
    family: str
    candidate: str
    seed: int
    block: int
    intervention: int
    y1: int
    y0_oracle: int
    witness: str
    active_min_margin: float
    shadow_min_margin: float
    active_first_step: int
    shadow_first_step: int
    active_digest: str
    shadow_digest: str

    def public_view(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "family": self.family,
            "candidate": self.candidate,
            "seed": self.seed,
            "block": self.block,
            "R": self.intervention,
            "Y1": self.y1,
            "W": self.witness,
            "active_min_margin": self.active_min_margin,
            "active_first_step": self.active_first_step,
            "active_digest": self.active_digest,
        }

    def oracle_view(self) -> dict[str, Any]:
        value = self.public_view()
        value.update(
            {
                "Y0_oracle": self.y0_oracle,
                "shadow_min_margin": self.shadow_min_margin,
                "shadow_first_step": self.shadow_first_step,
                "shadow_digest": self.shadow_digest,
            }
        )
        return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _candidate_registry() -> dict[str, dict[str, Any]]:
    path = (
        _repo_root()
        / "outputs"
        / "researchwrite"
        / "circa-ress"
        / "wp3_low_resource_confirmation"
        / "05_candidate_registry.json"
    )
    data = _load_json(path)
    candidates = {item["id"]: item for item in data["candidates"]}
    if len(candidates) != 12:
        raise WP3Error("candidate registry must contain exactly 12 unique candidates")
    return candidates


def _import_pybullet():
    try:
        import pybullet as p  # type: ignore
    except Exception as exc:
        raise WP3Error(f"PYBULLET_IMPORT_FAILED:{type(exc).__name__}") from exc
    version = getattr(p, "__version__", None)
    return p, version


def _closest_approach(candidate: dict[str, Any], ugv_x: float, uav_y: float,
                      vx: float, vy: float) -> float:
    rx = float(candidate["lateral_offset_m"]) - ugv_x
    ry = uav_y
    rvx = -vx
    rvy = vy
    denom = rvx * rvx + rvy * rvy
    t = 0.0 if denom == 0.0 else max(0.0, min(2.0, -(rx * rvx + ry * rvy) / denom))
    return math.hypot(rx + rvx * t, ry + rvy * t)


def _witness(candidate: dict[str, Any], rng: random.Random) -> str:
    px = float(candidate["ugv_start_x_m"]) + rng.uniform(-0.05, 0.05)
    py = float(candidate["uav_start_y_m"]) + rng.uniform(-0.05, 0.05)
    vx = float(candidate["ugv_speed_mps"]) + rng.uniform(-0.05, 0.05)
    vy = float(candidate["uav_speed_mps"]) + rng.uniform(-0.05, 0.05)
    predicted = _closest_approach(candidate, px, py, vx, vy)
    radius = (
        0.05
        + abs(float(candidate["sensor_bias_m"]))
        + 0.03
        * (
            int(candidate["observation_delay_steps"])
            + int(candidate["actuation_lag_steps"])
            + int(candidate["communication_delay_steps"])
        )
        + 0.10 * float(candidate["drag"])
    )
    threshold = 0.55
    if predicted + radius < threshold:
        return "IV"
    if predicted - radius >= threshold:
        return "NS"
    return "U"


def _simulate(
    candidate: dict[str, Any], seed: int, active: bool, client: int | None = None
) -> tuple[int, float, int, str]:
    p, _ = _import_pybullet()
    rng = random.Random(seed)
    own_client = client is None
    if client is None:
        client = p.connect(p.DIRECT)
    if client < 0:
        raise WP3Error("PYBULLET_DIRECT_CONNECT_FAILED")
    digest = hashlib.sha256()
    try:
        p.resetSimulation(physicsClientId=client)
        p.setGravity(0.0, 0.0, 0.0, physicsClientId=client)
        p.setTimeStep(1.0 / 60.0, physicsClientId=client)
        uav_shape = p.createCollisionShape(
            p.GEOM_SPHERE, radius=0.18, physicsClientId=client
        )
        ugv_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.25, 0.18, 0.12], physicsClientId=client
        )
        ugv_x = float(candidate["ugv_start_x_m"]) + rng.uniform(-0.05, 0.05)
        uav_y = float(candidate["uav_start_y_m"]) + rng.uniform(-0.05, 0.05)
        ugv_v = float(candidate["ugv_speed_mps"]) + rng.uniform(-0.05, 0.05)
        uav_v = float(candidate["uav_speed_mps"]) + rng.uniform(-0.05, 0.05)
        ugv = p.createMultiBody(
            8.0, ugv_shape, basePosition=[ugv_x, 0.0, 0.12], physicsClientId=client
        )
        uav = p.createMultiBody(
            1.5,
            uav_shape,
            basePosition=[float(candidate["lateral_offset_m"]), uav_y, 0.80],
            physicsClientId=client,
        )
        trigger_delay = (
            int(candidate["observation_delay_steps"])
            + int(candidate["communication_delay_steps"])
        )
        actuation_lag = int(candidate["actuation_lag_steps"])
        triggered_at: int | None = None
        first_step = -1
        min_margin = float("inf")
        for step in range(120):
            ugv_pos, _ = p.getBasePositionAndOrientation(ugv, physicsClientId=client)
            uav_pos, _ = p.getBasePositionAndOrientation(uav, physicsClientId=client)
            horizontal = math.hypot(uav_pos[0] - ugv_pos[0], uav_pos[1] - ugv_pos[1])
            margin = horizontal - 0.55
            min_margin = min(min_margin, margin)
            violation = horizontal < 0.55 and uav_pos[2] < 1.10
            if violation and first_step < 0:
                first_step = step
            digest.update(struct.pack("<6d", *(tuple(ugv_pos) + tuple(uav_pos))))

            rel_x = uav_pos[0] - ugv_pos[0]
            rel_y = uav_pos[1] - ugv_pos[1]
            rv_x = -ugv_v
            rv_y = uav_v
            denom = rv_x * rv_x + rv_y * rv_y
            tstar = 0.0 if denom == 0.0 else max(
                0.0, min(0.60, -(rel_x * rv_x + rel_y * rv_y) / denom)
            )
            projected = math.hypot(rel_x + rv_x * tstar, rel_y + rv_y * tstar)
            sensed = projected + float(candidate["sensor_bias_m"])
            if active and triggered_at is None and sensed < 0.65:
                triggered_at = step + trigger_delay

            climb = 0.0
            current_ugv_v = ugv_v
            if (
                active
                and triggered_at is not None
                and step >= triggered_at + actuation_lag
            ):
                elapsed = (step - triggered_at - actuation_lag + 1) / 60.0
                current_ugv_v = max(0.0, ugv_v - 1.50 * elapsed)
                climb = 0.80
            drag = max(0.0, 1.0 - float(candidate["drag"]) / 60.0)
            neighbor = float(candidate["neighbor_influence"])
            p.resetBaseVelocity(
                ugv,
                linearVelocity=[current_ugv_v * drag, neighbor * 0.02, 0.0],
                physicsClientId=client,
            )
            p.resetBaseVelocity(
                uav,
                linearVelocity=[-neighbor * 0.02, uav_v * drag, climb],
                physicsClientId=client,
            )
            p.stepSimulation(physicsClientId=client)
        return int(first_step >= 0), min_margin, first_step, digest.hexdigest()
    finally:
        if own_client:
            p.disconnect(physicsClientId=client)


def world_load_preflight() -> dict[str, Any]:
    p, version = _import_pybullet()
    client = p.connect(p.DIRECT)
    if client < 0:
        raise WP3Error("PYBULLET_DIRECT_CONNECT_FAILED")
    try:
        p.resetSimulation(physicsClientId=client)
        p.setGravity(0.0, 0.0, 0.0, physicsClientId=client)
        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1, physicsClientId=client)
        body = p.createMultiBody(1.0, shape, basePosition=[0, 0, 0], physicsClientId=client)
        p.stepSimulation(physicsClientId=client)
        pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=client)
        if not all(math.isfinite(float(v)) for v in pos):
            raise WP3Error("WORLD_LOAD_NONFINITE_STATE")
    finally:
        p.disconnect(physicsClientId=client)
    return {
        "route_id": ROUTE_ID,
        "kind": "NON_SCIENTIFIC_WORLD_LOAD_PREFLIGHT",
        "pybullet_reported_version": version,
        "direct_connect": "PASS",
        "world_load": "PASS",
        "step": "PASS",
        "scientific_seed_used": False,
        "scientific_output_created": False,
        "result": "PASS",
    }


def _budget_certificates(rows: Iterable[ScenarioResult]) -> dict[str, dict[str, float]]:
    grouped = {family: [] for family in FAMILIES}
    for row in rows:
        grouped[row.family].append(row)
    h = math.sqrt(math.log(2 * len(FAMILIES) / ALPHA_BETA) / (2 * 2048))
    result: dict[str, dict[str, float]] = {}
    for family, values in grouped.items():
        if len(values) != 2048:
            raise WP3Error(f"validation count mismatch for {family}")
        e_iv = sum(v.intervention and v.witness == "IV" and v.y0_oracle == 0 for v in values)
        e_ns = sum(v.intervention and v.witness == "NS" and v.y0_oracle == 1 for v in values)
        result[family] = {
            "beta_iv": min(1.0, e_iv / len(values) + h),
            "beta_ns": min(1.0, e_ns / len(values) + h),
            "observed_joint_error_iv": e_iv / len(values),
            "observed_joint_error_ns": e_ns / len(values),
            "certificate_radius": h,
        }
    return result


def _robust_interval(rows: list[ScenarioResult], beta: dict[str, float]) -> dict[str, float]:
    n = len(rows)
    if n == 0:
        raise WP3Error("empty family")
    gl = -sum(v.intervention * v.y1 for v in rows) / n
    gu = sum(v.intervention * (1 - v.y1) for v in rows) / n
    q_iv = sum(v.intervention and v.witness == "IV" for v in rows) / n
    q_ns = sum(v.intervention and v.witness == "NS" for v in rows) / n
    q_u = sum(v.intervention and v.witness == "U" for v in rows) / n
    lower = gl + q_iv - min(beta["beta_iv"], q_iv)
    upper = gu - q_ns + min(beta["beta_ns"], q_ns)
    oracle = sum(v.y0_oracle - v.y1 for v in rows) / n
    return {
        "general_lower": gl,
        "general_upper": gu,
        "robust_lower": lower,
        "robust_upper": upper,
        "oracle_delta": oracle,
        "q_iv": q_iv,
        "q_ns": q_ns,
        "q_u": q_u,
        "width_ratio": (upper - lower) / (gu - gl) if gu > gl else 1.0,
    }


def _feature(candidate: dict[str, Any]) -> tuple[float, ...]:
    """Frozen, dimensionless metric coordinates for recent smoothness methods."""
    return (
        float(candidate["ugv_start_x_m"]) / 3.0,
        float(candidate["ugv_speed_mps"]) / 2.0,
        float(candidate["uav_start_y_m"]) / 3.0,
        float(candidate["uav_speed_mps"]) / 2.0,
        float(candidate["lateral_offset_m"]) / 1.0,
        float(candidate["sensor_bias_m"]) / 0.25,
        (
            int(candidate["observation_delay_steps"])
            + int(candidate["actuation_lag_steps"])
            + int(candidate["communication_delay_steps"])
        )
        / 12.0,
        float(candidate["drag"]) / 0.30,
        float(candidate["neighbor_influence"]) / 0.50,
    )


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _smooth_policy_bounds(
    rows: list[ScenarioResult],
    candidates: dict[str, dict[str, Any]],
    *,
    network: bool,
) -> dict[str, dict[str, Any]]:
    """Finite candidate-grid implementation of KSU24/WLZ26 closed forms."""
    anchors: dict[str, float] = {}
    for candidate_id in ("PBF6-A", "PBF6-B"):
        observed = [
            row.y1
            for row in rows
            if row.candidate == candidate_id and row.intervention == 0
        ]
        if not observed:
            raise WP3Error("SMOOTHNESS_OVERLAP_ANCHOR_EMPTY")
        anchors[candidate_id] = sum(observed) / len(observed)
    lipschitz = 1.0
    theta = 0.05 if network else 0.0
    result: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        family_rows = [row for row in rows if row.family == family]
        active_risk = sum(row.y1 for row in family_rows) / len(family_rows)
        if network and family != "PBF5":
            result[family] = {"status": "NOT_COMPARABLE_NO_REGISTERED_INTERFERENCE"}
            continue
        if not network and family == "PBF5":
            result[family] = {"status": "NOT_COMPARABLE_INTERFERENCE"}
            continue
        lower_values: list[float] = []
        upper_values: list[float] = []
        for candidate_id in (f"{family}-A", f"{family}-B"):
            x = _feature(candidates[candidate_id])
            lower = max(
                anchors[anchor_id]
                - lipschitz * _distance(x, _feature(candidates[anchor_id]))
                for anchor_id in anchors
            )
            upper = min(
                anchors[anchor_id]
                + lipschitz * _distance(x, _feature(candidates[anchor_id]))
                for anchor_id in anchors
            )
            lower_values.append(max(0.0, lower - theta))
            upper_values.append(min(1.0, upper + theta))
        result[family] = {
            "status": "NUMERIC",
            "lower": sum(lower_values) / 2.0 - active_risk,
            "upper": sum(upper_values) / 2.0 - active_risk,
            "lipschitz_L": lipschitz,
            "truncation_remainder": theta,
            "anchor_candidates": sorted(anchors),
        }
    return result


def _overlap_estimators(rows: list[ScenarioResult]) -> dict[str, Any]:
    family_rows = [row for row in rows if row.family == "PBF6"]
    propensity_no_intervention = 0.5
    mu0_ipw = sum(
        (1 - row.intervention) * row.y1 / propensity_no_intervention
        for row in family_rows
    ) / len(family_rows)
    active_risk = sum(row.y1 for row in family_rows) / len(family_rows)
    fold_values: list[float] = []
    for held_out_parity in (0, 1):
        train = [
            row
            for row in family_rows
            if row.block % 2 != held_out_parity and row.intervention == 0
        ]
        test = [row for row in family_rows if row.block % 2 == held_out_parity]
        means: dict[str, float] = {}
        for candidate_id in ("PBF6-A", "PBF6-B"):
            values = [row.y1 for row in train if row.candidate == candidate_id]
            if not values:
                raise WP3Error("AIPW_TRAINING_CELL_EMPTY")
            means[candidate_id] = sum(values) / len(values)
        fold_values.extend(
            means[row.candidate]
            + (1 - row.intervention)
            / propensity_no_intervention
            * (row.y1 - means[row.candidate])
            for row in test
        )
    return {
        "IPW_OVERLAP": {
            "status": "NUMERIC",
            "delta": mu0_ipw - active_risk,
            "eligible_family": "PBF6",
        },
        "AIPW_XFIT_OVERLAP": {
            "status": "NUMERIC",
            "delta": sum(fold_values) / len(fold_values) - active_risk,
            "eligible_family": "PBF6",
        },
    }


def _comparator_results(
    science_rows: list[ScenarioResult],
    validation_rows: list[ScenarioResult],
    candidates: dict[str, dict[str, Any]],
    beta: dict[str, dict[str, float]],
    families: dict[str, Any],
) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    methods["MANSKI_BINARY"] = {
        family: {
            "status": "NUMERIC",
            "lower": values["general_lower"],
            "upper": values["general_upper"],
        }
        for family, values in families.items()
    }
    overlap = _overlap_estimators(science_rows)
    for method_id, pbf6_value in overlap.items():
        methods[method_id] = {
            family: (
                pbf6_value
                if family == "PBF6"
                else {"status": "UNIDENTIFIABLE_PROPENSITY"}
            )
            for family in FAMILIES
        }
    methods["KSU24_SMOOTH_PI"] = _smooth_policy_bounds(
        science_rows, candidates, network=False
    )
    methods["WLZ26_NETWORK_PI"] = _smooth_policy_bounds(
        science_rows, candidates, network=True
    )
    exact: dict[str, Any] = {}
    robust: dict[str, Any] = {}
    oracle: dict[str, Any] = {}
    for family in FAMILIES:
        validation = [row for row in validation_rows if row.family == family]
        error_count = sum(
            row.intervention
            and (
                (row.witness == "IV" and row.y0_oracle == 0)
                or (row.witness == "NS" and row.y0_oracle == 1)
            )
            for row in validation
        )
        family_rows = [row for row in science_rows if row.family == family]
        if error_count == 0:
            exact_values = _robust_interval(
                family_rows, {"beta_iv": 0.0, "beta_ns": 0.0}
            )
            exact[family] = {
                "status": "NUMERIC",
                "lower": exact_values["robust_lower"],
                "upper": exact_values["robust_upper"],
            }
        else:
            exact[family] = {"status": "INVALID_WITNESS"}
        robust[family] = {
            "status": "NUMERIC",
            "lower": families[family]["confidence_lower"],
            "upper": families[family]["confidence_upper"],
            "beta_iv": beta[family]["beta_iv"],
            "beta_ns": beta[family]["beta_ns"],
        }
        oracle[family] = {
            "status": "EVALUATION_ONLY",
            "delta": families[family]["oracle_delta"],
            "rankable": False,
        }
    methods["CIRCA_V5_EXACT"] = exact
    methods["CIRCA_V6_ROBUST"] = robust
    methods["ORACLE_Y0"] = oracle
    return methods


def _cp_all_success_lower(n: int, alpha: float = 0.05) -> float:
    return alpha ** (1.0 / n)


def _cp_zero_success_upper(n: int, alpha: float = 0.05) -> float:
    return 1.0 - alpha ** (1.0 / n)


def run_science(seed_manifest: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise WP3Error("SCIENTIFIC_TARGET_ALREADY_EXISTS")
    seeds = _load_json(seed_manifest)
    if seeds.get("route_id") != ROUTE_ID or seeds.get("frozen") is not True:
        raise WP3Error("INVALID_SEED_MANIFEST")
    candidates = _candidate_registry()
    output.mkdir(parents=False, exist_ok=False)
    validation_rows: list[ScenarioResult] = []
    science_rows: list[ScenarioResult] = []
    raw_path = output / "scenario_records.jsonl"
    p, _ = _import_pybullet()
    client = p.connect(p.DIRECT)
    if client < 0:
        raise WP3Error("PYBULLET_DIRECT_CONNECT_FAILED")
    try:
        with raw_path.open("x", encoding="utf-8", newline="\n") as stream:
            for phase, target in (("validation", validation_rows), ("science", science_rows)):
                expected = 1024 if phase == "validation" else 4096
                for candidate_id, candidate in candidates.items():
                    vector = seeds[phase][candidate_id]
                    if len(vector) != expected or len(set(vector)) != len(vector):
                        raise WP3Error(f"seed vector mismatch: {phase}:{candidate_id}")
                    for index, seed in enumerate(vector):
                        assignment_rng = random.Random(seed ^ 0x6A09E667)
                        intervention = int(
                            assignment_rng.random()
                            < float(candidate["intervention_propensity"])
                        )
                        y1, amin, afirst, adigest = _simulate(
                            candidate, seed, active=bool(intervention), client=client
                        )
                        y0, smin, sfirst, sdigest = _simulate(
                            candidate, seed, active=False, client=client
                        )
                        witness = _witness(
                            candidate, random.Random(seed ^ 0xBB67AE85)
                        )
                        row = ScenarioResult(
                            phase=phase,
                            family=candidate["family"],
                            candidate=candidate_id,
                            seed=seed,
                            block=-1 if phase == "validation" else index // 64,
                            intervention=intervention,
                            y1=y1,
                            y0_oracle=y0,
                            witness=witness,
                            active_min_margin=amin,
                            shadow_min_margin=smin,
                            active_first_step=afirst,
                            shadow_first_step=sfirst,
                            active_digest=adigest,
                            shadow_digest=sdigest,
                        )
                        target.append(row)
                        stream.write(
                            json.dumps(
                                row.oracle_view(), sort_keys=True, separators=(",", ":")
                            )
                            + "\n"
                        )
    finally:
        p.disconnect(physicsClientId=client)
    if raw_path.stat().st_size > CAPACITY_BYTES:
        raise WP3Error("OUTPUT_CAPACITY_EXCEEDED")
    beta = _budget_certificates(validation_rows)
    families: dict[str, Any] = {}
    block_coverage: dict[str, int] = {}
    block_positive: dict[str, int] = {}
    r = math.sqrt(math.log(4 * len(FAMILIES) / ALPHA_S) / (2 * 8192))
    for family in FAMILIES:
        rows = [v for v in science_rows if v.family == family]
        if len(rows) != 8192:
            raise WP3Error(f"science count mismatch for {family}")
        info = _robust_interval(rows, beta[family])
        info["confidence_lower"] = max(
            -1.0,
            info["general_lower"] - r
            + max(0.0, info["q_iv"] - r - beta[family]["beta_iv"]),
        )
        info["confidence_upper"] = min(
            1.0,
            info["general_upper"] + r
            - max(0.0, info["q_ns"] - r - beta[family]["beta_ns"]),
        )
        covered = 0
        positive = 0
        for block in range(64):
            block_rows = [v for v in rows if v.block == block]
            block_info = _robust_interval(block_rows, beta[family])
            covered += int(
                block_info["robust_lower"]
                <= block_info["oracle_delta"]
                <= block_info["robust_upper"]
            )
            positive += int(block_info["robust_lower"] > DELTA_STAR)
        block_coverage[family] = covered
        block_positive[family] = positive
        families[family] = info

    gates = {
        "all_family_pooled_oracle_coverage": all(
            value["confidence_lower"] <= value["oracle_delta"] <= value["confidence_upper"]
            for value in families.values()
        ),
        "block_coverage_cp_lower": all(
            count == 64 and _cp_all_success_lower(64) >= 0.95
            for count in block_coverage.values()
        ),
        "null_false_certification_cp_upper": (
            block_positive["PBF6"] == 0 and _cp_zero_success_upper(64) <= 0.05
        ),
        "worst_positive_robust_lower": min(
            families[f]["confidence_lower"] for f in POSITIVE_FAMILIES
        )
        > DELTA_STAR,
        "positive_width_ratio": all(
            families[f]["width_ratio"] < 0.80 for f in POSITIVE_FAMILIES
        ),
    }
    comparators = _comparator_results(
        science_rows, validation_rows, candidates, beta, families
    )
    summary = {
        "route_id": ROUTE_ID,
        "result": "PASS" if all(gates.values()) else "SCIENTIFIC_GATE_FAIL",
        "science_attempts": 1,
        "retry_allowed": False,
        "seed_manifest_sha256": hashlib.sha256(seed_manifest.read_bytes()).hexdigest(),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "raw_bytes": raw_path.stat().st_size,
        "beta_certificates": beta,
        "families": families,
        "block_coverage": block_coverage,
        "block_positive": block_positive,
        "gates": gates,
        "comparators": comparators,
    }
    (output / "result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    science = sub.add_parser("science")
    science.add_argument("--seed-manifest", type=Path, required=True)
    science.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(world_load_preflight(), sort_keys=True))
        return
    print(
        json.dumps(
            run_science(args.seed_manifest.resolve(), args.output.resolve()),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
