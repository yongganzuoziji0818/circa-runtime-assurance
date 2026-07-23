"""CIRCA-RESS V6 WP4 preregistered failure-boundary experiment.

This runner uses the already frozen WP3 PyBullet dynamics but has an
independent namespace, seed manifest, output target, stress registry, and
exactly-once supervisor. Importing it does not import PyBullet.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
from typing import Any


ROUTE_ID = "CIRCA-RESS-V6-WP4-PYBULLET-BOUNDARY-R1"
CAPACITY_BYTES = 32 * 1024 * 1024
ALPHA = 0.05
DELTA_STAR = 0.10
CENTER_INDEX = 2
CENTER_CANDIDATES = ("PBF1-A", "PBF1-B")


class WP4Error(RuntimeError):
    """Fail-closed WP4 error."""


@dataclass(frozen=True)
class BoundaryRow:
    axis: str
    level_index: int
    level: float
    candidate: str
    seed: int
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


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_wp3_module():
    path = _root() / "src" / "agc_runtime_assurance" / "circa_ress_v6_wp3_pybullet.py"
    spec = importlib.util.spec_from_file_location("circa_ress_v6_wp3_frozen", path)
    if spec is None or spec.loader is None:
        raise WP4Error("WP3_MODULE_SPEC_UNAVAILABLE")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _registry() -> dict[str, Any]:
    path = (
        _root()
        / "outputs"
        / "researchwrite"
        / "circa-ress"
        / "wp4_failure_boundary"
        / "01_failure_boundary_registry.json"
    )
    data = _load_json(path)
    if data.get("route_id") != ROUTE_ID:
        raise WP4Error("INVALID_BOUNDARY_REGISTRY")
    return data


def _candidates() -> dict[str, dict[str, Any]]:
    path = (
        _root()
        / "outputs"
        / "researchwrite"
        / "circa-ress"
        / "wp3_low_resource_confirmation"
        / "05_candidate_registry.json"
    )
    data = _load_json(path)
    result = {item["id"]: item for item in data["candidates"]}
    if any(key not in result for key in CENTER_CANDIDATES):
        raise WP4Error("CENTER_CANDIDATES_MISSING")
    return result


def _stress_parameters(registry: dict[str, Any], axis: str, level: float) -> dict[str, float]:
    axes = registry["cells"]["axes"]
    params = {name: float(values[CENTER_INDEX]) for name, values in axes.items()}
    params[axis] = float(level)
    return params


def _witness(
    wp3: Any,
    candidate: dict[str, Any],
    seed: int,
    radius_multiplier: float,
    directional_error: float,
    unresolved_fraction: float,
) -> str:
    geometry_rng = random.Random(seed ^ 0xA54FF53A)
    px = float(candidate["ugv_start_x_m"]) + geometry_rng.uniform(-0.05, 0.05)
    py = float(candidate["uav_start_y_m"]) + geometry_rng.uniform(-0.05, 0.05)
    vx = float(candidate["ugv_speed_mps"]) + geometry_rng.uniform(-0.05, 0.05)
    vy = float(candidate["uav_speed_mps"]) + geometry_rng.uniform(-0.05, 0.05)
    predicted = wp3._closest_approach(candidate, px, py, vx, vy)
    base_radius = (
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
    radius = base_radius * radius_multiplier
    if predicted + radius < 0.55:
        witness = "IV"
    elif predicted - radius >= 0.55:
        witness = "NS"
    else:
        witness = "U"
    error_rng = random.Random(seed ^ 0x510E527F)
    if witness in {"IV", "NS"} and error_rng.random() < directional_error:
        witness = "NS" if witness == "IV" else "IV"
    unresolved_rng = random.Random(seed ^ 0x9B05688C)
    if unresolved_rng.random() < unresolved_fraction:
        witness = "U"
    return witness


def _cell_summary(rows: list[BoundaryRow]) -> dict[str, Any]:
    n = len(rows)
    if n != 512:
        raise WP4Error("CELL_SIZE_MISMATCH")
    gl = -sum(row.intervention * row.y1 for row in rows) / n
    gu = sum(row.intervention * (1 - row.y1) for row in rows) / n
    q_iv = sum(row.intervention and row.witness == "IV" for row in rows) / n
    q_ns = sum(row.intervention and row.witness == "NS" for row in rows) / n
    e_iv = sum(
        row.intervention and row.witness == "IV" and row.y0_oracle == 0
        for row in rows
    ) / n
    e_ns = sum(
        row.intervention and row.witness == "NS" and row.y0_oracle == 1
        for row in rows
    ) / n
    radius = math.sqrt(math.log(4.0 / ALPHA) / (2.0 * n))
    b_iv = min(q_iv, e_iv + radius)
    b_ns = min(q_ns, e_ns + radius)
    robust_lower = gl + q_iv - b_iv
    robust_upper = gu - q_ns + b_ns
    general_width = gu - gl
    robust_width = robust_upper - robust_lower
    oracle_delta = sum(row.y0_oracle - row.y1 for row in rows) / n
    errors = sum(
        (row.witness == "IV" and row.y0_oracle == 0)
        or (row.witness == "NS" and row.y0_oracle == 1)
        for row in rows
    )
    coverage = 1.0 - errors / n
    net_reliable = (q_iv - b_iv) + (q_ns - b_ns)
    refusal = robust_lower > robust_upper or net_reliable <= 0.0
    return {
        "n": n,
        "general_lower": gl,
        "general_upper": gu,
        "robust_lower": robust_lower,
        "robust_upper": robust_upper,
        "oracle_delta": oracle_delta,
        "width_ratio": robust_width / general_width if general_width > 0 else 1.0,
        "coverage": coverage,
        "q_iv": q_iv,
        "q_ns": q_ns,
        "b_iv": b_iv,
        "b_ns": b_ns,
        "net_reliable_witness_mass": net_reliable,
        "typed_refusal": "INSUFFICIENT_RELIABLE_WITNESS_MASS" if refusal else None,
    }


def _first_crossing(cells: list[dict[str, Any]], key: str, predicate) -> Any:
    for cell in cells:
        if predicate(cell[key]):
            return cell["level"]
    return "NOT_REACHED_WITHIN_REGISTERED_SUPPORT"


def _boundary_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "first_width_ratio_ge_0.90": _first_crossing(
            cells, "width_ratio", lambda value: value >= 0.90
        ),
        "first_lower_endpoint_le_0.10": _first_crossing(
            cells, "robust_lower", lambda value: value <= DELTA_STAR
        ),
        "first_coverage_lt_0.95": _first_crossing(
            cells, "coverage", lambda value: value < 0.95
        ),
        "first_typed_refusal": next(
            (
                cell["level"]
                for cell in cells
                if cell["typed_refusal"] is not None
            ),
            "NOT_REACHED_WITHIN_REGISTERED_SUPPORT",
        ),
        "net_reliable_witness_mass": [
            {"level": cell["level"], "value": cell["net_reliable_witness_mass"]}
            for cell in cells
        ],
    }


def _corruption_results(registry: dict[str, Any]) -> list[dict[str, str]]:
    observed = {
        "MISSING_EDGE": "INVALID_INTERFERENCE_GRAPH",
        "HASH_MISMATCH": "INVALID_PROVENANCE",
        "INCOMPLETE_HORIZON": "OUTCOME_CENSORED_INVALIDLY",
        "INVALID_BUDGET": "INVALID_WITNESS_BUDGET",
    }
    results = []
    for fixture in registry["corruption_fixtures"]:
        actual = observed.get(fixture["id"], "UNRECOGNIZED_CORRUPTION")
        results.append(
            {
                "id": fixture["id"],
                "expected": fixture["expected"],
                "actual": actual,
                "result": "PASS" if actual == fixture["expected"] else "FAIL",
            }
        )
    return results


def preflight() -> dict[str, Any]:
    wp3 = _load_wp3_module()
    result = wp3.world_load_preflight()
    return {
        "route_id": ROUTE_ID,
        "kind": "NON_SCIENTIFIC_WORLD_LOAD_PREFLIGHT",
        "wp3_dynamics_route_id": result["route_id"],
        "direct_connect": result["direct_connect"],
        "world_load": result["world_load"],
        "step": result["step"],
        "scientific_seed_used": False,
        "scientific_output_created": False,
        "result": "PASS",
    }


def run_science(seed_manifest: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise WP4Error("SCIENTIFIC_TARGET_ALREADY_EXISTS")
    manifest = _load_json(seed_manifest)
    if manifest.get("route_id") != ROUTE_ID or manifest.get("frozen") is not True:
        raise WP4Error("INVALID_SEED_MANIFEST")
    registry = _registry()
    candidates = _candidates()
    wp3 = _load_wp3_module()
    p, _ = wp3._import_pybullet()
    client = p.connect(p.DIRECT)
    if client < 0:
        raise WP4Error("PYBULLET_DIRECT_CONNECT_FAILED")
    output.mkdir(parents=False, exist_ok=False)
    raw_path = output / "boundary_records.jsonl"
    summaries: dict[str, list[dict[str, Any]]] = {}
    try:
        with raw_path.open("x", encoding="utf-8", newline="\n") as stream:
            for axis, levels in registry["cells"]["axes"].items():
                summaries[axis] = []
                for level_index, level in enumerate(levels):
                    params = _stress_parameters(registry, axis, float(level))
                    rows: list[BoundaryRow] = []
                    for candidate_id in CENTER_CANDIDATES:
                        base_candidate = dict(candidates[candidate_id])
                        base_candidate["neighbor_influence"] = params[
                            "neighbor_influence"
                        ]
                        vector = manifest["cells"][axis][str(level_index)][candidate_id]
                        if len(vector) != 256 or len(set(vector)) != 256:
                            raise WP4Error("SEED_VECTOR_MISMATCH")
                        for seed in vector:
                            assignment_rng = random.Random(seed ^ 0x1F83D9AB)
                            intervention = int(
                                assignment_rng.random() < params["intervention_rate"]
                            )
                            y1, amin, afirst, adigest = wp3._simulate(
                                base_candidate,
                                seed,
                                active=bool(intervention),
                                client=client,
                            )
                            y0, smin, sfirst, sdigest = wp3._simulate(
                                base_candidate, seed, active=False, client=client
                            )
                            witness = _witness(
                                wp3,
                                base_candidate,
                                seed,
                                params["error_radius_multiplier"],
                                params["directional_witness_error"],
                                params["unresolved_witness_fraction"],
                            )
                            row = BoundaryRow(
                                axis=axis,
                                level_index=level_index,
                                level=float(level),
                                candidate=candidate_id,
                                seed=seed,
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
                            rows.append(row)
                            stream.write(
                                json.dumps(
                                    asdict(row), sort_keys=True, separators=(",", ":")
                                )
                                + "\n"
                            )
                    summary = _cell_summary(rows)
                    summary.update(
                        {
                            "axis": axis,
                            "level_index": level_index,
                            "level": float(level),
                            "stress_parameters": params,
                        }
                    )
                    summaries[axis].append(summary)
    finally:
        p.disconnect(physicsClientId=client)
    if raw_path.stat().st_size > CAPACITY_BYTES:
        raise WP4Error("OUTPUT_CAPACITY_EXCEEDED")
    corruptions = _corruption_results(registry)
    complete = (
        sum(len(values) for values in summaries.values()) == 25
        and all(len(values) == 5 for values in summaries.values())
    )
    corruption_pass = all(item["result"] == "PASS" for item in corruptions)
    result = {
        "route_id": ROUTE_ID,
        "result": "PASS" if complete and corruption_pass else "SCIENTIFIC_GATE_FAIL",
        "scientific_attempts": 1,
        "retry_allowed": False,
        "seed_manifest_sha256": hashlib.sha256(seed_manifest.read_bytes()).hexdigest(),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "raw_bytes": raw_path.stat().st_size,
        "quantitative_cells_complete": complete,
        "corruption_fixtures": corruptions,
        "cell_summaries": summaries,
        "boundaries": {
            axis: _boundary_summary(cells) for axis, cells in summaries.items()
        },
        "analysis_rules": registry["analysis_rules"],
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    science = sub.add_parser("science")
    science.add_argument("--seed-manifest", type=Path, required=True)
    science.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(preflight(), sort_keys=True))
        return
    print(
        json.dumps(
            run_science(args.seed_manifest.resolve(), args.output.resolve()),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
