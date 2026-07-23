"""Independent successor runner for claim-ineligible CIRCA Gazebo GZ0-v6."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

from .circa_gazebo_gz0_v5 import (
    CONTROLLERS,
    DRIVERS,
    TRACE_FIELDS,
    _diagnostic,
    _run_regime,
    validate_design_manifest as validate_v5_core_design,
)


class CircaGazeboGZ0V6Error(RuntimeError):
    pass


SEED_NAMESPACE = "circa-gz0-v6"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scenario_seeds(manifest: dict[str, Any]) -> tuple[int, ...]:
    seeds = []
    for candidate in manifest["candidates"]:
        for index in range(int(manifest["seeds_per_candidate"])):
            digest = hashlib.sha256(
                f"{SEED_NAMESPACE}|{manifest['master_seed']}|{candidate['candidate_id']}|{index}".encode()
            ).digest()
            seeds.append(int.from_bytes(digest[:8], "big") % (2**31 - 1))
    return tuple(seeds)


def validate_design_manifest(manifest: dict[str, Any], root: Path):
    required = {
        "stage": "claim_ineligible_gazebo_gz0_v6_independent_successor",
        "route_authorized": True,
        "scientific_claim_allowed": False,
        "circa_gz1_authorized": False,
        "retry_allowed": False,
        "seed_top_up_allowed": False,
        "sealed_data_authorized": False,
        "formal_experiment_authorized": False,
        "gpu_count": 0,
        "resource_class": "CPU-SHARED",
        "route": "module_entrypoint_verified_factorial_speed_projection_by_predictive_shield",
        "prior_result_reuse_allowed": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise CircaGazeboGZ0V6Error(f"manifest {key} must equal {expected!r}")
    if manifest.get("predecessor_terminal_state") != "TERMINAL_STOP_PRE_SCIENTIFIC_IMPORT_ERROR_NO_RETRY":
        raise CircaGazeboGZ0V6Error("v5 terminal predecessor state is not frozen")
    if manifest.get("seed_namespace") != SEED_NAMESPACE:
        raise CircaGazeboGZ0V6Error("v6 seed namespace mismatch")
    proxy = dict(manifest)
    proxy["stage"] = "claim_ineligible_gazebo_gz0_v5_factorial_mechanism_development"
    proxy["route"] = "factorial_speed_projection_by_predictive_shield_mechanism_screen"
    try:
        world, candidates = validate_v5_core_design(proxy, root)
    except Exception as exc:
        raise CircaGazeboGZ0V6Error(str(exc)) from exc
    seeds = scenario_seeds(manifest)
    if len(seeds) != 60 or len(set(seeds)) != 60:
        raise CircaGazeboGZ0V6Error("v6 seed block must contain 60 unique independent seeds")
    expected = len(candidates) * int(manifest["seeds_per_candidate"]) * len(DRIVERS) * len(CONTROLLERS)
    if expected != 360 or expected != int(manifest["max_regime_rollouts"]):
        raise CircaGazeboGZ0V6Error("v6 factorial schedule size mismatch")
    return world, candidates


def validate_runnable_manifest(manifest: dict[str, Any], root: Path):
    world, candidates = validate_design_manifest(manifest, root)
    required = {
        "status": "authorized_exactly_once",
        "scientific_run_authorized": True,
        "exactly_once_authorization": True,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise CircaGazeboGZ0V6Error(f"runnable manifest {key} must equal {expected!r}")
    return world, candidates


def run_gz0_v6(manifest_path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    if root not in manifest_file.parents or not manifest_file.is_file():
        raise CircaGazeboGZ0V6Error("manifest path is unsafe or absent")
    manifest_raw = manifest_file.read_bytes()
    manifest = json.loads(manifest_raw)
    world, candidates = validate_runnable_manifest(manifest, root)
    output = (root / manifest["output_path"]).resolve()
    if root not in output.parents or output.exists():
        raise CircaGazeboGZ0V6Error("exactly-once v6 output path already exists or is unsafe")
    hazard_indices = {int(value) for value in manifest["hazard_active_seed_indices"]}
    schedule = []
    seeds = iter(scenario_seeds(manifest))
    for candidate in candidates:
        for index in range(int(manifest["seeds_per_candidate"])):
            seed = next(seeds)
            for driver in DRIVERS:
                for controller in CONTROLLERS:
                    schedule.append((candidate, seed, index in hazard_indices, driver, controller))
    if len(schedule) != int(manifest["max_regime_rollouts"]):
        raise CircaGazeboGZ0V6Error("generated v6 schedule does not match frozen budget")
    random.Random(int(manifest["schedule_seed"])).shuffle(schedule)
    start = time.perf_counter()
    rows = []
    for candidate, seed, hazard_active, driver, controller in schedule:
        if time.perf_counter() - start > manifest["wall_time_seconds_max"]:
            raise CircaGazeboGZ0V6Error("GZ0-v6 wall-time budget exceeded")
        rows.append(_run_regime(candidate, seed, hazard_active, driver, controller, world, manifest))
    elapsed = time.perf_counter() - start
    manipulation, diagnostic = _diagnostic(rows, tuple(manifest["families"]), manifest)
    all_families = all(value["passing_candidate_count"] > 0 for value in diagnostic.values())
    result = {
        "result_id": manifest["manifest_id"],
        "claim_eligible": False,
        "circa_gz1_run": False,
        "sealed_data_used": False,
        "formal_experiment_run": False,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "world_sha256": manifest["world_sha256"],
        "seed_namespace": SEED_NAMESPACE,
        "independent_seed_count": 60,
        "rows": rows,
        "factorial_manipulation": manipulation,
        "development_diagnostic": diagnostic,
        "all_families_have_passing_candidate": all_families,
        "route_gate_pass": bool(manipulation["pass"] and all_families),
        "elapsed_seconds": elapsed,
        "boundary": "claim-ineligible GZ0-v6 independent successor development only; no GZ1 authorization",
    }
    body = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if len(body) > manifest["output_bytes_max"]:
        raise CircaGazeboGZ0V6Error("GZ0-v6 output budget exceeded")
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_bytes(body)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    result = run_gz0_v6(args.manifest, args.repo_root)
    print(json.dumps({
        "result_id": result["result_id"],
        "rows": len(result["rows"]),
        "elapsed_seconds": result["elapsed_seconds"],
        "factorial_manipulation_pass": result["factorial_manipulation"]["pass"],
        "all_families_have_passing_candidate": result["all_families_have_passing_candidate"],
        "route_gate_pass": result["route_gate_pass"],
        "claim_eligible": False,
        "circa_gz1_run": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
