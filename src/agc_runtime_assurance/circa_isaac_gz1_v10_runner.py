"""Fail-closed exactly-once runner for the frozen CIRCA-GZ1-v10 route."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np

from .circa_gazebo_gz0 import SceneCandidate
from .circa_gazebo_gz0_v3 import derive_operational_envelope
from . import circa_gazebo_gz0_v9_runner as v9
from .circa_isaac_gz1_v10 import IsaacAirGroundEnvV10
from .circa_isaac_gz1_v10_schema import (
    DEFAULT_CAP_BYTES,
    DEFAULT_HORIZON,
    DEFAULT_ROLLOUTS,
    array_schema,
)


DRIVERS = v9.DRIVERS
METHODS = v9.METHODS
SEED_NAMESPACE = "circa-gz1-v10"
ISAAC_VERSION = "6.0.1-rc.7+release.42383.32955d8d.gl"
PHYSICS_BACKEND = "PhysX"


class CircaIsaacGZ1V10RunnerError(RuntimeError):
    """A terminal, non-retryable v10 execution error."""


@dataclass(frozen=True)
class FactorialRun:
    candidate_index: int
    scenario_index: int
    driver_index: int
    method_index: int
    hazard_active: bool
    scenario_seed: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise CircaIsaacGZ1V10RunnerError(f"path escapes repository: {relative}")
    return path


def _seed(master_seed: int, candidate_id: str, scenario_index: int) -> int:
    if master_seed < 0 or scenario_index < 0 or not candidate_id:
        raise CircaIsaacGZ1V10RunnerError("invalid frozen seed input")
    digest = hashlib.sha256(
        f"{SEED_NAMESPACE}|{master_seed}|{candidate_id}|{scenario_index}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def compile_schedule(manifest: Mapping[str, Any]) -> list[FactorialRun]:
    if manifest.get("seed_namespace") != SEED_NAMESPACE:
        raise CircaIsaacGZ1V10RunnerError("seed namespace drifted")
    candidates = manifest["candidates"]
    schedule: list[FactorialRun] = []
    for candidate_index, candidate in enumerate(candidates):
        for scenario_index in range(128):
            # Each frozen 32-unit domain block contains 16 hazard and 16 negative units.
            hazard = scenario_index % 32 < 16
            scenario_seed = _seed(
                int(manifest["master_seed"]), candidate["candidate_id"], scenario_index
            )
            for driver_index in range(len(DRIVERS)):
                for method_index in range(len(METHODS)):
                    schedule.append(
                        FactorialRun(
                            candidate_index,
                            scenario_index,
                            driver_index,
                            method_index,
                            hazard,
                            scenario_seed,
                        )
                    )
    if len(schedule) != DEFAULT_ROLLOUTS:
        raise CircaIsaacGZ1V10RunnerError("frozen factorial size drifted")
    random.Random(int(manifest["schedule_seed"])).shuffle(schedule)
    return schedule


def _validate_manifest(
    manifest: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], tuple[SceneCandidate, ...], Path, Path]:
    required = {
        "status": "AUTHORIZED_EXACTLY_ONCE_CIRCA_GZ1_V10",
        "route_name": "CIRCA-GZ1-v10",
        "design_frozen": True,
        "scientific_run_authorized": True,
        "scientific_seed_material_generated": True,
        "scientific_output_authorized": True,
        "exactly_once_authorization": True,
        "retry_allowed": False,
        "seed_top_up_allowed": False,
        "sealed_data_authorized": False,
        "formal_experiment_authorized": False,
        "g2_authorized": False,
        "px4_hardware_field_authorized": False,
        "seed_namespace": SEED_NAMESPACE,
        "prospective_rollouts": DEFAULT_ROLLOUTS,
        "horizon_steps": DEFAULT_HORIZON,
        "output_bytes_max": DEFAULT_CAP_BYTES,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise CircaIsaacGZ1V10RunnerError(f"manifest {key} drifted")
    if tuple(manifest.get("drivers", ())) != DRIVERS:
        raise CircaIsaacGZ1V10RunnerError("driver set drifted")
    if tuple(manifest.get("methods", ())) != METHODS:
        raise CircaIsaacGZ1V10RunnerError("method set drifted")
    design_path = _safe_child(root, manifest["design_manifest_path"])
    if _sha256(design_path) != manifest["design_manifest_sha256"]:
        raise CircaIsaacGZ1V10RunnerError("v10 design lock failed")
    frozen_design = json.loads(design_path.read_text(encoding="utf-8"))
    if not frozen_design.get("design_frozen"):
        raise CircaIsaacGZ1V10RunnerError("v10 design is not frozen")
    source_path = _safe_child(root, manifest["candidate_source_path"])
    if _sha256(source_path) != manifest["candidate_source_sha256"]:
        raise CircaIsaacGZ1V10RunnerError("candidate source lock failed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if manifest.get("candidates") != source.get("candidates"):
        raise CircaIsaacGZ1V10RunnerError("candidate records are not byte-semantic copies")
    immutable = {
        "drivers": source["drivers"],
        "methods": source["regimes_per_driver"],
        "horizon_steps": source["horizon_steps"],
        "operational_envelope_assumptions": source["operational_envelope_assumptions"],
        "design_speed_tolerance_mps": source["design_speed_tolerance_mps"],
        "development_gates": source["development_gates"],
    }
    for key, expected in immutable.items():
        if manifest.get(key) != expected:
            raise CircaIsaacGZ1V10RunnerError(f"frozen scientific field drifted: {key}")
    for relative, expected in manifest.get("source_files", {}).items():
        if _sha256(_safe_child(root, relative)) != expected:
            raise CircaIsaacGZ1V10RunnerError(f"source lock failed: {relative}")
    world = _safe_child(root, manifest["world_path"])
    if _sha256(world) != manifest["world_sha256"]:
        raise CircaIsaacGZ1V10RunnerError("Isaac stage lock failed")
    output = _safe_child(root, manifest["output_path"])
    if output.exists():
        raise CircaIsaacGZ1V10RunnerError("exactly-once scientific target exists")
    candidates = tuple(SceneCandidate.from_dict(item) for item in manifest["candidates"])
    if len(candidates) != 12 or len({item.candidate_id for item in candidates}) != 12:
        raise CircaIsaacGZ1V10RunnerError("candidate cardinality drifted")
    return source, candidates, world, output


def allocate_arrays() -> dict[str, np.ndarray]:
    arrays = {
        name: np.zeros(shape, dtype=dtype)
        for name, (dtype, shape) in array_schema(DEFAULT_ROLLOUTS, DEFAULT_HORIZON).items()
    }
    for name in ("nominal_tube_feasible", "backup_tube_feasible", "terminal_reachability"):
        arrays[name].fill(-1)
    arrays["worst_set_margin"].fill(np.nan)
    arrays["certificate_validity_interval"].fill(-1)
    arrays["minimum_operational_margin_m"].fill(np.nan)
    arrays["minimum_hard_margin_m"].fill(np.nan)
    return arrays


def validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    expected = array_schema(DEFAULT_ROLLOUTS, DEFAULT_HORIZON)
    if set(arrays) != set(expected):
        raise CircaIsaacGZ1V10RunnerError("evidence member set drifted")
    for name, (dtype, shape) in expected.items():
        value = np.asarray(arrays[name])
        if value.dtype != dtype or value.shape != shape or value.dtype.hasobject:
            raise CircaIsaacGZ1V10RunnerError(f"evidence schema drifted: {name}")
    if not np.all(arrays["completed_steps"] == DEFAULT_HORIZON):
        raise CircaIsaacGZ1V10RunnerError("incomplete trajectory is unavailable evidence")
    if not np.all(arrays["trace_valid"]):
        raise CircaIsaacGZ1V10RunnerError("invalid trace is unavailable evidence")
    common = {
        name: arrays[name]
        for name in v9.array_schema(DEFAULT_ROLLOUTS, DEFAULT_HORIZON)
    }
    # v10 widens these two provenance integers for 128 future units and a
    # disjoint 63-bit seed namespace.  The inherited v9 consistency validator
    # does not inspect their values, so compatible sentinels preserve its
    # method/trace checks without narrowing scientific provenance.
    common["scenario_index"] = np.zeros(DEFAULT_ROLLOUTS, dtype="u1")
    common["scenario_seed"] = np.zeros(DEFAULT_ROLLOUTS, dtype="<i4")
    v9.validate_arrays(common, DEFAULT_ROLLOUTS, DEFAULT_HORIZON)


def _fill_isaac_fields(
    arrays: dict[str, np.ndarray], row: int, spec: FactorialRun,
    candidate: SceneCandidate, env: IsaacAirGroundEnvV10,
    stage_hash: str, backend_hash: str,
) -> None:
    if len(env.body_state_history) != DEFAULT_HORIZON or len(env.contact_impulse_history) != DEFAULT_HORIZON:
        raise CircaIsaacGZ1V10RunnerError("Isaac trajectory evidence is incomplete")
    arrays["physics_time_s"][row] = (np.arange(DEFAULT_HORIZON) + 1) * env.dt
    arrays["isaac_body_state"][row] = np.asarray(env.body_state_history)
    arrays["isaac_contact_impulse"][row] = np.asarray(env.contact_impulse_history)
    arrays["solver_substeps"][row].fill(10)
    arrays["physics_backend_hash"][:] = np.frombuffer(bytes.fromhex(backend_hash), dtype=np.uint8)
    arrays["stage_hash"][:] = np.frombuffer(bytes.fromhex(stage_hash), dtype=np.uint8)
    block = spec.scenario_index // 32
    arrays["stress_block_index"][row] = block
    arrays["sensor_bias_xy_m"][row].fill(candidate.sensor_bias)
    arrays["observation_delay_steps_by_agent"][row].fill(candidate.observation_delay_steps)
    arrays["communication_delay_steps_by_direction"][row].fill(candidate.communication_delay_steps)
    arrays["communication_dropout_rate"][row] = 0.0
    lag_steps = int(np.floor(candidate.actuator_lag * 10.0 + 0.5))
    arrays["actuation_lag_steps_by_agent"][row].fill(lag_steps)
    arrays["mass_scale_by_agent"][row] = (candidate.uav_mass, 1.0)
    arrays["friction_scale_by_agent"][row] = (0.0, candidate.ugv_friction)
    arrays["drag_scale_by_agent"][row] = (candidate.uav_drag, 0.0)
    arrays["wind_xy_mps"][row] = (0.0, 0.0)
    arrays["witness_radius_scale"][row] = 1.0
    arrays["randomization_block_index"][row] = spec.candidate_index * 4 + block
    arrays["pair_id"][row] = spec.candidate_index * 128 + spec.scenario_index


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def preflight_world(world_path: str | Path) -> dict[str, Any]:
    """Load the frozen world without generating scientific seeds or output."""
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    env = None
    try:
        env = IsaacAirGroundEnvV10(app, world_path, horizon=DEFAULT_HORIZON)
        return {
            "status": "PASS_NON_SCIENTIFIC_SIMULATIONAPP_WORLD_LOAD_PREFLIGHT",
            "scientific_seed_material_generated": False,
            "scientific_output_generated": False,
            "scientific_run_executed": False,
            "isaac_version": ISAAC_VERSION,
            "physics_backend": PHYSICS_BACKEND,
        }
    finally:
        if env is not None:
            env.close()
        app.close()


def run_exactly_once(manifest_path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    path = Path(manifest_path).resolve()
    if root not in path.parents or not path.is_file():
        raise CircaIsaacGZ1V10RunnerError("unsafe or absent runnable manifest")
    raw = path.read_bytes()
    manifest = json.loads(raw)
    design, candidates, world, output = _validate_manifest(manifest, root)
    schedule = compile_schedule(manifest)
    arrays = allocate_arrays()
    output.mkdir(parents=True, exist_ok=False)
    attempt = output / "SCIENTIFIC_ATTEMPT_CONSUMED.json"
    fd = os.open(attempt, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, b'{"scientific_attempts_consumed":1,"retry_allowed":false}\n')
    os.close(fd)
    state_path = output / "execution_state.json"
    _write_json(state_path, {
        "status": "RUNNING_UNIQUE_SCIENTIFIC_ATTEMPT",
        "completed_rollouts": 0,
        "total_rollouts": len(schedule),
        "scientific_attempts_consumed": 1,
        "retry_allowed": False,
    })

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    env: IsaacAirGroundEnvV10 | None = None
    original_environment = v9._environment
    started = time.perf_counter()
    stage_hash = _sha256(world)
    backend_hash = hashlib.sha256(f"{ISAAC_VERSION}|{PHYSICS_BACKEND}".encode()).hexdigest()
    try:
        env = IsaacAirGroundEnvV10(app, world, horizon=DEFAULT_HORIZON)

        def isaac_environment(driver: str, _world: Path, candidate: SceneCandidate, source: Mapping[str, Any]):
            envelope = derive_operational_envelope(candidate, source["operational_envelope_assumptions"])
            limit = 0.5 * envelope.design_relative_speed_mps if driver == "planar_speed_projected_v4" else None
            assert env is not None
            env.configure(driver=driver, shift=candidate.shift(), planar_speed_limit_mps=limit)
            return env

        v9._environment = isaac_environment
        for row, item in enumerate(schedule):
            spec = v9.FactorialRun(
                item.candidate_index, item.scenario_index, item.driver_index,
                item.method_index, item.hazard_active, item.scenario_seed,
            )
            v9._run_one(spec, row, arrays, candidates, world, design)
            _fill_isaac_fields(arrays, row, item, candidates[item.candidate_index], env, stage_hash, backend_hash)
            if (row + 1) % 32 == 0 or row + 1 == len(schedule):
                _write_json(state_path, {
                    "status": "RUNNING_UNIQUE_SCIENTIFIC_ATTEMPT",
                    "completed_rollouts": row + 1,
                    "total_rollouts": len(schedule),
                    "scientific_attempts_consumed": 1,
                    "retry_allowed": False,
                    "elapsed_seconds": time.perf_counter() - started,
                })
        validate_arrays(arrays)
        archive_path = output / "trace_arrays.npz"
        np.savez_compressed(archive_path, **{name: arrays[name] for name in sorted(arrays)})
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays):
                raise CircaIsaacGZ1V10RunnerError("saved archive member set drifted")
            for name, expected in arrays.items():
                restored = archive[name]
                if restored.dtype != expected.dtype or restored.shape != expected.shape:
                    raise CircaIsaacGZ1V10RunnerError(f"saved archive schema drifted: {name}")
                if not np.array_equal(restored, expected, equal_nan=True):
                    raise CircaIsaacGZ1V10RunnerError(f"lossless archive verification failed: {name}")
        elapsed = time.perf_counter() - started
        summary = {
            "status": "COMPLETE_CIRCA_GZ1_V10_AWAITING_FROZEN_ANALYSIS",
            "route_name": "CIRCA-GZ1-v10",
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "rollouts": len(schedule),
            "horizon_steps": DEFAULT_HORIZON,
            "scientific_attempts_consumed": 1,
            "retry_allowed": False,
            "scientific_run_executed": True,
            "elapsed_seconds": elapsed,
            "trace_archive_bytes": archive_path.stat().st_size,
            "trace_archive_sha256": _sha256(archive_path),
            "claim_allowed": False,
            "next_gate": "frozen_analysis_and_claim_evidence_audit",
        }
        _write_json(output / "result.json", summary)
        total = sum(item.stat().st_size for item in output.iterdir() if item.is_file())
        if total > DEFAULT_CAP_BYTES:
            raise CircaIsaacGZ1V10RunnerError("scientific output exceeds frozen capacity")
        _write_json(state_path, {**summary, "status": "COMPLETE_UNIQUE_SCIENTIFIC_ATTEMPT"})
        return summary
    except Exception as error:
        _write_json(state_path, {
            "status": "TERMINAL_SCIENTIFIC_OR_EVIDENCE_FAILURE_NO_RETRY",
            "scientific_attempts_consumed": 1,
            "retry_allowed": False,
            "error_type": type(error).__name__,
            "error": str(error),
        })
        raise
    finally:
        v9._environment = original_environment
        if env is not None:
            env.close()
        app.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--world", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.run:
        raise SystemExit("select exactly one of --preflight or --run")
    if args.preflight:
        if args.world is None:
            raise SystemExit("--world is required for preflight")
        print(json.dumps(preflight_world(args.world), sort_keys=True))
    else:
        if args.repo_root is None or args.manifest is None:
            raise SystemExit("--repo-root and --manifest are required for run")
        print(json.dumps(run_exactly_once(args.manifest, args.repo_root), sort_keys=True))


if __name__ == "__main__":
    main()
