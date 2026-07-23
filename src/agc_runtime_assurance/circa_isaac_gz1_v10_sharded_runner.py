"""Versioned sharded exactly-once executor for the frozen CIRCA-GZ1-v10 design."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from .circa_gazebo_gz0 import SceneCandidate
from . import circa_gazebo_gz0_v9_runner as v9
from . import circa_isaac_gz1_v10_runner as base
from .circa_isaac_gz1_v10 import IsaacAirGroundEnvV10
from .circa_isaac_gz1_v10_schema import (
    DEFAULT_CAP_BYTES,
    DEFAULT_HORIZON,
    DEFAULT_ROLLOUTS,
    array_schema,
)


ROUTE = "CIRCA-GZ1-v10-SCI-PAR-R1"
EXPECTED_MANIFEST_SHA256 = "32d9f9d43798319a394fd19d4a487d447f32dcfd8e5bd7e67eddf457467bcab1"
EXPECTED_SEED_RECEIPT_SHA256 = "47acb941b4353a2a27bc24ce2278338c6bb82c8bb73d570b9336435c5fd846d4"
EXPECTED_SEED_VECTOR_SHA256 = "d2dbd2af4d391a6ccb66457346aa351d57d3f45c3257df649ec41998ccaca7f2"
EXPECTED_SCHEDULE_SHA256 = "256c3c7619e591ca8a7170da3fe7c338c6251171192e7c4b5191ef541e1936d8"
ALLOWED_SHARD_COUNTS = (8, 12)


class CircaIsaacGZ1V10ShardedError(RuntimeError):
    """A terminal aggregate or shard evidence error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise CircaIsaacGZ1V10ShardedError(f"stale temporary JSON exists: {temporary.name}")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def partition_indices(total: int, shard_count: int, shard_id: int) -> tuple[int, ...]:
    if total != DEFAULT_ROLLOUTS:
        raise CircaIsaacGZ1V10ShardedError("frozen rollout count drifted")
    if shard_count not in ALLOWED_SHARD_COUNTS:
        raise CircaIsaacGZ1V10ShardedError("shard count is not authorized")
    if not 0 <= shard_id < shard_count:
        raise CircaIsaacGZ1V10ShardedError("shard id is outside the frozen partition")
    return tuple(range(shard_id, total, shard_count))


def verify_partition(shard_count: int) -> dict[str, Any]:
    parts = [partition_indices(DEFAULT_ROLLOUTS, shard_count, shard_id) for shard_id in range(shard_count)]
    flattened = [index for part in parts for index in part]
    if len(flattened) != DEFAULT_ROLLOUTS or len(set(flattened)) != DEFAULT_ROLLOUTS:
        raise CircaIsaacGZ1V10ShardedError("partition is not disjoint and complete")
    if sorted(flattened) != list(range(DEFAULT_ROLLOUTS)):
        raise CircaIsaacGZ1V10ShardedError("partition does not cover the frozen schedule")
    sizes = [len(part) for part in parts]
    if max(sizes) - min(sizes) > 1:
        raise CircaIsaacGZ1V10ShardedError("partition is imbalanced")
    payload = json.dumps(parts, separators=(",", ":")).encode("utf-8")
    return {
        "shard_count": shard_count,
        "total_indices": len(flattened),
        "minimum_shard_size": min(sizes),
        "maximum_shard_size": max(sizes),
        "partition_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _allocate_arrays(rollouts: int) -> dict[str, np.ndarray]:
    arrays = {
        name: np.zeros(shape, dtype=dtype)
        for name, (dtype, shape) in array_schema(rollouts, DEFAULT_HORIZON).items()
    }
    for name in ("nominal_tube_feasible", "backup_tube_feasible", "terminal_reachability"):
        arrays[name].fill(-1)
    for name in ("worst_set_margin", "minimum_operational_margin_m", "minimum_hard_margin_m"):
        arrays[name].fill(np.nan)
    arrays["certificate_validity_interval"].fill(-1)
    return arrays


def _validate_arrays(arrays: Mapping[str, np.ndarray], rollouts: int) -> None:
    expected = array_schema(rollouts, DEFAULT_HORIZON)
    if set(arrays) != set(expected):
        raise CircaIsaacGZ1V10ShardedError("evidence member set drifted")
    for name, (dtype, shape) in expected.items():
        value = np.asarray(arrays[name])
        if value.dtype != dtype or value.shape != shape or value.dtype.hasobject:
            raise CircaIsaacGZ1V10ShardedError(f"evidence schema drifted: {name}")
    if not np.all(arrays["completed_steps"] == DEFAULT_HORIZON):
        raise CircaIsaacGZ1V10ShardedError("incomplete trajectory is unavailable evidence")
    if not np.all(arrays["trace_valid"]):
        raise CircaIsaacGZ1V10ShardedError("invalid trace is unavailable evidence")
    common = {name: arrays[name] for name in v9.array_schema(rollouts, DEFAULT_HORIZON)}
    common["scenario_index"] = np.zeros(rollouts, dtype="u1")
    common["scenario_seed"] = np.zeros(rollouts, dtype="<i4")
    v9.validate_arrays(common, rollouts, DEFAULT_HORIZON)


def _validate_frozen_inputs(
    manifest_path: Path,
    seed_receipt_path: Path,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], tuple[SceneCandidate, ...], Path, list[base.FactorialRun]]:
    if root not in manifest_path.parents or root not in seed_receipt_path.parents:
        raise CircaIsaacGZ1V10ShardedError("manifest or receipt escapes the repository")
    if _sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise CircaIsaacGZ1V10ShardedError("authorized manifest hash drifted")
    if _sha256(seed_receipt_path) != EXPECTED_SEED_RECEIPT_SHA256:
        raise CircaIsaacGZ1V10ShardedError("authorized seed receipt hash drifted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(seed_receipt_path.read_text(encoding="utf-8"))
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
        "seed_namespace": base.SEED_NAMESPACE,
        "prospective_rollouts": DEFAULT_ROLLOUTS,
        "horizon_steps": DEFAULT_HORIZON,
        "output_bytes_max": DEFAULT_CAP_BYTES,
        "seed_vector_sha256": EXPECTED_SEED_VECTOR_SHA256,
        "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise CircaIsaacGZ1V10ShardedError(f"manifest {key} drifted")
    if tuple(manifest.get("drivers", ())) != base.DRIVERS or tuple(manifest.get("methods", ())) != base.METHODS:
        raise CircaIsaacGZ1V10ShardedError("driver or method set drifted")
    receipt_required = {
        "status": "PASS_VERSIONED_NONSCIENTIFIC_SEED_RECEIPT_SALVAGE_R1",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "seed_vector_sha256": EXPECTED_SEED_VECTOR_SHA256,
        "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "derived_future_scenario_seeds": 1536,
        "schedule_entries": DEFAULT_ROLLOUTS,
        "new_seed_generated": False,
        "existing_seed_changed": False,
        "scientific_run_executed": False,
    }
    for key, expected in receipt_required.items():
        if receipt.get(key) != expected:
            raise CircaIsaacGZ1V10ShardedError(f"seed receipt {key} drifted")
    design_path = base._safe_child(root, manifest["design_manifest_path"])
    if _sha256(design_path) != manifest["design_manifest_sha256"]:
        raise CircaIsaacGZ1V10ShardedError("v10 design lock failed")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if not design.get("design_frozen"):
        raise CircaIsaacGZ1V10ShardedError("v10 design is not frozen")
    source_path = base._safe_child(root, manifest["candidate_source_path"])
    if _sha256(source_path) != manifest["candidate_source_sha256"]:
        raise CircaIsaacGZ1V10ShardedError("candidate source lock failed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if manifest.get("candidates") != source.get("candidates"):
        raise CircaIsaacGZ1V10ShardedError("candidate records drifted")
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
            raise CircaIsaacGZ1V10ShardedError(f"frozen scientific field drifted: {key}")
    active_namespace = Path(__file__).resolve().parents[2]
    frozen_prefix = Path("staging_f737e8c16f5c8866")
    for relative, expected in manifest.get("source_files", {}).items():
        relative_path = Path(relative)
        if _sha256(base._safe_child(root, relative)) != expected:
            raise CircaIsaacGZ1V10ShardedError(f"source lock failed: {relative}")
        try:
            active_relative = relative_path.relative_to(frozen_prefix)
        except ValueError as error:
            raise CircaIsaacGZ1V10ShardedError(f"unexpected source namespace: {relative}") from error
        active_path = (active_namespace / active_relative).resolve()
        if active_namespace not in active_path.parents or _sha256(active_path) != expected:
            raise CircaIsaacGZ1V10ShardedError(f"active copied source lock failed: {relative}")
    world = base._safe_child(root, manifest["world_path"])
    if _sha256(world) != manifest["world_sha256"]:
        raise CircaIsaacGZ1V10ShardedError("Isaac stage lock failed")
    candidates = tuple(SceneCandidate.from_dict(item) for item in manifest["candidates"])
    if len(candidates) != 12 or len({item.candidate_id for item in candidates}) != 12:
        raise CircaIsaacGZ1V10ShardedError("candidate cardinality drifted")
    schedule = base.compile_schedule(manifest)
    serialized_schedule = [
        [item.candidate_index, item.scenario_index, item.driver_index, item.method_index, item.hazard_active, item.scenario_seed]
        for item in schedule
    ]
    reproduced = hashlib.sha256(json.dumps(serialized_schedule, separators=(",", ":")).encode()).hexdigest()
    if reproduced != EXPECTED_SCHEDULE_SHA256:
        raise CircaIsaacGZ1V10ShardedError("frozen schedule hash does not reproduce")
    return manifest, source, candidates, world, schedule


def run_shard(
    manifest_path: str | Path,
    seed_receipt_path: str | Path,
    repo_root: str | Path,
    output_root: str | Path,
    shard_count: int,
    shard_id: int,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    output_parent = Path(output_root).resolve()
    if root != output_parent and root not in output_parent.parents:
        raise CircaIsaacGZ1V10ShardedError("scientific output root escapes repository")
    if not output_parent.is_dir():
        raise CircaIsaacGZ1V10ShardedError("aggregate scientific output root is absent")
    shard_path = output_parent / "shards" / f"shard_{shard_id:02d}"
    if shard_path.exists():
        raise CircaIsaacGZ1V10ShardedError("exactly-once shard target exists")
    manifest, design, candidates, world, schedule = _validate_frozen_inputs(
        Path(manifest_path).resolve(), Path(seed_receipt_path).resolve(), root
    )
    indices = partition_indices(len(schedule), shard_count, shard_id)
    shard_path.mkdir(parents=True, exist_ok=False)
    marker = shard_path / "SCIENTIFIC_SHARD_ATTEMPT_CONSUMED.json"
    marker.write_text(json.dumps({
        "aggregate_scientific_attempts_consumed": 1,
        "shard_id": shard_id,
        "shard_count": shard_count,
        "retry_allowed": False,
    }, sort_keys=True) + "\n", encoding="utf-8")
    state_path = shard_path / "execution_state.json"
    _write_json(state_path, {
        "status": "RUNNING_UNIQUE_PARALLEL_SCIENTIFIC_SHARD",
        "shard_id": shard_id,
        "shard_count": shard_count,
        "completed_rollouts": 0,
        "total_rollouts": len(indices),
        "aggregate_scientific_attempts_consumed": 1,
        "retry_allowed": False,
    })
    arrays = _allocate_arrays(len(indices))
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    env: IsaacAirGroundEnvV10 | None = None
    original_environment = v9._environment
    started = time.perf_counter()
    stage_hash = _sha256(world)
    backend_hash = hashlib.sha256(f"{base.ISAAC_VERSION}|{base.PHYSICS_BACKEND}".encode()).hexdigest()
    try:
        env = IsaacAirGroundEnvV10(app, world, horizon=DEFAULT_HORIZON)

        def isaac_environment(driver: str, _world: Path, candidate: SceneCandidate, source_design: Mapping[str, Any]):
            envelope = base.derive_operational_envelope(candidate, source_design["operational_envelope_assumptions"])
            limit = 0.5 * envelope.design_relative_speed_mps if driver == "planar_speed_projected_v4" else None
            assert env is not None
            env.configure(driver=driver, shift=candidate.shift(), planar_speed_limit_mps=limit)
            return env

        v9._environment = isaac_environment
        for local_row, global_row in enumerate(indices):
            item = schedule[global_row]
            spec = v9.FactorialRun(
                item.candidate_index,
                item.scenario_index,
                item.driver_index,
                item.method_index,
                item.hazard_active,
                item.scenario_seed,
            )
            v9._run_one(spec, local_row, arrays, candidates, world, design)
            base._fill_isaac_fields(arrays, local_row, item, candidates[item.candidate_index], env, stage_hash, backend_hash)
            if (local_row + 1) % 16 == 0 or local_row + 1 == len(indices):
                _write_json(state_path, {
                    "status": "RUNNING_UNIQUE_PARALLEL_SCIENTIFIC_SHARD",
                    "shard_id": shard_id,
                    "shard_count": shard_count,
                    "completed_rollouts": local_row + 1,
                    "total_rollouts": len(indices),
                    "aggregate_scientific_attempts_consumed": 1,
                    "retry_allowed": False,
                    "elapsed_seconds": time.perf_counter() - started,
                })
        _validate_arrays(arrays, len(indices))
        archive_path = shard_path / "trace_arrays.npz"
        np.savez_compressed(archive_path, **{name: arrays[name] for name in sorted(arrays)})
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays):
                raise CircaIsaacGZ1V10ShardedError("saved shard member set drifted")
            for name, expected in arrays.items():
                restored = archive[name]
                if restored.dtype != expected.dtype or restored.shape != expected.shape:
                    raise CircaIsaacGZ1V10ShardedError(f"saved shard schema drifted: {name}")
                if not np.array_equal(restored, expected, equal_nan=True):
                    raise CircaIsaacGZ1V10ShardedError(f"lossless shard verification failed: {name}")
        summary = {
            "status": "COMPLETE_CIRCA_GZ1_V10_SCI_PAR_R1_SHARD_AWAITING_AGGREGATION",
            "route_name": ROUTE,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "seed_receipt_sha256": EXPECTED_SEED_RECEIPT_SHA256,
            "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
            "shard_id": shard_id,
            "shard_count": shard_count,
            "rollouts": len(indices),
            "first_global_schedule_index": indices[0],
            "last_global_schedule_index": indices[-1],
            "horizon_steps": DEFAULT_HORIZON,
            "aggregate_scientific_attempts_consumed": 1,
            "retry_allowed": False,
            "elapsed_seconds": time.perf_counter() - started,
            "trace_archive_bytes": archive_path.stat().st_size,
            "trace_archive_sha256": _sha256(archive_path),
            "claim_allowed": False,
        }
        _write_json(shard_path / "result.json", summary)
        _write_json(state_path, {**summary, "status": "COMPLETE_UNIQUE_PARALLEL_SCIENTIFIC_SHARD"})
        return summary
    except Exception as error:
        _write_json(state_path, {
            "status": "TERMINAL_PARALLEL_SCIENTIFIC_SHARD_FAILURE_NO_RETRY",
            "shard_id": shard_id,
            "shard_count": shard_count,
            "aggregate_scientific_attempts_consumed": 1,
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


def aggregate_shards(output_root: str | Path, shard_count: int) -> dict[str, Any]:
    output = Path(output_root).resolve()
    marker = output / "SCIENTIFIC_PARALLEL_ATTEMPT_CONSUMED.json"
    if not output.is_dir() or not marker.is_file():
        raise CircaIsaacGZ1V10ShardedError("aggregate output or attempt marker is absent")
    if (output / "trace_arrays.npz").exists() or (output / "result.json").exists():
        raise CircaIsaacGZ1V10ShardedError("aggregate output already exists")
    partition = verify_partition(shard_count)
    full = base.allocate_arrays()
    full_schema = array_schema(DEFAULT_ROLLOUTS, DEFAULT_HORIZON)
    shard_manifest: list[dict[str, Any]] = []
    global_initialized: set[str] = set()
    for shard_id in range(shard_count):
        indices = np.asarray(partition_indices(DEFAULT_ROLLOUTS, shard_count, shard_id), dtype=np.int64)
        shard_path = output / "shards" / f"shard_{shard_id:02d}"
        result_path = shard_path / "result.json"
        archive_path = shard_path / "trace_arrays.npz"
        if not result_path.is_file() or not archive_path.is_file():
            raise CircaIsaacGZ1V10ShardedError(f"shard {shard_id} evidence is incomplete")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        required = {
            "status": "COMPLETE_CIRCA_GZ1_V10_SCI_PAR_R1_SHARD_AWAITING_AGGREGATION",
            "route_name": ROUTE,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "seed_receipt_sha256": EXPECTED_SEED_RECEIPT_SHA256,
            "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
            "shard_id": shard_id,
            "shard_count": shard_count,
            "rollouts": len(indices),
            "horizon_steps": DEFAULT_HORIZON,
            "aggregate_scientific_attempts_consumed": 1,
            "retry_allowed": False,
        }
        for key, expected in required.items():
            if result.get(key) != expected:
                raise CircaIsaacGZ1V10ShardedError(f"shard {shard_id} result drifted: {key}")
        archive_sha256 = _sha256(archive_path)
        if archive_sha256 != result.get("trace_archive_sha256"):
            raise CircaIsaacGZ1V10ShardedError(f"shard {shard_id} archive hash drifted")
        shard_schema = array_schema(len(indices), DEFAULT_HORIZON)
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != set(full_schema):
                raise CircaIsaacGZ1V10ShardedError(f"shard {shard_id} member set drifted")
            for name, (dtype, shape) in shard_schema.items():
                value = archive[name]
                if value.dtype != dtype or value.shape != shape or value.dtype.hasobject:
                    raise CircaIsaacGZ1V10ShardedError(f"shard {shard_id} schema drifted: {name}")
                if shape != full_schema[name][1]:
                    full[name][indices] = value
                elif name not in global_initialized:
                    full[name][...] = value
                    global_initialized.add(name)
                elif not np.array_equal(full[name], value, equal_nan=True):
                    raise CircaIsaacGZ1V10ShardedError(f"global field differs across shards: {name}")
        shard_manifest.append({
            "shard_id": shard_id,
            "rollouts": len(indices),
            "result_sha256": _sha256(result_path),
            "trace_archive_sha256": archive_sha256,
            "trace_archive_bytes": archive_path.stat().st_size,
        })
    base.validate_arrays(full)
    archive_path = output / "trace_arrays.npz"
    np.savez_compressed(archive_path, **{name: full[name] for name in sorted(full)})
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != set(full):
            raise CircaIsaacGZ1V10ShardedError("aggregate member set drifted")
        for name, expected in full.items():
            restored = archive[name]
            if restored.dtype != expected.dtype or restored.shape != expected.shape:
                raise CircaIsaacGZ1V10ShardedError(f"aggregate schema drifted: {name}")
            if not np.array_equal(restored, expected, equal_nan=True):
                raise CircaIsaacGZ1V10ShardedError(f"lossless aggregate verification failed: {name}")
    manifest_payload = {
        "status": "PASS_DISJOINT_COMPLETE_SHARD_AGGREGATION",
        "route_name": ROUTE,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "seed_receipt_sha256": EXPECTED_SEED_RECEIPT_SHA256,
        "seed_vector_sha256": EXPECTED_SEED_VECTOR_SHA256,
        "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        **partition,
        "shards": shard_manifest,
        "intermediate_shard_archives_removed_after_lossless_aggregate": True,
    }
    _write_json(output / "shard_manifest.json", manifest_payload)
    for shard_id in range(shard_count):
        (output / "shards" / f"shard_{shard_id:02d}" / "trace_arrays.npz").unlink()
    summary = {
        "status": "COMPLETE_CIRCA_GZ1_V10_SCI_PAR_R1_AWAITING_FROZEN_ANALYSIS",
        "route_name": ROUTE,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "seed_receipt_sha256": EXPECTED_SEED_RECEIPT_SHA256,
        "seed_vector_sha256": EXPECTED_SEED_VECTOR_SHA256,
        "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "shard_count": shard_count,
        "rollouts": DEFAULT_ROLLOUTS,
        "horizon_steps": DEFAULT_HORIZON,
        "scientific_attempts_consumed": 1,
        "retry_allowed": False,
        "scientific_run_executed": True,
        "trace_archive_bytes": archive_path.stat().st_size,
        "trace_archive_sha256": _sha256(archive_path),
        "shard_manifest_sha256": _sha256(output / "shard_manifest.json"),
        "claim_allowed": False,
        "next_gate": "frozen_analysis_and_claim_evidence_audit",
    }
    _write_json(output / "result.json", summary)
    total = sum(item.stat().st_size for item in output.rglob("*") if item.is_file())
    if total > DEFAULT_CAP_BYTES:
        raise CircaIsaacGZ1V10ShardedError("aggregate scientific output exceeds frozen capacity")
    _write_json(output / "execution_state.json", {**summary, "status": "COMPLETE_UNIQUE_PARALLEL_SCIENTIFIC_ATTEMPT"})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-shard", action="store_true")
    mode.add_argument("--aggregate", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--seed-receipt", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--shard-id", type=int)
    args = parser.parse_args()
    if args.run_shard:
        if args.manifest is None or args.seed_receipt is None or args.repo_root is None or args.shard_id is None:
            raise SystemExit("shard execution requires manifest, receipt, repo root and shard id")
        result = run_shard(
            args.manifest,
            args.seed_receipt,
            args.repo_root,
            args.output_root,
            args.shard_count,
            args.shard_id,
        )
    else:
        if args.shard_id is not None:
            raise SystemExit("aggregation does not accept a shard id")
        result = aggregate_shards(args.output_root, args.shard_count)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
