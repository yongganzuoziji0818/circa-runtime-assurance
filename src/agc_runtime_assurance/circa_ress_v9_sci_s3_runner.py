"""S3 bounded-memory execution successor for the frozen CIRCA-RESS-V9 design.

The scientific schedule, seeds, methods, candidates, endpoints, thresholds,
schema, and final archive remain unchanged.  Gazebo TestFixture instances are
bounded to a versioned worker process and therefore released at every segment
boundary.  Full-size scientific arrays are disk-backed in the supervisor.
Unlike S2, workers validate the already-created supervisor claim rather than
re-running the pre-claim target-absence gate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np

from . import circa_ress_v9_runner as _v9


SUCCESSOR_ID = "CIRCA-RESS-V9-SCI-S3"
SEGMENT_ROLLOUTS = 64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _allocate_segment(count: int) -> dict[str, np.ndarray]:
    arrays = {
        name: np.zeros(shape, dtype=dtype)
        for name, (dtype, shape) in _v9.array_schema(count, _v9.HORIZON).items()
    }
    for name in (
        "nominal_tube_feasible",
        "backup_tube_feasible",
        "terminal_reachability",
    ):
        arrays[name].fill(-1)
    for name in (
        "operational_margin_m",
        "hard_margin_m",
        "minimum_operational_margin_m",
        "minimum_hard_margin_m",
    ):
        arrays[name].fill(np.nan)
    return arrays


def validate_worker_claim(
    manifest_path: Path, repo_root: Path
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = _sha256(manifest_path)
    output = (repo_root / str(manifest.get("output_path", ""))).resolve()
    claim_path = output / "ATTEMPT_STARTED.json"
    if not claim_path.is_file():
        raise _v9.DiversityRunnerError("S3 supervisor claim is absent")
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    fixture = manifest.get("non_scientific_fixture") is True
    expected_status = (
        "NONSCIENTIFIC_FIXTURE_CLAIM"
        if fixture
        else "SCIENTIFIC_ATTEMPT_STARTED"
    )
    expected_attempts = 0 if fixture else 1
    if (
        claim.get("execution_successor_id") != SUCCESSOR_ID
        or claim.get("manifest_sha256") != manifest_sha
        or claim.get("status") != expected_status
        or claim.get("scientific_attempts_consumed") != expected_attempts
        or claim.get("retry_allowed") is not False
    ):
        raise _v9.DiversityRunnerError("S3 supervisor claim identity mismatch")
    return manifest


def _worker(
    manifest_path: Path,
    repo_root: Path,
    start: int,
    stop: int,
    segment_path: Path,
) -> None:
    manifest = validate_worker_claim(manifest_path, repo_root)
    with _v9._configured_core():
        design = json.loads(
            (repo_root / str(manifest["design_path"])).read_text(encoding="utf-8")
        )
        operationalization = json.loads(
            (repo_root / str(manifest["operationalization_path"])).read_text(
                encoding="utf-8"
            )
        )
        registry = json.loads(
            (repo_root / str(manifest["world_registry_path"])).read_text(
                encoding="utf-8"
            )
        )
        schedule = _v9.compile_schedule(manifest)
        arrays = _allocate_segment(stop - start)
        _v9._core._write_hash(
            arrays["world_sha256"], _v9._core.WORLD_REGISTRY_SHA256
        )
        _v9._core._write_hash(
            arrays["adapter_source_sha256"],
            manifest["source_hashes"][
                "src/agc_runtime_assurance/circa_ress_v8_sim_diversity_adapter.py"
            ],
        )
        _v9._core._write_hash(
            arrays["schema_source_sha256"],
            manifest["source_hashes"][
                "src/agc_runtime_assurance/circa_ress_v8_sim_diversity_schema.py"
            ],
        )
        _v9._core._write_hash(arrays["codebook_hash"], _v9._core.codebook_hash())
        for local_row, schedule_row in enumerate(range(start, stop)):
            _v9._core._run_one(
                schedule[schedule_row],
                local_row,
                arrays,
                repo_root,
                design,
                operationalization,
                registry,
            )
        if np.any(arrays["completed_steps"] != _v9.HORIZON):
            raise _v9.DiversityRunnerError("segment contains an incomplete rollout")
        segment_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = segment_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle, **{name: arrays[name] for name in sorted(arrays)}
            )
        os.replace(temporary, segment_path)


def _allocate_memmaps(root: Path) -> dict[str, np.memmap]:
    root.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, np.memmap] = {}
    for name, (dtype, shape) in _v9.array_schema(
        _v9.ROLLOUTS, _v9.HORIZON
    ).items():
        arrays[name] = np.lib.format.open_memmap(
            root / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )
    return arrays


def _merge_segment(
    arrays: Mapping[str, np.memmap], segment_path: Path, start: int, stop: int
) -> None:
    with np.load(segment_path, allow_pickle=False) as segment:
        for name, target in arrays.items():
            source = segment[name]
            if target.ndim == 1 and target.shape == (32,):
                target[...] = source
            elif target.shape[0] == _v9.ROLLOUTS:
                target[start:stop] = source
            else:
                target[...] = source
    for value in arrays.values():
        value.flush()


def run(manifest_path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest_sha = _sha256(manifest_file)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("execution_successor_id") != SUCCESSOR_ID:
        raise _v9.DiversityRunnerError("S2 execution successor binding is absent")
    if manifest.get("segment_rollouts") != SEGMENT_ROLLOUTS:
        raise _v9.DiversityRunnerError("S2 segment size drifted")
    expected_runner_hash = manifest.get("source_hashes", {}).get(
        "src/agc_runtime_assurance/circa_ress_v9_sci_s3_runner.py"
    )
    if not isinstance(expected_runner_hash, str) or _sha256(Path(__file__)) != (
        expected_runner_hash
    ):
        raise _v9.DiversityRunnerError("S2 runner source lock failed")
    with _v9._configured_core():
        _design, _operationalization, _registry, output, state_path = (
            _v9._validate_v9_manifest(manifest, repo)
        )
    output.mkdir(parents=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    scratch = state_path.parent / f".{SUCCESSOR_ID.lower()}-scratch"
    started_at_ns = time.time_ns()
    attempt = {
        "schema_version": "1.0",
        "route_id": _v9.ROUTE_ID,
        "execution_successor_id": SUCCESSOR_ID,
        "status": "SCIENTIFIC_ATTEMPT_STARTED",
        "manifest_sha256": manifest_sha,
        "started_at_ns": started_at_ns,
        "scientific_attempts_consumed": 1,
        "retry_allowed": False,
    }
    _atomic_json(output / "ATTEMPT_STARTED.json", attempt)
    _atomic_json(
        state_path,
        {
            **attempt,
            "status": "RUNNING_SCIENTIFIC_EXACTLY_ONCE",
            "completed_rollouts": 0,
            "total_rollouts": _v9.ROLLOUTS,
        },
    )
    arrays = _allocate_memmaps(scratch / "arrays")
    started = time.perf_counter()
    completed = 0
    try:
        for start in range(0, _v9.ROLLOUTS, SEGMENT_ROLLOUTS):
            stop = min(start + SEGMENT_ROLLOUTS, _v9.ROLLOUTS)
            segment_path = scratch / "segments" / f"{start:05d}_{stop:05d}.npz"
            command = [
                sys.executable,
                "-m",
                "agc_runtime_assurance.circa_ress_v9_sci_s3_runner",
                "--worker",
                "--manifest",
                str(manifest_file),
                "--repo-root",
                str(repo),
                "--start",
                str(start),
                "--stop",
                str(stop),
                "--segment",
                str(segment_path),
            ]
            subprocess.run(command, cwd=repo, check=True)
            _merge_segment(arrays, segment_path, start, stop)
            completed = stop
            _atomic_json(
                state_path,
                {
                    **attempt,
                    "status": "RUNNING_SCIENTIFIC_EXACTLY_ONCE",
                    "completed_rollouts": completed,
                    "total_rollouts": _v9.ROLLOUTS,
                    "elapsed_seconds": time.perf_counter() - started,
                    "bounded_worker_rollouts": SEGMENT_ROLLOUTS,
                },
            )
        with _v9._configured_core():
            _v9._core._validate_scientific_arrays(arrays)
            runtime_seconds = time.perf_counter() - started
            summary = _v9._core._summary(arrays, manifest_sha, runtime_seconds)
        summary["result_id"] = "circa-ress-v9-feasible-initial-domain-r1-scientific-result-s2"
        summary["independent_units"] = _v9.INDEPENDENT_UNITS
        summary["execution_successor_id"] = SUCCESSOR_ID
        summary_bytes = (
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        trace_tmp = output / "trace_arrays.npz.tmp"
        with trace_tmp.open("wb") as handle:
            np.savez_compressed(
                handle, **{name: arrays[name] for name in sorted(arrays)}
            )
        result_tmp = output / "result.json.tmp"
        result_tmp.write_bytes(summary_bytes)
        total_bytes = (
            trace_tmp.stat().st_size
            + result_tmp.stat().st_size
            + (output / "ATTEMPT_STARTED.json").stat().st_size
        )
        if total_bytes > _v9.OUTPUT_CAPACITY_BYTES:
            raise _v9.DiversityRunnerError(
                f"lossless output capacity exceeded: {total_bytes}"
            )
        os.replace(trace_tmp, output / "trace_arrays.npz")
        os.replace(result_tmp, output / "result.json")
        terminal = {
            **attempt,
            "status": "PASS_SCIENTIFIC_RUN_COMPLETE_PENDING_FROZEN_ANALYSIS",
            "completed_rollouts": _v9.ROLLOUTS,
            "total_rollouts": _v9.ROLLOUTS,
            "runtime_seconds": runtime_seconds,
            "output_bytes": total_bytes,
            "output_capacity_bytes": _v9.OUTPUT_CAPACITY_BYTES,
            "trace_arrays_sha256": _sha256(output / "trace_arrays.npz"),
            "result_sha256": _sha256(output / "result.json"),
            "bounded_worker_rollouts": SEGMENT_ROLLOUTS,
        }
        _atomic_json(state_path, terminal)
        return terminal
    except BaseException as error:
        _atomic_json(
            state_path,
            {
                **attempt,
                "status": "TERMINAL_FAIL_SCIENTIFIC_NO_RETRY",
                "completed_rollouts": completed,
                "total_rollouts": _v9.ROLLOUTS,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        for value in arrays.values():
            value.flush()
        del arrays
        gc.collect()
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if str(state.get("status", "")).startswith("PASS_"):
                shutil.rmtree(scratch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--segment")
    arguments = parser.parse_args()
    if arguments.worker:
        if (
            arguments.start is None
            or arguments.stop is None
            or arguments.segment is None
        ):
            raise SystemExit("worker arguments are incomplete")
        _worker(
            Path(arguments.manifest).resolve(),
            Path(arguments.repo_root).resolve(),
            arguments.start,
            arguments.stop,
            Path(arguments.segment).resolve(),
        )
    else:
        print(
            json.dumps(
                run(arguments.manifest, arguments.repo_root),
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
