"""Exactly-once in-memory schema-capacity audit for CIRCA-RESS-V8-SIM-DIVERSITY-R1.

The deterministic SHAKE256 fixture is audit-only entropy, never a scientific
seed.  The lossless NPZ exists only in memory and is discarded before exit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


ROUTE_ID = "CIRCA-RESS-V8-SIM-DIVERSITY-R1"
AUDIT_DOMAIN = b"circa-ress-v8-sim-diversity-r1-schema-capacity-audit-v1"
CAPACITY_BYTES = 268_435_456
RECEIPT_NAME = (
    "circa_ress_v8_sim_diversity_r1_schema_capacity_256mib_receipt_20260724.json"
)
LOCK_NAME = "circa_ress_v8_sim_diversity_r1_schema_capacity_256mib_attempt.lock"
AUTH_RECORD_NAME = (
    "P4_CIRCA_RESS_V8_SIM_DIVERSITY_R1_"
    "SCHEMA_CAPACITY_AUDIT_AUTHORIZATION_20260724.json"
)
EXPECTED_HASHES = {
    "design_manifest": "f84325ab6d901d2f03f37f2e6b34ebba570d513c8f8b8f29a4581bf9d363aaa9",
    "contract_confirmation": "169f82064bce089620184ce8d24b4b11c57b3c5f9ce4dec9faa9844a3868ad44",
    "implementation_schema_receipt": "ed6f0b569b782d35f07a52031625d7128fb0fd62d42fd1206b2996afd8e3b042",
    "schema_source": "e530232047406d165429c7a034d44d5a20572a6bc99982455e7a1ef9ed2680b1",
    "execution_queue": "5ef8a0546796a250799e8e1a0efa6d455d285a96a42e8059ac3652c8b11ce030",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entropy_bytes(name: str, byte_count: int) -> bytes:
    return hashlib.shake_256(
        AUDIT_DOMAIN + b":" + name.encode("utf-8")
    ).digest(byte_count)


def _fill_entropy(array: np.ndarray, name: str) -> None:
    flat = array.view(np.uint8).reshape(-1)
    chunk_bytes = 4 * 1024 * 1024
    for offset in range(0, flat.size, chunk_bytes):
        stop = min(flat.size, offset + chunk_bytes)
        block = hashlib.shake_256(
            AUDIT_DOMAIN
            + b":"
            + name.encode("utf-8")
            + b":"
            + offset.to_bytes(8, "little")
        ).digest(stop - offset)
        flat[offset:stop] = np.frombuffer(block, dtype=np.uint8)


def build_high_entropy_arrays(
    rollouts: int,
    horizon: int,
) -> dict[str, np.ndarray]:
    """Build a valid audit-only fixture without RNG or seed material."""
    from agc_runtime_assurance.circa_ress_v8_sim_diversity_schema import (
        CANDIDATE_CODEBOOK,
        DRIVER_CODEBOOK,
        FAMILY_CODEBOOK,
        METHOD_CODEBOOK,
        SPLIT_CODEBOOK,
        array_schema,
        validate_schema_arrays,
    )

    arrays: dict[str, np.ndarray] = {}
    for name, (dtype, shape) in array_schema(rollouts, horizon).items():
        value = np.empty(shape, dtype=dtype)
        if dtype == np.dtype("<f8"):
            _fill_entropy(value, name)
            words = value.view(np.uint64)
            words &= np.uint64((1 << 52) - 1)
            words |= np.uint64(0x3FE0000000000000)
        elif dtype == np.dtype("?"):
            raw = np.frombuffer(_entropy_bytes(name, value.size), dtype=np.uint8)
            value.reshape(-1)[:] = (raw & 1).astype(bool)
        else:
            _fill_entropy(value, name)
        arrays[name] = value

    arrays["future_seed_sentinel"].fill(-1)
    for name in (
        "nominal_tube_feasible",
        "backup_tube_feasible",
        "terminal_reachability",
    ):
        source = arrays[name].view(np.uint8)
        arrays[name][...] = (source % 3).astype(np.int8) - 1

    for name, size in (
        ("family_index", len(FAMILY_CODEBOOK)),
        ("candidate_index", len(CANDIDATE_CODEBOOK)),
        ("split_index", len(SPLIT_CODEBOOK)),
        ("driver_index", len(DRIVER_CODEBOOK)),
        ("method_index", len(METHOD_CODEBOOK)),
    ):
        arrays[name] %= size

    arrays["refusal_code"] %= 8
    arrays["fault_code"] %= 8
    arrays["completed_steps"] %= horizon + 1
    arrays["completed_step_mask"][...] = (
        np.arange(horizon, dtype=np.uint16)[None, :]
        < arrays["completed_steps"].astype(np.uint16)[:, None]
    )
    arrays["typed_refusal"].fill(True)
    arrays["design_speed_limit_mps"] += 0.5
    validate_schema_arrays(arrays, rollouts, horizon)
    return arrays


def archive_and_verify(
    arrays: dict[str, np.ndarray],
) -> tuple[bytes, str, str, int]:
    """Return lossless in-memory NPZ bytes and aggregate verification evidence."""
    source_digest = hashlib.sha256()
    for name, value in arrays.items():
        source_digest.update(name.encode("utf-8"))
        source_digest.update(value.dtype.str.encode("ascii"))
        source_digest.update(json.dumps(value.shape).encode("ascii"))
        source_digest.update(value.tobytes(order="C"))

    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    archive = stream.getvalue()
    archive_hash = hashlib.sha256(archive).hexdigest()

    verified = 0
    restored_digest = hashlib.sha256()
    with np.load(BytesIO(archive), allow_pickle=False) as restored:
        if set(restored.files) != set(arrays):
            raise RuntimeError("lossless archive member set drifted")
        for name, expected in arrays.items():
            actual = restored[name]
            if actual.dtype != expected.dtype or actual.shape != expected.shape:
                raise RuntimeError(f"lossless round-trip schema mismatch: {name}")
            if actual.tobytes(order="C") != expected.tobytes(order="C"):
                raise RuntimeError(f"lossless round-trip payload mismatch: {name}")
            restored_digest.update(name.encode("utf-8"))
            restored_digest.update(actual.dtype.str.encode("ascii"))
            restored_digest.update(json.dumps(actual.shape).encode("ascii"))
            restored_digest.update(actual.tobytes(order="C"))
            verified += 1
    if restored_digest.hexdigest() != source_digest.hexdigest():
        raise RuntimeError("aggregate lossless round-trip digest mismatch")
    return archive, archive_hash, source_digest.hexdigest(), verified


def _stable_receipt_bytes(result: dict[str, Any], archive_bytes: int) -> bytes:
    previous = None
    for _ in range(16):
        body = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
        summary_bytes = len(body)
        total_bytes = archive_bytes + summary_bytes
        capacity_pass = total_bytes <= CAPACITY_BYTES
        result.update(
            {
                "receipt_json_bytes": summary_bytes,
                "total_contract_bytes": total_bytes,
                "headroom_bytes": CAPACITY_BYTES - total_bytes,
                "capacity_pass": capacity_pass,
                "status": (
                    "PASS_HIGH_ENTROPY_LOSSLESS_SCHEMA_CAPACITY"
                    if capacity_pass
                    else "TERMINAL_FAIL_SCHEMA_CAPACITY_NO_SCIENTIFIC_RUN"
                ),
                "terminal": not capacity_pass,
                "next_gate": (
                    "CAPACITY_ONLY_PASS_AWAIT_SEPARATE_NONSCIENTIFIC_PREFLIGHT_AUTHORIZATION"
                    if capacity_pass
                    else "PRESERVE_TERMINAL_FAILURE_NO_RETRY"
                ),
            }
        )
        state = (summary_bytes, total_bytes, capacity_pass)
        if state == previous:
            return (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            )
        previous = state
    raise RuntimeError("receipt length did not converge")


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)


def _preflight_hashes(repo_root: Path) -> tuple[dict[str, str], Path]:
    workspace_root = repo_root.parents[3]
    paths = {
        "design_manifest": repo_root
        / "experiments/manifests/circa_ress_v8_sim_diversity_r1_DESIGN_ONLY.json",
        "contract_confirmation": workspace_root
        / "0.总控/governance/P4_CIRCA_RESS_V8_SIM_DIVERSITY_R1_CONTRACT_CONFIRMATION_20260724.json",
        "implementation_schema_receipt": repo_root
        / "experiments/manifests/circa_ress_v8_sim_diversity_r1_implementation_schema_receipt_20260724.json",
        "schema_source": repo_root
        / "src/agc_runtime_assurance/circa_ress_v8_sim_diversity_schema.py",
        "execution_queue": workspace_root
        / "0.总控/governance/execution_queue_current.json",
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    if actual != EXPECTED_HASHES:
        raise RuntimeError(
            "frozen hash preflight failed: "
            + json.dumps({"expected": EXPECTED_HASHES, "actual": actual}, sort_keys=True)
        )

    authorization_path = (
        workspace_root / "0.总控/governance" / AUTH_RECORD_NAME
    )
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    action = authorization["authorized_action"]
    prohibitions = authorization["strict_prohibitions"]
    if (
        authorization["route_id"] != ROUTE_ID
        or action["attempts_authorized"] != 1
        or action["retry_allowed"]
        or action["frozen_capacity_bytes"] != CAPACITY_BYTES
        or any(prohibitions.values())
    ):
        raise RuntimeError("capacity-audit authorization record failed closed")
    return actual, authorization_path


def run_exactly_once(repo_root: Path) -> dict[str, Any]:
    from agc_runtime_assurance.circa_ress_v8_sim_diversity_schema import (
        DEFAULT_HORIZON,
        DEFAULT_ROLLOUTS,
        SCHEMA_VERSION,
        array_schema,
    )

    root = repo_root.resolve()
    manifest_dir = root / "experiments" / "manifests"
    receipt_path = manifest_dir / RECEIPT_NAME
    lock_path = manifest_dir / LOCK_NAME
    if receipt_path.exists() or lock_path.exists():
        raise RuntimeError("exactly-once capacity audit already consumed")

    frozen_hashes, authorization_path = _preflight_hashes(root)
    lock_payload = {
        "route_id": ROUTE_ID,
        "attempt": 1,
        "attempts_authorized": 1,
        "retry_allowed": False,
        "attempt_consumed_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "non-scientific high-entropy lossless schema capacity audit",
    }
    _write_exclusive(
        lock_path,
        (json.dumps(lock_payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )

    try:
        schema = array_schema(DEFAULT_ROLLOUTS, DEFAULT_HORIZON)
        raw_bytes = sum(
            dtype.itemsize * math.prod(shape) for dtype, shape in schema.values()
        )
        arrays = build_high_entropy_arrays(DEFAULT_ROLLOUTS, DEFAULT_HORIZON)
        archive, archive_hash, fixture_hash, verified = archive_and_verify(arrays)
        archive_bytes = len(archive)
        del arrays

        result: dict[str, Any] = {
            "schema_version": "1.0",
            "receipt_id": "circa-ress-v8-sim-diversity-r1-schema-capacity-256mib-20260724",
            "route_id": ROUTE_ID,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "audit_type": "HIGH_ENTROPY_LOSSLESS_SCHEMA_CAPACITY",
            "audit_attempts": 1,
            "attempts_authorized": 1,
            "retry_allowed": False,
            "schema_contract_version": SCHEMA_VERSION,
            "rollouts": DEFAULT_ROLLOUTS,
            "horizon_steps": DEFAULT_HORIZON,
            "field_count": len(schema),
            "raw_array_bytes": raw_bytes,
            "in_memory_npz_bytes": archive_bytes,
            "frozen_cap_bytes": CAPACITY_BYTES,
            "lossless_round_trip": True,
            "verified_member_count": verified,
            "archive_sha256": archive_hash,
            "audit_fixture_aggregate_sha256": fixture_hash,
            "entropy_generator": "SHAKE256 domain-separated fixture; audit-only",
            "entropy_generator_is_scientific_seed": False,
            "future_seed_sentinel_preserved": -1,
            "in_memory_archive_discarded": True,
            "archive_written_to_disk": False,
            "authorization_record_sha256": sha256_file(authorization_path),
            "frozen_input_hashes": frozen_hashes,
            "auditor_source_sha256": sha256_file(Path(__file__)),
            "zero_action_record": {
                "gazebo_or_simulator_invoked": False,
                "scientific_seed_generated_or_materialized": False,
                "scientific_output_directory_created": False,
                "scientific_output_created": False,
                "remote_deployment_or_execution": False,
                "scientific_runner_invoked": False,
                "scientific_experiment_executed": False,
                "queue_modified": False,
            },
            "scientific_attempts_authorized": 0,
            "scientific_attempts_consumed": 0,
        }
        payload = _stable_receipt_bytes(result, archive_bytes)
        del archive
        _write_exclusive(receipt_path, payload)
        return result
    except Exception as error:
        failure = {
            "schema_version": "1.0",
            "receipt_id": "circa-ress-v8-sim-diversity-r1-schema-capacity-256mib-20260724",
            "route_id": ROUTE_ID,
            "status": "TERMINAL_FAIL_SCHEMA_AUDIT_IMPLEMENTATION_NO_SCIENTIFIC_RUN",
            "terminal": True,
            "audit_attempts": 1,
            "attempts_authorized": 1,
            "retry_allowed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "scientific_seed_generated_or_materialized": False,
            "scientific_output_created": False,
            "gazebo_or_simulator_invoked": False,
            "remote_deployment_or_execution": False,
            "scientific_experiment_executed": False,
            "scientific_attempts_consumed": 0,
            "next_gate": "PRESERVE_TERMINAL_FAILURE_NO_RETRY",
        }
        if not receipt_path.exists():
            _write_exclusive(
                receipt_path,
                (json.dumps(failure, indent=2, sort_keys=True) + "\n").encode(
                    "utf-8"
                ),
            )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_exactly_once(args.repo_root)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["capacity_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
