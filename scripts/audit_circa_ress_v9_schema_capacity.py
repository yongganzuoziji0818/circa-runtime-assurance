"""Exactly-once in-memory schema-capacity audit for CIRCA-RESS-V9.

SHAKE256 fixture bytes are audit-only entropy, never scientific seed material.
The lossless archive exists only in memory and is discarded before exit.
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


ROUTE_ID = "CIRCA-RESS-V9-FEASIBLE-INITIAL-DOMAIN-R1"
AUDIT_DOMAIN = b"circa-ress-v9-feasible-initial-domain-r1-schema-capacity-v1"
CAPACITY_BYTES = 402_653_184
RECEIPT_NAME = "circa_ress_v9_schema_capacity_384mib_receipt_20260725.json"
LOCK_NAME = "circa_ress_v9_schema_capacity_384mib_attempt.lock"
AUTH_NAME = "P4_CIRCA_RESS_V9_SCHEMA_CAPACITY_AUDIT_DERIVED_AUTHORIZATION_20260725.json"
EXPECTED_HASHES = {
    "design_manifest": "135ad014ffa031b2349757e25a0100189ac16a2664db0c951b367fcffa85fd2b",
    "contract_confirmation": "191801e10343c9044fd83e326f34962825cdfce913979651d58d5e3e6ed95c49",
    "implementation_schema_build": "58e90c091243993d6de4b4156cd5690f0f8eb4e7d4c5c7c4cd8a9d12ab22722e",
    "schema_source": "397ec0b8d7d6e0eb4810b852810eaaa7822136cea89a701b5115e12865498c1c",
    "execution_queue": "e3c35e44c88c18e35df6a12db9551a614b28a2d8964514a88a8470f9fc87496a",
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
    from agc_runtime_assurance.circa_ress_v9_schema import (
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
    source_digest = hashlib.sha256()
    for name, value in arrays.items():
        source_digest.update(name.encode("utf-8"))
        source_digest.update(value.dtype.str.encode("ascii"))
        source_digest.update(json.dumps(value.shape).encode("ascii"))
        source_digest.update(value.tobytes(order="C"))
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    archive = stream.getvalue()
    verified = 0
    restored_digest = hashlib.sha256()
    with np.load(BytesIO(archive), allow_pickle=False) as restored:
        if set(restored.files) != set(arrays):
            raise RuntimeError("lossless archive member set drifted")
        for name, expected in arrays.items():
            actual = restored[name]
            if (
                actual.dtype != expected.dtype
                or actual.shape != expected.shape
                or actual.tobytes(order="C") != expected.tobytes(order="C")
            ):
                raise RuntimeError(f"lossless round-trip mismatch: {name}")
            restored_digest.update(name.encode("utf-8"))
            restored_digest.update(actual.dtype.str.encode("ascii"))
            restored_digest.update(json.dumps(actual.shape).encode("ascii"))
            restored_digest.update(actual.tobytes(order="C"))
            verified += 1
    if restored_digest.hexdigest() != source_digest.hexdigest():
        raise RuntimeError("aggregate lossless round-trip digest mismatch")
    return (
        archive,
        hashlib.sha256(archive).hexdigest(),
        source_digest.hexdigest(),
        verified,
    )


def _stable_receipt_bytes(result: dict[str, Any], archive_bytes: int) -> bytes:
    previous = None
    for _ in range(16):
        body = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
        total = archive_bytes + len(body)
        passed = total <= CAPACITY_BYTES
        result.update(
            {
                "receipt_json_bytes": len(body),
                "total_contract_bytes": total,
                "headroom_bytes": CAPACITY_BYTES - total,
                "capacity_pass": passed,
                "status": (
                    "PASS_HIGH_ENTROPY_LOSSLESS_SCHEMA_CAPACITY"
                    if passed
                    else "TERMINAL_FAIL_SCHEMA_CAPACITY_NO_SCIENTIFIC_RUN"
                ),
                "terminal": not passed,
                "next_gate": (
                    "CAPACITY_PASS_AWAIT_VERSIONED_NONSCIENTIFIC_PREFLIGHT"
                    if passed
                    else "PRESERVE_TERMINAL_FAILURE_NO_RETRY"
                ),
            }
        )
        state = (len(body), total, passed)
        if state == previous:
            return (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
        previous = state
    raise RuntimeError("receipt length did not converge")


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)


def _preflight(repo_root: Path) -> tuple[dict[str, str], Path]:
    workspace = repo_root.parents[3]
    design_dir = (
        workspace
        / "4.运行时安全保障/.staging/p4_v9_design"
        / "circa_ress_v9_feasible_initial_domain_r1_20260725"
    )
    paths = {
        "design_manifest": repo_root
        / "experiments/manifests/circa_ress_v9_feasible_initial_domain_r1_DESIGN_FROZEN.json",
        "contract_confirmation": design_dir
        / "CIRCA_RESS_V9_CONTRACT_CONFIRMATION_20260725.json",
        "implementation_schema_build": repo_root
        / "experiments/manifests/circa_ress_v9_feasible_initial_domain_r1_IMPLEMENTATION_SCHEMA_BUILD_20260725.json",
        "schema_source": repo_root
        / "src/agc_runtime_assurance/circa_ress_v9_schema.py",
        "execution_queue": workspace / "0.总控/governance/execution_queue_current.json",
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    if actual != EXPECTED_HASHES:
        raise RuntimeError(
            "frozen hash preflight failed: "
            + json.dumps({"expected": EXPECTED_HASHES, "actual": actual})
        )
    authorization = workspace / "0.总控/governance" / AUTH_NAME
    body = json.loads(authorization.read_text(encoding="utf-8"))
    action = body["authorized_action"]
    if (
        body["route_id"] != ROUTE_ID
        or action["attempts_authorized"] != 1
        or action["retry_allowed"]
        or action["frozen_capacity_bytes"] != CAPACITY_BYTES
        or any(body["strict_prohibitions"].values())
    ):
        raise RuntimeError("capacity authorization failed closed")
    return actual, authorization


def run_exactly_once(repo_root: Path) -> dict[str, Any]:
    from agc_runtime_assurance.circa_ress_v9_schema import (
        DEFAULT_HORIZON,
        DEFAULT_ROLLOUTS,
        SCHEMA_VERSION,
        array_schema,
    )

    root = repo_root.resolve()
    manifests = root / "experiments/manifests"
    receipt = manifests / RECEIPT_NAME
    lock = manifests / LOCK_NAME
    if receipt.exists() or lock.exists():
        raise RuntimeError("exactly-once capacity audit already consumed")
    frozen_hashes, authorization = _preflight(root)
    _write_exclusive(
        lock,
        (
            json.dumps(
                {
                    "route_id": ROUTE_ID,
                    "attempt": 1,
                    "attempts_authorized": 1,
                    "retry_allowed": False,
                    "attempt_consumed_at": datetime.now(timezone.utc).isoformat(),
                    "purpose": "non-scientific high-entropy lossless schema capacity audit",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )
    try:
        schema = array_schema(DEFAULT_ROLLOUTS, DEFAULT_HORIZON)
        raw_bytes = sum(
            dtype.itemsize * math.prod(shape) for dtype, shape in schema.values()
        )
        arrays = build_high_entropy_arrays(DEFAULT_ROLLOUTS, DEFAULT_HORIZON)
        archive, archive_hash, fixture_hash, verified = archive_and_verify(arrays)
        del arrays
        result: dict[str, Any] = {
            "schema_version": "1.0",
            "receipt_id": "circa-ress-v9-schema-capacity-384mib-20260725",
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
            "in_memory_npz_bytes": len(archive),
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
            "authorization_record_sha256": sha256_file(authorization),
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
        payload = _stable_receipt_bytes(result, len(archive))
        del archive
        _write_exclusive(receipt, payload)
        return result
    except Exception as error:
        failure = {
            "schema_version": "1.0",
            "receipt_id": "circa-ress-v9-schema-capacity-384mib-20260725",
            "route_id": ROUTE_ID,
            "status": "TERMINAL_FAIL_SCHEMA_AUDIT_IMPLEMENTATION_NO_SCIENTIFIC_RUN",
            "terminal": True,
            "audit_attempts": 1,
            "attempts_authorized": 1,
            "retry_allowed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "scientific_attempts_consumed": 0,
            "next_gate": "PRESERVE_TERMINAL_FAILURE_NO_RETRY",
        }
        if not receipt.exists():
            _write_exclusive(
                receipt,
                (json.dumps(failure, indent=2, sort_keys=True) + "\n").encode(),
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
