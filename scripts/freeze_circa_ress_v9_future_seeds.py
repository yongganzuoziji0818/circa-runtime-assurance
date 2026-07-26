"""Exactly-once future-seed freeze for the confirmed CIRCA-RESS-V9 route."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets

from agc_runtime_assurance.circa_ress_v9_runner import (
    INDEPENDENT_UNITS,
    ROLLOUTS,
    compile_schedule,
)


ROUTE_ID = "CIRCA-RESS-V9-FEASIBLE-INITIAL-DOMAIN-R1"
RECEIPT = "circa_ress_v9_FUTURE_SEED_FREEZE_20260725.json"
LOCK = "circa_ress_v9_FUTURE_SEED_FREEZE_20260725.lock"
QUEUE_SHA256 = "d0ac237695556a21c1e6f28ba42fbdf493394c76a43930144d65958c7941d965"
SOURCE_LOCK_SHA256 = "4a42ba73f4ea53be7748d521f0b4d92283dd718cc176460231f82ca86ab1f70f"
PREFLIGHT_SHA256 = "c428324c55446f19d1eb0e6c1587cb4c7fc87a5abd9c51ebb8ba71c9e09de583"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exclusive(path: Path, payload: dict) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    try:
        os.write(descriptor, data)
    finally:
        os.close(descriptor)


def run(repo_root: Path, queue_path: Path) -> dict:
    root = repo_root.resolve()
    manifests = root / "experiments/manifests"
    receipt = manifests / RECEIPT
    lock = manifests / LOCK
    if receipt.exists() or lock.exists():
        raise RuntimeError("V9 future-seed freeze already consumed")
    if sha256(queue_path.resolve()) != QUEUE_SHA256:
        raise RuntimeError("authoritative queue changed before seed freeze")
    if sha256(manifests / "circa_ress_v9_SOURCE_LOCK_R2.json") != SOURCE_LOCK_SHA256:
        raise RuntimeError("source lock changed before seed freeze")
    if (
        sha256(manifests / "circa_ress_v9_REMOTE_PREFLIGHT_R3_PASS_20260725.json")
        != PREFLIGHT_SHA256
    ):
        raise RuntimeError("remote preflight evidence changed before seed freeze")
    _exclusive(
        lock,
        {
            "route_id": ROUTE_ID,
            "status": "FUTURE_SEED_FREEZE_ATTEMPT_CONSUMED",
            "attempt": 1,
            "retry_allowed": False,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    master_seed = secrets.randbits(63)
    schedule_seed = secrets.randbits(63)
    manifest = {"master_seed": master_seed, "schedule_seed": schedule_seed}
    schedule = compile_schedule(manifest)
    units = {}
    for row in schedule:
        key = (
            int(row.pair_id),
            int(row.family_index),
            int(row.candidate_index),
            int(row.split_index),
            int(row.seed_index),
            int(row.future_seed),
        )
        units[key] = None
    ordered = [
        {
            "pair_id": pair_id,
            "family_index": family,
            "candidate_index": candidate,
            "split_index": split,
            "seed_index": seed_index,
            "future_seed": future_seed,
        }
        for pair_id, family, candidate, split, seed_index, future_seed in sorted(units)
    ]
    if len(schedule) != ROLLOUTS or len(ordered) != INDEPENDENT_UNITS:
        raise RuntimeError("frozen V9 schedule dimensions drifted")
    vector_body = json.dumps(
        ordered, sort_keys=True, separators=(",", ":")
    ).encode()
    payload = {
        "schema_version": "1.0",
        "receipt_id": "circa-ress-v9-future-seed-freeze-20260725",
        "route_id": ROUTE_ID,
        "status": "FROZEN_FUTURE_SEEDS",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "seed_namespace": "circa-ress-v9-feasible-initial-domain-r1",
        "master_seed": master_seed,
        "schedule_seed": schedule_seed,
        "ordered_seed_vector": ordered,
        "ordered_seed_vector_sha256": hashlib.sha256(vector_body).hexdigest(),
        "independent_unit_count": INDEPENDENT_UNITS,
        "schedule_rollout_count": ROLLOUTS,
        "validation_unit_count": 320,
        "evaluation_unit_count": 768,
        "queue_sha256": QUEUE_SHA256,
        "source_lock_sha256": SOURCE_LOCK_SHA256,
        "remote_preflight_sha256": PREFLIGHT_SHA256,
        "seed_replacement_allowed": False,
        "seed_top_up_allowed": False,
        "sample_top_up_allowed": False,
        "retry_allowed": False,
        "scientific_output_created": False,
        "scientific_runner_invoked": False,
        "scientific_attempts_consumed": 0,
    }
    _exclusive(receipt, payload)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--queue", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.repo_root, args.queue)
    print(
        json.dumps(
            {
                "status": result["status"],
                "independent_unit_count": result["independent_unit_count"],
                "schedule_rollout_count": result["schedule_rollout_count"],
                "ordered_seed_vector_sha256": result[
                    "ordered_seed_vector_sha256"
                ],
            },
            sort_keys=True,
        )
    )
