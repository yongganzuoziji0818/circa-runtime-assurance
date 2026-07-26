"""Non-scientific supervisor-worker boundary preflight for V9 S3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from agc_runtime_assurance.circa_ress_v9_sci_s3_runner import (
    SUCCESSOR_ID,
    validate_worker_claim,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exclusive_json(path: Path, payload: dict) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    try:
        os.write(descriptor, data)
    finally:
        os.close(descriptor)


def run(root: Path, receipt: Path) -> dict:
    root = root.resolve()
    fixture_root = (
        root / "experiments/non_scientific/circa_ress_v9_sci_s3_boundary_r1"
    )
    if fixture_root.exists() or receipt.exists():
        raise RuntimeError("S3 boundary fixture target is not absent")
    output = fixture_root / "fixture_claim_output"
    output.mkdir(parents=True)
    manifest_path = fixture_root / "fixture_manifest.json"
    manifest = {
        "schema_version": "1.0",
        "execution_successor_id": SUCCESSOR_ID,
        "non_scientific_fixture": True,
        "output_path": str(output.relative_to(root).as_posix()),
        "scientific_seed_access": False,
        "scientific_output": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    claim = {
        "schema_version": "1.0",
        "execution_successor_id": SUCCESSOR_ID,
        "status": "NONSCIENTIFIC_FIXTURE_CLAIM",
        "manifest_sha256": sha256(manifest_path),
        "scientific_attempts_consumed": 0,
        "retry_allowed": False,
    }
    exclusive_json(output / "ATTEMPT_STARTED.json", claim)
    validated = validate_worker_claim(manifest_path, root)
    if validated != manifest:
        raise RuntimeError("S3 fixture manifest identity changed")
    memory_command = [
        sys.executable,
        "-B",
        str(root / "scripts/preflight_circa_ress_v9_sci_s2_memory_worker_r2.py"),
        "--repo-root",
        str(root),
    ]
    completed = subprocess.run(
        memory_command, cwd=root, check=True, capture_output=True, text=True
    )
    memory_result = json.loads(completed.stdout)
    if memory_result.get("status") != "PASS_NONSCIENTIFIC_S2_MEMORY_WORKER_R2":
        raise RuntimeError("bounded fixture worker did not pass")
    payload = {
        "schema_version": "1.0",
        "receipt_id": "circa-ress-v9-sci-s3-boundary-r1-20260726",
        "execution_successor_id": SUCCESSOR_ID,
        "status": "PASS_NONSCIENTIFIC_SUPERVISOR_WORKER_BOUNDARY",
        "fixture_manifest_sha256": sha256(manifest_path),
        "fixture_claim_sha256": sha256(output / "ATTEMPT_STARTED.json"),
        "claim_identity_validated_after_output_creation": True,
        "target_absence_revalidated_by_worker": False,
        "bounded_worker": memory_result,
        "scientific_seed_accessed": False,
        "scientific_output_created": False,
        "scientific_runner_invoked": False,
        "scientific_attempts_consumed": 0,
    }
    exclusive_json(receipt.resolve(), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.repo_root, arguments.receipt), sort_keys=True))


if __name__ == "__main__":
    main()
