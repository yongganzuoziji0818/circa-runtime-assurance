"""Fail-closed audit of the frozen AoI-CBF original-training receipt."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


class TrainingReceiptError(ValueError):
    pass


@dataclass(frozen=True)
class TrainingReceiptAudit:
    receipt_hash: str
    manifest_hash: str
    completed_steps: int
    checkpoint_count: int
    evidence_bytes: int
    wall_clock_seconds: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingReceiptError(f"{label} must be an object")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise TrainingReceiptError(f"{label} must be a SHA256 digest")
    return value.lower()


def verify_training_receipt(
    manifest_path: str | Path,
    receipt_path: str | Path,
) -> TrainingReceiptAudit:
    """Verify a local receipt produced from read-only remote evidence checks."""
    manifest_file = Path(manifest_path).resolve()
    receipt_file = Path(receipt_path).resolve()
    manifest = _object(json.loads(manifest_file.read_text(encoding="utf-8")), "manifest")
    receipt = _object(json.loads(receipt_file.read_text(encoding="utf-8")), "receipt")
    manifest_hash = _sha256(manifest_file)

    if receipt.get("status") != "completed":
        raise TrainingReceiptError("receipt status must be completed")
    if receipt.get("claim_generation_allowed") is not False:
        raise TrainingReceiptError("training receipt must forbid claim generation")
    if receipt.get("sealed_data_used") is not False or receipt.get("formal_or_g2") is not False:
        raise TrainingReceiptError("training receipt crosses sealed/formal boundary")

    bound_manifest = _object(receipt.get("manifest"), "receipt.manifest")
    if bound_manifest.get("manifest_id") != manifest.get("manifest_id"):
        raise TrainingReceiptError("manifest id mismatch")
    if _digest(bound_manifest.get("sha256"), "receipt.manifest.sha256") != manifest_hash:
        raise TrainingReceiptError("manifest hash mismatch")

    source_manifest = _object(manifest.get("source"), "manifest.source")
    source = _object(receipt.get("source"), "receipt.source")
    if source.get("commit") != source_manifest.get("commit"):
        raise TrainingReceiptError("source commit mismatch")
    if source.get("clean_before") is not True or source.get("clean_after") is not True:
        raise TrainingReceiptError("source checkout was not clean before and after")

    expected = _object(manifest.get("execution"), "manifest.execution")
    execution = _object(receipt.get("execution"), "receipt.execution")
    if execution.get("run_id") != expected.get("run_id"):
        raise TrainingReceiptError("run id mismatch")
    if execution.get("command") != expected.get("command"):
        raise TrainingReceiptError("training command mismatch")
    if execution.get("exit_code") != 0:
        raise TrainingReceiptError("training exit code was not zero")
    expected_steps = int(expected.get("steps"))
    if execution.get("requested_steps") != expected_steps:
        raise TrainingReceiptError("requested step count mismatch")
    if execution.get("completed_steps") != expected_steps:
        raise TrainingReceiptError("training did not reach the exact endpoint")

    interval = int(expected.get("expected_checkpoint_interval"))
    expected_checkpoint_steps = list(range(interval, expected_steps + 1, interval))
    if execution.get("checkpoint_steps") != expected_checkpoint_steps:
        raise TrainingReceiptError("checkpoint step sequence is incomplete or unexpected")
    if execution.get("checkpoint_count") != len(expected_checkpoint_steps):
        raise TrainingReceiptError("checkpoint count mismatch")
    _digest(execution.get("checkpoint_tree_sha256"), "checkpoint tree hash")
    final_hashes = _object(execution.get("final_checkpoint_sha256"), "final checkpoint hashes")
    if set(final_hashes) != {"actor.pkl", "cbf.pkl", "predictor.pkl"}:
        raise TrainingReceiptError("final checkpoint hash set is incomplete")
    for name, value in final_hashes.items():
        _digest(value, f"final checkpoint {name}")
    _digest(execution.get("settings_sha256"), "settings hash")
    _digest(execution.get("raw_log_sha256"), "raw log hash")

    budget_contract = _object(manifest.get("budget"), "manifest.budget")
    budget = _object(receipt.get("budget"), "receipt.budget")
    wall_clock_seconds = float(budget.get("wall_clock_seconds"))
    if wall_clock_seconds < 0 or wall_clock_seconds > float(
        budget_contract.get("max_wall_clock_seconds")
    ):
        raise TrainingReceiptError("wall-clock budget exceeded")
    evidence_bytes = int(budget.get("evidence_bytes"))
    max_bytes = int(float(budget_contract.get("max_added_storage_gib")) * 1024**3)
    if evidence_bytes < 0 or evidence_bytes > max_bytes:
        raise TrainingReceiptError("evidence storage budget exceeded")
    peak_gpu_memory_mib = int(budget.get("peak_gpu_memory_mib"))
    if peak_gpu_memory_mib < 0 or peak_gpu_memory_mib > int(
        budget_contract.get("max_gpu_memory_mib")
    ):
        raise TrainingReceiptError("GPU-memory budget exceeded")
    if budget.get("visible_gpu_count") != 1 or budget.get("gpu_tasks_serial") is not True:
        raise TrainingReceiptError("single-GPU serial contract was not verified")

    integrity = _object(receipt.get("integrity"), "receipt.integrity")
    if integrity.get("gpu_compute_processes_empty_after") is not True:
        raise TrainingReceiptError("GPU was not empty after training")
    relative = integrity.get("verification_transcript_path")
    if not isinstance(relative, str) or not relative:
        raise TrainingReceiptError("verification transcript path is missing")
    transcript = (receipt_file.parent / relative).resolve()
    if receipt_file.parent not in transcript.parents or not transcript.is_file():
        raise TrainingReceiptError("verification transcript is missing or escapes receipt directory")
    if _digest(integrity.get("verification_transcript_sha256"), "transcript hash") != _sha256(transcript):
        raise TrainingReceiptError("verification transcript hash mismatch")

    return TrainingReceiptAudit(
        receipt_hash=_sha256(receipt_file),
        manifest_hash=manifest_hash,
        completed_steps=expected_steps,
        checkpoint_count=len(expected_checkpoint_steps),
        evidence_bytes=evidence_bytes,
        wall_clock_seconds=wall_clock_seconds,
    )
