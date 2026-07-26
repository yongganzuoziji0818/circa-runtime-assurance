"""Author-maintained, one-command verification of frozen V9 public artifacts.

This verifies archive and code identities, checks the evidence inventory, and
recomputes the already-frozen analysis from the immutable trace in a temporary
directory.  It never calls a simulator or scientific runner and is not an
external independent reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile


EVIDENCE_ZIP_SHA256 = (
    "57a39f079ea1dd1558ca90645a5384714f8ec8445e4ec8bff12d6769442a11dc"
)
INPUTS_ZIP_SHA256 = (
    "3f1b00763145d3cb88cab2884e45a5ca79abfa9e551c1a4762617995dc65fc74"
)
CODE_COMMIT = "38dda10b89bd21e64b48a7b59da4098713880c35"
ANALYSIS_SOURCE_SHA256 = (
    "7dabac1a16493c9c473a7157f63505e9e5de5abb1d32c030bf6f4b376c9a2437"
)
EXPECTED_ANALYSIS_SHA256 = (
    "8a0f492e2896273d980a716749de2fcdcd48818d865341721eda4932309fefe6"
)
MANIFEST_MEMBER = "inputs/V9_SCI_S3_SCHEDULED_RUNNABLE_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_zip_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts or not member.parts:
        raise ValueError(f"unsafe ZIP member: {name!r}")
    return member


def extract_archive(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path, "r") as archive:
        for info in archive.infolist():
            member = validate_zip_member(info.filename)
            target = destination.joinpath(*member.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            with archive.open(info, "r") as source, target.open("xb") as sink:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    sink.write(block)


def extract_manifest(archive_path: Path, destination: Path) -> Path:
    with zipfile.ZipFile(archive_path, "r") as archive:
        info = archive.getinfo(MANIFEST_MEMBER)
        validate_zip_member(info.filename)
        target = destination / "V9_SCI_S3_SCHEDULED_RUNNABLE_MANIFEST.json"
        with archive.open(info, "r") as source, target.open("xb") as sink:
            sink.write(source.read())
    return target


def verify_inventory(directory: Path) -> int:
    inventory = directory / "SHA256SUMS.txt"
    records = 0
    for line_number, line in enumerate(
        inventory.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"invalid evidence inventory line {line_number}")
        expected, relative = parts
        member = validate_zip_member(relative)
        path = directory.joinpath(*member.parts)
        if not path.is_file():
            raise FileNotFoundError(relative)
        if sha256(path) != expected:
            raise ValueError(f"evidence SHA-256 mismatch: {relative}")
        records += 1
    return records


def git_head(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_receipt(
    evidence_zip: Path,
    inputs_zip: Path,
    code_repository: Path,
) -> dict:
    if sha256(evidence_zip) != EVIDENCE_ZIP_SHA256:
        raise ValueError("frozen evidence archive SHA-256 mismatch")
    if sha256(inputs_zip) != INPUTS_ZIP_SHA256:
        raise ValueError("external-input archive SHA-256 mismatch")
    if git_head(code_repository) != CODE_COMMIT:
        raise ValueError("public code repository is not at the frozen commit")

    analysis_script = (
        code_repository / "scripts" / "analyze_circa_ress_v9_sci_s3_frozen.py"
    )
    if sha256(analysis_script) != ANALYSIS_SOURCE_SHA256:
        raise ValueError("frozen analysis source SHA-256 mismatch")

    with tempfile.TemporaryDirectory(prefix="circa-v9-public-verify-") as temp:
        temporary = Path(temp)
        evidence = temporary / "evidence"
        evidence.mkdir()
        extract_archive(evidence_zip, evidence)
        inventory_records = verify_inventory(evidence)
        manifest = extract_manifest(inputs_zip, temporary)
        recomputed = temporary / "recomputed_analysis.json"
        environment = os.environ.copy()
        source_root = str(code_repository / "src")
        environment["PYTHONPATH"] = source_root + os.pathsep + environment.get(
            "PYTHONPATH", ""
        )
        process = subprocess.run(
            [
                sys.executable,
                str(analysis_script),
                "--result-dir",
                str(evidence),
                "--manifest",
                str(manifest),
                "--output",
                str(recomputed),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        recomputed_hash = sha256(recomputed) if recomputed.is_file() else None
        if process.returncode != 0:
            raise RuntimeError(
                "frozen analysis recomputation failed: "
                + process.stderr[-2000:]
            )
        if recomputed_hash != EXPECTED_ANALYSIS_SHA256:
            raise ValueError("recomputed frozen analysis SHA-256 mismatch")
        archived_hash = sha256(
            evidence / "circa_ress_v9_sci_s3_frozen_analysis_20260726.json"
        )
        if archived_hash != EXPECTED_ANALYSIS_SHA256:
            raise ValueError("archived frozen analysis SHA-256 mismatch")

    return {
        "schema": "circa_ress_v10_author_public_artifact_verification/v1",
        "status": "PASS_AUTHOR_MAINTAINED_REPRODUCTION_NOT_EXTERNAL_INDEPENDENT",
        "evidence_zip_sha256": EVIDENCE_ZIP_SHA256,
        "external_inputs_zip_sha256": INPUTS_ZIP_SHA256,
        "code_commit": CODE_COMMIT,
        "analysis_source_sha256": ANALYSIS_SOURCE_SHA256,
        "evidence_inventory_records_verified": inventory_records,
        "recomputed_analysis_sha256": EXPECTED_ANALYSIS_SHA256,
        "matches_archived_frozen_analysis": True,
        "scientific_runner_invoked": False,
        "simulator_invoked": False,
        "new_or_replacement_seed_generated": False,
        "scientific_attempt_consumed": False,
        "external_independent_reproduction_claim": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-zip", required=True, type=Path)
    parser.add_argument("--external-inputs-zip", required=True, type=Path)
    parser.add_argument("--code-repository", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.receipt.exists():
        raise FileExistsError("verification receipt already exists")
    receipt = build_receipt(
        arguments.evidence_zip.resolve(),
        arguments.external_inputs_zip.resolve(),
        arguments.code_repository.resolve(),
    )
    arguments.receipt.parent.mkdir(parents=True, exist_ok=False)
    arguments.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
