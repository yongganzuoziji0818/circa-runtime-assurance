"""Cross-platform R2 author verification of frozen V9 public artifacts.

The only delta from R1 is explicit UTF-8/LF JSON canonicalization plus semantic
equality checking.  Scientific values, archives, code commit, and analysis
source are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


R1_PATH = (
    Path(__file__).resolve().parent
    / "verify_circa_ress_v9_public_artifacts_one_command.py"
)
SPEC = importlib.util.spec_from_file_location("circa_public_verify_r1", R1_PATH)
R1 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(R1)

R1_SOURCE_SHA256 = (
    "2c22fd3a4c53b2ea4d80533612f8d0d7ce36e26cddc515c16dbe6eeb74b0a304"
)


def canonical_json_bytes(path: Path) -> bytes:
    value = json.loads(path.read_text(encoding="utf-8"))
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_receipt_r2(
    evidence_zip: Path,
    inputs_zip: Path,
    code_repository: Path,
) -> dict:
    if R1.sha256(R1_PATH) != R1_SOURCE_SHA256:
        raise ValueError("locked R1 verifier source SHA-256 mismatch")
    if R1.sha256(evidence_zip) != R1.EVIDENCE_ZIP_SHA256:
        raise ValueError("frozen evidence archive SHA-256 mismatch")
    if R1.sha256(inputs_zip) != R1.INPUTS_ZIP_SHA256:
        raise ValueError("external-input archive SHA-256 mismatch")
    if R1.git_head(code_repository) != R1.CODE_COMMIT:
        raise ValueError("public code repository is not at the frozen commit")

    analysis_script = (
        code_repository / "scripts" / "analyze_circa_ress_v9_sci_s3_frozen.py"
    )
    if R1.sha256(analysis_script) != R1.ANALYSIS_SOURCE_SHA256:
        raise ValueError("frozen analysis source SHA-256 mismatch")

    with tempfile.TemporaryDirectory(prefix="circa-v9-public-verify-r2-") as temp:
        temporary = Path(temp)
        evidence = temporary / "evidence"
        evidence.mkdir()
        R1.extract_archive(evidence_zip, evidence)
        inventory_records = R1.verify_inventory(evidence)
        manifest = R1.extract_manifest(inputs_zip, temporary)
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
        if process.returncode != 0:
            raise RuntimeError(
                "frozen analysis recomputation failed: " + process.stderr[-2000:]
            )

        archived = (
            evidence / "circa_ress_v9_sci_s3_frozen_analysis_20260726.json"
        )
        recomputed_object = json.loads(recomputed.read_text(encoding="utf-8"))
        archived_object = json.loads(archived.read_text(encoding="utf-8"))
        if recomputed_object != archived_object:
            raise ValueError("recomputed and archived analysis JSON differ semantically")

        raw_recomputed_hash = R1.sha256(recomputed)
        canonical_recomputed_hash = bytes_sha256(canonical_json_bytes(recomputed))
        canonical_archived_hash = bytes_sha256(canonical_json_bytes(archived))
        if canonical_recomputed_hash != R1.EXPECTED_ANALYSIS_SHA256:
            raise ValueError("canonical recomputed analysis SHA-256 mismatch")
        if canonical_archived_hash != R1.EXPECTED_ANALYSIS_SHA256:
            raise ValueError("canonical archived analysis SHA-256 mismatch")

    return {
        "schema": "circa_ress_v10_author_public_artifact_verification/v2",
        "status": "PASS_AUTHOR_MAINTAINED_REPRODUCTION_NOT_EXTERNAL_INDEPENDENT",
        "r2_delta": "UTF8_LF_CANONICAL_JSON_AND_SEMANTIC_EQUALITY_ONLY",
        "evidence_zip_sha256": R1.EVIDENCE_ZIP_SHA256,
        "external_inputs_zip_sha256": R1.INPUTS_ZIP_SHA256,
        "code_commit": R1.CODE_COMMIT,
        "analysis_source_sha256": R1.ANALYSIS_SOURCE_SHA256,
        "evidence_inventory_records_verified": inventory_records,
        "raw_platform_recomputed_sha256": raw_recomputed_hash,
        "canonical_recomputed_analysis_sha256": canonical_recomputed_hash,
        "canonical_archived_analysis_sha256": canonical_archived_hash,
        "semantic_json_equality": True,
        "matches_frozen_analysis_commitment": True,
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
        raise FileExistsError("R2 verification receipt already exists")
    receipt = build_receipt_r2(
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
