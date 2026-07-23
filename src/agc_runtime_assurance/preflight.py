"""Fail-closed experiment authorization and split-boundary preflight."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import string
from typing import Any


class PreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class DevelopmentAuthorization:
    manifest_id: str
    experiment_family: str
    manifest_hash: str
    code_hash: str
    calibration_hash: str
    development_hash: str


_EXPERIMENT_FAMILIES = {
    "contract_smoke", "synthetic_counterexample", "strong_baseline_matrix",
}
_REQUIRED_STRONG_BASELINES = {
    "aoi_cbf", "fallback_safe_mpc", "acofi", "multiagent_conformal_cbf",
}


def verify_development_manifest(path: str | Path) -> DevelopmentAuthorization:
    raw = Path(path).read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    _require(manifest, "manifest_id", "stage", "authorized", "formal_experiment_authorized",
             "sealed_data_authorized", "claim_generation_allowed", "allowed_splits",
             "code_hash", "split_hashes", "experiment_family")
    if manifest["stage"] != "development_pilot":
        raise PreflightError("only development_pilot manifests are accepted by this runner")
    if manifest["authorized"] is not True:
        raise PreflightError("development pilot is not explicitly authorized")
    if manifest["formal_experiment_authorized"] is not False:
        raise PreflightError("formal experiment authority must remain false")
    if manifest["sealed_data_authorized"] is not False:
        raise PreflightError("sealed-data authority must remain false")
    if manifest["claim_generation_allowed"] is not False:
        raise PreflightError("development runs cannot generate paper claims")
    family = manifest["experiment_family"]
    if family not in _EXPERIMENT_FAMILIES:
        raise PreflightError("unknown or missing development experiment family")
    allowed = set(manifest["allowed_splits"])
    if allowed != {"calibration", "development"}:
        raise PreflightError("allowed_splits must be exactly calibration and development")
    hashes = manifest["split_hashes"]
    if set(hashes) != {"calibration", "development"}:
        raise PreflightError("only calibration and development hashes may be exposed")
    for name, value in {"code_hash": manifest["code_hash"], **hashes}.items():
        _verify_digest(value, name)
    if family == "strong_baseline_matrix":
        _verify_strong_baseline_artifacts(manifest)
    return DevelopmentAuthorization(
        str(manifest["manifest_id"]), str(family), hashlib.sha256(raw).hexdigest(),
        str(manifest["code_hash"]),
        str(hashes["calibration"]), str(hashes["development"]),
    )


def _verify_strong_baseline_artifacts(manifest: dict[str, Any]) -> None:
    _require(manifest, "baseline_artifacts")
    artifacts = manifest["baseline_artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != _REQUIRED_STRONG_BASELINES:
        raise PreflightError(
            "strong baseline matrix requires exactly AoI-CBF, Fallback-Safe MPC, "
            "ACoFi, and multi-agent conformal-CBF artifacts"
        )
    for name, record in artifacts.items():
        if not isinstance(record, dict):
            raise PreflightError(f"baseline artifact {name} must be an object")
        if record.get("status") != "verified":
            raise PreflightError(f"baseline artifact {name} is not verified")
        url = record.get("upstream_url")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise PreflightError(f"baseline artifact {name} needs an HTTPS upstream URL")
        if record.get("license_status") not in {
            "verified_permissive", "permission_documented",
        }:
            raise PreflightError(f"baseline artifact {name} license is not cleared")
        if record.get("reproduction_scope") != "original_and_1u1g_adapted":
            raise PreflightError(
                f"baseline artifact {name} lacks original-and-adapted task reproduction"
            )
        for digest_name in (
            "source_hash", "implementation_hash", "original_task_result_hash",
            "adapted_task_result_hash", "budget_contract_hash",
        ):
            _verify_digest(record.get(digest_name), f"baseline artifact {name} {digest_name}")


def _verify_digest(value: Any, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in string.hexdigits for character in value)
    ):
        raise PreflightError(f"{name} must be a 64-character hexadecimal digest")


def _require(manifest: dict[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in manifest]
    if missing:
        raise PreflightError(f"manifest missing required keys: {missing}")
