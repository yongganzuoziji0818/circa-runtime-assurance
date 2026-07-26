"""Versioned scientific runner surface for the confirmed CIRCA-RESS-V9 route.

Importing this module is inert.  The entry point remains fail-closed behind a
future runnable manifest.  The implementation reuses frozen V8 method/world
primitives while replacing only the V9 schedule dimensions, route identity,
schema dimensions, and feasible initial-state generator.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterator, Mapping

from . import circa_ress_v8_sim_diversity_runner as _core
from .circa_ress_v9_initial_domain import (
    SEED_NAMESPACE,
    initial_state_from_future_seed,
)
from .circa_ress_v9_schema import array_schema


ROUTE_ID = "CIRCA-RESS-V9-FEASIBLE-INITIAL-DOMAIN-R1"
ROLLOUTS = 7424
HORIZON = 80
OUTPUT_CAPACITY_BYTES = 402_653_184
CAPACITY_RECEIPT_SHA256 = (
    "abf852867859d0562c26fb033015e57f450430860cab9e6356faf0f1042118f9"
)
VALIDATION_COUNT_PER_CANDIDATE = 32
POSITIVE_EVALUATION_COUNT_PER_CANDIDATE = 80
SDF5_EVALUATION_COUNT_PER_CANDIDATE = 64
INDEPENDENT_UNITS = 1088

RunSpec = _core.RunSpec
DiversityRunnerError = _core.DiversityRunnerError


def derive_future_seed(
    master_seed: int,
    split: str,
    family: str,
    candidate: str,
    index: int,
) -> int:
    if (
        not isinstance(master_seed, int)
        or master_seed < 0
        or split not in _core.SPLIT_CODEBOOK
        or family not in _core.FAMILY_CODEBOOK
        or candidate not in _core.CANDIDATE_CODEBOOK
        or not isinstance(index, int)
        or index < 0
    ):
        raise DiversityRunnerError("invalid future seed derivation input")
    payload = f"{SEED_NAMESPACE}|{master_seed}|{split}|{family}|{candidate}|{index}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (
        2**31 - 1
    )


def compile_schedule(manifest: Mapping[str, Any]) -> list[RunSpec]:
    master_seed = manifest.get("master_seed")
    schedule_seed = manifest.get("schedule_seed")
    if not isinstance(master_seed, int) or not isinstance(schedule_seed, int):
        raise DiversityRunnerError("scientific seed material is absent")
    schedule: list[RunSpec] = []
    pair_id = 0
    seen: set[tuple[str, str, str, int]] = set()
    for split_index, split in enumerate(_core.SPLIT_CODEBOOK):
        methods = (
            _core.VALIDATION_METHODS
            if split == "validation"
            else _core.METHOD_CODEBOOK
        )
        for family_index, family in enumerate(_core.FAMILY_CODEBOOK):
            count = (
                VALIDATION_COUNT_PER_CANDIDATE
                if split == "validation"
                else (
                    SDF5_EVALUATION_COUNT_PER_CANDIDATE
                    if family == "SDF5"
                    else POSITIVE_EVALUATION_COUNT_PER_CANDIDATE
                )
            )
            for candidate_index, candidate in enumerate(_core.CANDIDATE_CODEBOOK):
                for seed_index in range(count):
                    seed = derive_future_seed(
                        master_seed, split, family, candidate, seed_index
                    )
                    key = (split, family, candidate, seed)
                    if key in seen:
                        raise DiversityRunnerError("future seed collision")
                    seen.add(key)
                    for driver_index in range(len(_core.DRIVER_CODEBOOK)):
                        for method in methods:
                            schedule.append(
                                RunSpec(
                                    family_index=family_index,
                                    candidate_index=candidate_index,
                                    split_index=split_index,
                                    seed_index=seed_index,
                                    future_seed=seed,
                                    driver_index=driver_index,
                                    method_index=_core.METHOD_CODEBOOK.index(method),
                                    pair_id=pair_id,
                                )
                            )
                    pair_id += 1
    if (
        len(schedule) != ROLLOUTS
        or pair_id != INDEPENDENT_UNITS
        or len(seen) != INDEPENDENT_UNITS
    ):
        raise DiversityRunnerError("frozen V9 paired schedule dimensions drifted")
    random.Random(schedule_seed).shuffle(schedule)
    return schedule


class _FeasibleInitialStateEnvironment:
    def __init__(self, wrapped: Any):
        self._wrapped = wrapped

    def reset(
        self,
        *,
        seed: int,
        initial_state: Any = None,
        position_jitter_scale: float = 0.0,
        **kwargs: Any,
    ) -> Any:
        if initial_state is None:
            raise DiversityRunnerError("V9 requires the frozen nominal state")
        state = initial_state_from_future_seed(seed)
        return self._wrapped.reset(
            seed=seed,
            initial_state=state,
            position_jitter_scale=0.0,
            **kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


def _environment(
    driver: str,
    world: Path,
    shift: Any,
    action_limit: float,
) -> _FeasibleInitialStateEnvironment:
    return _FeasibleInitialStateEnvironment(
        _ORIGINALS["_environment"](driver, world, shift, action_limit)
    )


_PATCH_NAMES = (
    "ROUTE_ID",
    "SEED_NAMESPACE",
    "ROLLOUTS",
    "HORIZON",
    "OUTPUT_CAPACITY_BYTES",
    "CAPACITY_RECEIPT_SHA256",
    "array_schema",
    "compile_schedule",
    "derive_future_seed",
    "_environment",
    "validate_runnable_manifest",
)
_ORIGINALS = {name: getattr(_core, name) for name in _PATCH_NAMES}


def _safe_json(repo_root: Path, relative: str, expected_hash: str) -> dict[str, Any]:
    path = _core._safe_child(repo_root, relative)
    if not path.is_file() or _core.sha256(path) != expected_hash:
        raise DiversityRunnerError(f"V9 manifest lock failed: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_v9_manifest(
    manifest: Mapping[str, Any], repo_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path]:
    v9_design = _safe_json(
        repo_root,
        str(manifest.get("v9_design_path", "")),
        str(manifest.get("v9_design_sha256", "")),
    )
    confirmation = _safe_json(
        repo_root,
        str(manifest.get("v9_confirmation_path", "")),
        str(manifest.get("v9_confirmation_sha256", "")),
    )
    if (
        v9_design.get("route_id") != ROUTE_ID
        or v9_design.get("status") != "CONFIRMED_FROZEN_SCIENTIFIC_CONTRACT"
        or v9_design.get("schedule", {}).get("rollouts") != ROLLOUTS
        or v9_design.get("schedule", {}).get("independent_units_total")
        != INDEPENDENT_UNITS
    ):
        raise DiversityRunnerError("V9 frozen design lock failed")
    if (
        confirmation.get("route_id") != ROUTE_ID
        or confirmation.get("status") != "CONFIRMED_FROZEN_SCIENTIFIC_CONTRACT"
        or not confirmation.get("new_route_not_v8_retry")
        or confirmation.get("retry_allowed")
    ):
        raise DiversityRunnerError("V9 confirmation lock failed")
    capacity = _safe_json(
        repo_root,
        str(manifest.get("capacity_receipt_path", "")),
        CAPACITY_RECEIPT_SHA256,
    )
    if (
        capacity.get("status") != "PASS_HIGH_ENTROPY_LOSSLESS_SCHEMA_CAPACITY"
        or capacity.get("frozen_cap_bytes") != OUTPUT_CAPACITY_BYTES
        or capacity.get("rollouts") != ROLLOUTS
    ):
        raise DiversityRunnerError("V9 capacity receipt did not pass")
    seed_receipt = _safe_json(
        repo_root,
        str(manifest.get("seed_receipt_path", "")),
        str(manifest.get("seed_receipt_sha256", "")),
    )
    if (
        seed_receipt.get("route_id") != ROUTE_ID
        or seed_receipt.get("schedule_rollout_count") != ROLLOUTS
        or seed_receipt.get("independent_unit_count") != INDEPENDENT_UNITS
    ):
        raise DiversityRunnerError("V9 future seed receipt dimensions drifted")
    return _ORIGINALS["validate_runnable_manifest"](manifest, repo_root)


@contextmanager
def _configured_core() -> Iterator[None]:
    replacements = {
        "ROUTE_ID": ROUTE_ID,
        "SEED_NAMESPACE": SEED_NAMESPACE,
        "ROLLOUTS": ROLLOUTS,
        "HORIZON": HORIZON,
        "OUTPUT_CAPACITY_BYTES": OUTPUT_CAPACITY_BYTES,
        "CAPACITY_RECEIPT_SHA256": CAPACITY_RECEIPT_SHA256,
        "array_schema": array_schema,
        "compile_schedule": compile_schedule,
        "derive_future_seed": derive_future_seed,
        "_environment": _environment,
        "validate_runnable_manifest": _validate_v9_manifest,
    }
    for name, value in replacements.items():
        setattr(_core, name, value)
    try:
        yield
    finally:
        for name, value in _ORIGINALS.items():
            setattr(_core, name, value)


def validate_runnable_manifest(
    manifest: Mapping[str, Any], repo_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path]:
    with _configured_core():
        return _validate_v9_manifest(manifest, repo_root)


def run(manifest_path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    with _configured_core():
        return _core.run(manifest_path, repo_root)


__all__ = [
    "DiversityRunnerError",
    "HORIZON",
    "INDEPENDENT_UNITS",
    "OUTPUT_CAPACITY_BYTES",
    "ROLLOUTS",
    "ROUTE_ID",
    "RunSpec",
    "compile_schedule",
    "derive_future_seed",
    "run",
    "validate_runnable_manifest",
]
