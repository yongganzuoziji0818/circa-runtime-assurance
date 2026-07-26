from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from agc_runtime_assurance.circa_ress_v8_sim_diversity_schema import (
    validate_schema_arrays,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "audit_circa_ress_v8_sim_diversity_r1_schema_capacity.py"
)


def load_auditor():
    spec = importlib.util.spec_from_file_location("v8_capacity_auditor", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_small_high_entropy_fixture_is_valid_and_seedless() -> None:
    auditor = load_auditor()
    arrays = auditor.build_high_entropy_arrays(7, 5)
    validate_schema_arrays(arrays, 7, 5)
    assert np.all(arrays["future_seed_sentinel"] == -1)
    assert np.isfinite(arrays["physics_time_s"]).all()


def test_small_lossless_archive_round_trip_is_verified_in_memory() -> None:
    auditor = load_auditor()
    arrays = auditor.build_high_entropy_arrays(7, 5)
    archive, archive_hash, source_hash, verified = auditor.archive_and_verify(arrays)
    assert archive.startswith(b"PK")
    assert len(archive_hash) == 64
    assert len(source_hash) == 64
    assert verified == 45


def test_exactly_once_and_nonrunning_guards_are_present() -> None:
    body = SCRIPT.read_text(encoding="utf-8")
    assert "os.O_EXCL" in body
    assert "attempt_consumed_when_lock_is_created" not in body
    assert "np.savez_compressed" in body
    assert "archive_written_to_disk" in body
    assert "subprocess" not in body
    assert "scientific_attempts_consumed" in body


def test_frozen_capacity_is_exactly_256_mib() -> None:
    auditor = load_auditor()
    assert auditor.CAPACITY_BYTES == 256 * 1024 * 1024
