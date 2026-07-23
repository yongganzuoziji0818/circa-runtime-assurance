import json

import pytest

from agc_runtime_assurance.audit import HashChainAuditLog
from agc_runtime_assurance.splits import build_split_manifest


def test_hash_chain_detects_tampering(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = HashChainAuditLog(path)
    log.append({"mode": "nominal", "step": 1})
    log.append({"mode": "backup", "step": 2})
    assert log.verify()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["payload"]["mode"] = "tampered"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert not log.verify()


def test_split_leakage_is_fail_closed():
    with pytest.raises(ValueError, match="split leakage"):
        build_split_manifest(train=[1, 2], calibration=[2, 3], development=[4], sealed=[5])


def test_manifest_stores_hashes_not_seed_values():
    manifest = build_split_manifest(train=[1], calibration=[2], development=[3], sealed=[4])
    assert manifest.counts == {"train": 1, "calibration": 1, "development": 1, "sealed": 1}
    assert len(manifest.sealed_hash) == 64
