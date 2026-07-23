"""Tamper-evident JSONL audit trail for assurance decisions."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable


GENESIS_HASH = "0" * 64


def _digest(previous_hash: str, payload: dict[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256((previous_hash + body).encode()).hexdigest()


class HashChainAuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, payload: dict[str, Any]) -> str:
        previous = GENESIS_HASH
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as handle:
                last = list(handle)[-1]
            previous = json.loads(last)["hash"]
        entry_hash = _digest(previous, payload)
        entry = {"previous_hash": previous, "payload": payload, "hash": entry_hash}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
        return entry_hash

    def verify(self) -> bool:
        previous = GENESIS_HASH
        if not self.path.exists():
            return True
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                entry = json.loads(line)
                if entry["previous_hash"] != previous:
                    return False
                if entry["hash"] != _digest(previous, entry["payload"]):
                    return False
                previous = entry["hash"]
        return True


def verify_entries(entries: Iterable[dict[str, Any]]) -> bool:
    previous = GENESIS_HASH
    for entry in entries:
        if entry.get("previous_hash") != previous:
            return False
        if entry.get("hash") != _digest(previous, entry.get("payload", {})):
            return False
        previous = entry["hash"]
    return True
