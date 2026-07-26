import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts" / "verify_circa_ress_v9_public_artifacts_one_command_r2.py"
)
SPEC = importlib.util.spec_from_file_location("public_verify_r2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_canonical_json_erases_platform_line_ending_difference(tmp_path):
    lf = tmp_path / "lf.json"
    crlf = tmp_path / "crlf.json"
    lf.write_bytes(b'{\n  "a": 1\n}\n')
    crlf.write_bytes(b'{\r\n  "a": 1\r\n}\r\n')
    assert MODULE.R1.sha256(lf) != MODULE.R1.sha256(crlf)
    assert MODULE.canonical_json_bytes(lf) == MODULE.canonical_json_bytes(crlf)


def test_canonical_json_sorts_keys_and_uses_lf(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"z": 2, "a": 1}', encoding="utf-8")
    canonical = MODULE.canonical_json_bytes(source)
    assert canonical == b'{\n  "a": 1,\n  "z": 2\n}\n'
    assert b"\r" not in canonical


def test_canonicalization_does_not_hide_value_change(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"value": 0.1}), encoding="utf-8")
    second.write_text(json.dumps({"value": 0.2}), encoding="utf-8")
    assert MODULE.canonical_json_bytes(first) != MODULE.canonical_json_bytes(
        second
    )
