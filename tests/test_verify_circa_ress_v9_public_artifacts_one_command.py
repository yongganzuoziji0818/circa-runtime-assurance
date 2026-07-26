import importlib.util
from pathlib import Path
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_circa_ress_v9_public_artifacts_one_command.py"
SPEC = importlib.util.spec_from_file_location("public_verify", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_validate_zip_member_accepts_posix_relative_path():
    member = MODULE.validate_zip_member("inputs/trace_arrays.npz")
    assert member.parts == ("inputs", "trace_arrays.npz")


@pytest.mark.parametrize(
    "name",
    ["/absolute.txt", "../escape.txt", "inputs/../../escape.txt"],
)
def test_validate_zip_member_refuses_traversal(name):
    with pytest.raises(ValueError):
        MODULE.validate_zip_member(name)


def test_verify_inventory_checks_all_members(tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"immutable")
    digest = MODULE.sha256(payload)
    (tmp_path / "SHA256SUMS.txt").write_text(
        f"{digest}  payload.bin\n", encoding="utf-8"
    )
    assert MODULE.verify_inventory(tmp_path) == 1


def test_extract_manifest_selects_only_frozen_manifest(tmp_path):
    archive_path = tmp_path / "inputs.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(MODULE.MANIFEST_MEMBER, b'{"frozen": true}\n')
        archive.writestr("inputs/other.json", b"{}")
    output = tmp_path / "out"
    output.mkdir()
    manifest = MODULE.extract_manifest(archive_path, output)
    assert manifest.read_bytes() == b'{"frozen": true}\n'
    assert sorted(path.name for path in output.iterdir()) == [
        "V9_SCI_S3_SCHEDULED_RUNNABLE_MANIFEST.json"
    ]
