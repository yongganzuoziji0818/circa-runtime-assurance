"""V3 evidence-storage revision for the frozen proposal-efficiency experiment.

Scientific computation is delegated unchanged to V2.  V3 only separates the
aggregate JSON summary from lossless typed replication arrays so the complete
evidence fits the original 16 MiB output budget.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "p4_proposal_efficiency_g1_v2_preserved", HERE / "proposal_efficiency_g1_v2.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load preserved proposal-efficiency G1 v2 runner")
V2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = V2
SPEC.loader.exec_module(V2)


ARRAY_DTYPES = {
    "total_path_budget": np.dtype("<i4"),
    "method_index": np.dtype("u1"),
    "family_index": np.dtype("u1"),
    "replication": np.dtype("<u2"),
    "estimate": np.dtype("<f8"),
    "lower": np.dtype("<f8"),
    "upper": np.dtype("<f8"),
    "width": np.dtype("<f8"),
    "covered": np.dtype("u1"),
    "positive_lower": np.dtype("u1"),
    "ess_fraction": np.dtype("<f8"),
    "maximum_weight": np.dtype("<f8"),
    "screen_paths": np.dtype("<i4"),
    "evaluation_paths": np.dtype("<i4"),
    "simulator_calls": np.dtype("<i4"),
    "proposal_sha256": np.dtype("S64"),
    "method_order_sha256": np.dtype("S64"),
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays_from_analysis(
    analysis: dict[str, Any], methods: list[str], families: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    columns: dict[str, list[Any]] = {name: [] for name in ARRAY_DTYPES}
    summary_budgets = []
    for budget_result in analysis["budget_results"]:
        budget = int(budget_result["total_path_budget"])
        summary_budgets.append(
            {key: value for key, value in budget_result.items() if key != "replication_records"}
        )
        for record in budget_result["replication_records"]:
            method = record["method"]
            family = record["family"]
            columns["total_path_budget"].append(budget)
            columns["method_index"].append(methods.index(method))
            columns["family_index"].append(families.index(family))
            for name in (
                "replication",
                "estimate",
                "lower",
                "upper",
                "width",
                "covered",
                "positive_lower",
                "ess_fraction",
                "maximum_weight",
                "screen_paths",
                "evaluation_paths",
                "simulator_calls",
                "proposal_sha256",
                "method_order_sha256",
            ):
                columns[name].append(record[name])
    arrays = {
        name: np.asarray(values, dtype=dtype) for name, (dtype, values) in (
            (name, (ARRAY_DTYPES[name], columns[name])) for name in ARRAY_DTYPES
        )
    }
    row_counts = {array.shape[0] for array in arrays.values()}
    if len(row_counts) != 1:
        raise V2.BASE.ProposalEfficiencyG1Error("V3 evidence columns have unequal lengths")
    summary = dict(analysis)
    summary["budget_results"] = summary_budgets
    summary["replication_records_storage"] = "lossless_npz_columns"
    summary["replication_record_count"] = row_counts.pop()
    return arrays, summary


def schema_only_arrays(record_count: int) -> dict[str, np.ndarray]:
    if not isinstance(record_count, int) or record_count < 1:
        raise V2.BASE.ProposalEfficiencyG1Error("schema record count must be positive")
    arrays = {}
    for name, dtype in ARRAY_DTYPES.items():
        if dtype.kind == "S":
            arrays[name] = np.full(record_count, b"f" * 64, dtype=dtype)
        elif dtype.kind == "f":
            arrays[name] = np.full(record_count, np.finfo(dtype).max / 4.0, dtype=dtype)
        else:
            arrays[name] = np.zeros(record_count, dtype=dtype)
    return arrays


def write_lossless_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if set(arrays) != set(ARRAY_DTYPES):
        raise V2.BASE.ProposalEfficiencyG1Error("V3 evidence array schema mismatch")
    for name, expected in ARRAY_DTYPES.items():
        if arrays[name].dtype != expected or arrays[name].ndim != 1:
            raise V2.BASE.ProposalEfficiencyG1Error(f"V3 evidence dtype mismatch: {name}")
    np.savez(path, **arrays)


def verify_lossless_arrays(path: Path, expected_rows: int) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != set(ARRAY_DTYPES):
            raise V2.BASE.ProposalEfficiencyG1Error("V3 NPZ field set mismatch")
        for name, dtype in ARRAY_DTYPES.items():
            array = payload[name]
            if array.dtype != dtype or array.shape != (expected_rows,):
                raise V2.BASE.ProposalEfficiencyG1Error(f"V3 NPZ verification failed: {name}")
    return {
        "row_count": expected_rows,
        "field_count": len(ARRAY_DTYPES),
        "sha256": _digest(path),
        "bytes": path.stat().st_size,
    }


def schema_capacity_preflight(record_count: int, summary_reserve_bytes: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="p4_v3_schema_") as temporary:
        path = Path(temporary) / "replications.npz"
        arrays = schema_only_arrays(record_count)
        write_lossless_arrays(path, arrays)
        verified = verify_lossless_arrays(path, record_count)
    return {
        "record_count": record_count,
        "field_count": len(ARRAY_DTYPES),
        "npz_bytes": verified["bytes"],
        "summary_reserve_bytes": summary_reserve_bytes,
        "projected_total_bytes": verified["bytes"] + summary_reserve_bytes,
    }


def validate_manifest_v3(manifest: dict[str, Any], root: Path) -> None:
    if manifest.get("manifest_id") != "p4-proposal-efficiency-g1-v3-2026-07-19":
        raise V2.BASE.ProposalEfficiencyG1Error("V3 manifest id mismatch")
    if manifest.get("engineering_revision") != "lossless_split_evidence_storage_only":
        raise V2.BASE.ProposalEfficiencyG1Error("V3 engineering revision expanded")
    if manifest.get("supersedes_failed_v2_manifest_sha256") != (
        "a16823e5556b12aeeaa2174b06512b35033d5a37c80292a25f99443a8f9554b4"
    ):
        raise V2.BASE.ProposalEfficiencyG1Error("V3 does not bind failed V2")
    if manifest.get("output_path") != "experiments/results/proposal_efficiency_g1_v3/summary.json":
        raise V2.BASE.ProposalEfficiencyG1Error("V3 summary path mismatch")
    if manifest.get("arrays_path") != "experiments/results/proposal_efficiency_g1_v3/replications.npz":
        raise V2.BASE.ProposalEfficiencyG1Error("V3 arrays path mismatch")
    if manifest.get("evidence_schema") != "summary_json_plus_lossless_npz_v1":
        raise V2.BASE.ProposalEfficiencyG1Error("V3 evidence schema mismatch")
    if manifest.get("expected_replication_records") != 24000:
        raise V2.BASE.ProposalEfficiencyG1Error("V3 evidence cardinality changed")
    transformed = dict(manifest)
    transformed["manifest_id"] = "p4-proposal-efficiency-g1-v2-2026-07-19"
    transformed["engineering_revision"] = "exact_nominal_rho_one_boundary_only"
    transformed["output_path"] = "experiments/results/proposal_efficiency_g1_v2/result.json"
    transformed["launch_authorization_path"] = (
        "experiments/manifests/proposal_efficiency_g1_v2_launch_authorization.json"
    )
    V2.validate_manifest_v2(transformed, root)
    for protected in (
        "experiments/results/proposal_efficiency_g1_v1",
        "experiments/results/proposal_efficiency_g1_v2",
    ):
        if protected not in manifest["protected_paths"]:
            raise V2.BASE.ProposalEfficiencyG1Error("V3 failed-run protection is incomplete")


def validate_launch_authorization_v3(
    authorization: dict[str, Any], manifest_raw: bytes, manifest_path: Path
) -> None:
    expected = (
        "C:/Users/liaoy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe "
        "src/agc_runtime_assurance/proposal_efficiency_g1_v3.py "
        "--manifest experiments/manifests/proposal_efficiency_g1_v3.json "
        "--authorization experiments/manifests/proposal_efficiency_g1_v3_launch_authorization.json "
        "--repo-root . --output experiments/results/proposal_efficiency_g1_v3/summary.json"
    )
    if authorization.get("authorization_id") != "p4-proposal-efficiency-g1-v3-final-one-shot":
        raise V2.BASE.ProposalEfficiencyG1Error("V3 authorization id mismatch")
    if authorization.get("authorized") is not True or authorization.get("retry_allowed") is not False:
        raise V2.BASE.ProposalEfficiencyG1Error("V3 authorization is invalid")
    if authorization.get("manifest_sha256") != hashlib.sha256(manifest_raw).hexdigest():
        raise V2.BASE.ProposalEfficiencyG1Error("V3 authorization does not bind manifest")
    if authorization.get("manifest_path") != manifest_path.as_posix():
        raise V2.BASE.ProposalEfficiencyG1Error("V3 authorization path mismatch")
    if authorization.get("authorized_command") != expected:
        raise V2.BASE.ProposalEfficiencyG1Error("V3 authorization command mismatch")


def run_v3(
    manifest_path: str | Path,
    authorization_path: str | Path,
    repo_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest_raw = manifest_file.read_bytes()
    manifest = json.loads(manifest_raw.decode("utf-8"))
    validate_manifest_v3(manifest, root)
    authorization = json.loads(Path(authorization_path).read_text(encoding="utf-8"))
    validate_launch_authorization_v3(
        authorization, manifest_raw, manifest_file.relative_to(root)
    )
    output = Path(output_path).resolve()
    expected_summary = (root / manifest["output_path"]).resolve()
    expected_arrays = (root / manifest["arrays_path"]).resolve()
    if output != expected_summary or output.parent != expected_arrays.parent:
        raise V2.BASE.ProposalEfficiencyG1Error("V3 output paths differ from manifest")
    if output.parent.exists():
        raise V2.BASE.ProposalEfficiencyG1Error("V3 output directory exists; re-execution refused")

    families = V2.BASE._load_benchmark(root, manifest)
    start = time.perf_counter()
    analysis = V2.BASE.simulate(manifest, families)
    arrays, summary_analysis = arrays_from_analysis(
        analysis, list(manifest["methods"]), list(manifest["registered_families"])
    )
    if summary_analysis["replication_record_count"] != manifest["expected_replication_records"]:
        raise V2.BASE.ProposalEfficiencyG1Error("V3 produced unexpected evidence cardinality")

    results_root = expected_summary.parent.parent
    results_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="p4_g1_v3_", dir=results_root) as temporary:
        temporary_dir = Path(temporary)
        arrays_file = temporary_dir / "replications.npz"
        summary_file = temporary_dir / "summary.json"
        write_lossless_arrays(arrays_file, arrays)
        arrays_receipt = verify_lossless_arrays(
            arrays_file, manifest["expected_replication_records"]
        )
        result = {
            "result_id": "p4-proposal-efficiency-g1-v3",
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "benchmark_sha256": V2.BASE._digest(root / manifest["benchmark_path"]),
            "engineering_revision": "lossless_split_evidence_storage_only",
            "failed_v2_manifest_sha256": manifest["supersedes_failed_v2_manifest_sha256"],
            "scientific_route_confirmed": True,
            "paper_efficacy_claim_allowed": False,
            "controller_efficacy_experiment": False,
            "development_proposal_efficiency_evidence": True,
            "sealed_data_used": False,
            "formal_or_g2": False,
            "gpu_used": False,
            "workers": 1,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "evidence_arrays": arrays_receipt,
            "analysis": summary_analysis,
            "elapsed_seconds": time.perf_counter() - start,
            "inference_boundary": "Known-probability equal-call proposal-efficiency development evidence only; no controller efficacy, Gazebo, 1U1G, sealed, formal, G2, pilot, hardware, or real-platform claim.",
        }
        body = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
        summary_file.write_bytes(body)
        elapsed = time.perf_counter() - start
        total_bytes = summary_file.stat().st_size + arrays_file.stat().st_size
        if elapsed > manifest["budgets"]["max_runtime_seconds"]:
            raise V2.BASE.ProposalEfficiencyG1Error("V3 runtime budget exceeded")
        if summary_file.stat().st_size > manifest["summary_max_bytes"]:
            raise V2.BASE.ProposalEfficiencyG1Error("V3 summary budget exceeded")
        if total_bytes > manifest["budgets"]["max_output_bytes"]:
            raise V2.BASE.ProposalEfficiencyG1Error("V3 total output budget exceeded")
        temporary_dir.replace(output.parent)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_v3(args.manifest, args.authorization, args.repo_root, args.output)
    summary = {
        "elapsed_seconds": result["elapsed_seconds"],
        "summary_bytes": Path(args.output).stat().st_size,
        "arrays_bytes": result["evidence_arrays"]["bytes"],
        **{key: value for key, value in result["analysis"].items() if key.endswith("passed")},
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
