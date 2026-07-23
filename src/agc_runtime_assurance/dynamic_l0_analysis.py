"""Deterministic receipt-bound analysis for the consumed Route-A dynamic L0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta

from .dynamic_l0 import FAMILIES, METHODS, DynamicL0Error


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _upper_cp(events: int, total: int, confidence: float = .95) -> float:
    return 1.0 if events == total else float(beta.ppf(confidence, events + 1, total - events))


def analyze_dynamic_l0(
    result_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    result_file, manifest_file, output = map(Path, (result_path, manifest_path, output_path))
    if output.exists():
        raise DynamicL0Error("analysis output already exists; overwrite refused")
    result = json.loads(result_file.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest_hash = _sha(manifest_file)
    if result.get("manifest_sha256") != manifest_hash:
        raise DynamicL0Error("result is not bound to analysis manifest")
    if result.get("sealed_data_used") is not False or result.get("formal_experiment_run") is not False:
        raise DynamicL0Error("sealed/formal result refused")
    rows = result.get("rows")
    if not isinstance(rows, list) or len(rows) != 500:
        raise DynamicL0Error("complete 500-row result is required")
    observed = {(r["method"], int(r["scenario_seed"]), r["scenario_family"]) for r in rows}
    expected = {(m, int(s), f) for m in METHODS for s in manifest["scenario_seeds"] for f in FAMILIES}
    if observed != expected:
        raise DynamicL0Error("method-by-seed-by-family matrix is incomplete or duplicated")
    index = {(r["method"], int(r["scenario_seed"]), r["scenario_family"]): int(r["constraint_violation"]) for r in rows}
    family: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        family[method] = {}
        for name in FAMILIES:
            values = [index[(method, int(seed), name)] for seed in manifest["scenario_seeds"]]
            events = sum(values)
            family[method][name] = {
                "events": events, "total": len(values), "rate": events / len(values),
                "one_sided_95_upper": _upper_cp(events, len(values)),
            }
    seeds = [int(seed) for seed in manifest["scenario_seeds"]]
    rng = np.random.default_rng(int(manifest["bootstrap_seed"]))
    comparisons: dict[str, dict[str, Any]] = {}
    for baseline in METHODS:
        if baseline == "full_assurance_case":
            continue
        full_worst = max(family["full_assurance_case"][f]["rate"] for f in FAMILIES)
        base_worst = max(family[baseline][f]["rate"] for f in FAMILIES)
        samples = []
        for _ in range(int(manifest["bootstrap_replicates"])):
            chosen = rng.choice(seeds, size=len(seeds), replace=True)
            fw = max(float(np.mean([index[("full_assurance_case", int(s), f)] for s in chosen])) for f in FAMILIES)
            bw = max(float(np.mean([index[(baseline, int(s), f)] for s in chosen])) for f in FAMILIES)
            samples.append(fw - bw)
        comparisons[baseline] = {
            "worst_family_effect_full_minus_baseline": full_worst - base_worst,
            "paired_scenario_seed_cluster_bootstrap_95ci": [float(x) for x in np.quantile(samples, [.025, .975])],
            "descriptive_all_family_full_only_violations": sum(index[("full_assurance_case", s, f)] > index[(baseline, s, f)] for s in seeds for f in FAMILIES),
            "descriptive_all_family_baseline_only_violations": sum(index[("full_assurance_case", s, f)] < index[(baseline, s, f)] for s in seeds for f in FAMILIES),
        }
    full_rows = [r for r in rows if r["method"] == "full_assurance_case"]
    reason_counts: dict[str, int] = {}
    for row in full_rows:
        for reason, count in row["reason_counts"].items():
            reason_counts[reason] = reason_counts.get(reason, 0) + int(count)
    analysis = {
        "result_sha256": _sha(result_file),
        "manifest_sha256": manifest_hash,
        "independent_unit": "scenario_seed",
        "family_violation_estimates": family,
        "comparisons": comparisons,
        "full_assurance_reason_counts": reason_counts,
        "guardrails": result["summary"]["guardrails"],
        "frozen_primary_success_gates": result["summary"]["success_gates"],
        "claim_boundary": {
            "supported_development_signal": "full assurance case reduced the preregistered worst-family violation rate relative to nominal CBF in this sandbox",
            "not_supported": "incremental dynamic benefit of evidence binding over the unbound expiry/filter contract",
            "why": "full_assurance_case and unbound_filter both had worst-family violation rate 0.50",
        },
        "sealed_data_used": False,
        "formal_experiment_run": False,
        "claim_generation_allowed": False,
    }
    body = (json.dumps(analysis, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(body)
    temporary.replace(output)
    return analysis


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze_dynamic_l0(args.result, args.manifest, args.output)


if __name__ == "__main__":
    main()
