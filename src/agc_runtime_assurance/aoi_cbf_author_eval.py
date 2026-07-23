"""Sequential, auditable wrapper around the unmodified MACBF-GNN ``test.py``.

The upstream script evaluates with ``delay_aware=False`` and therefore does not
load ``predictor.pkl``.  This wrapper preserves that behavior, isolates the
script's CSV output from the training evidence, and records a receipt.  It must
not be interpreted as evidence for the delay-aware predictor.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence

from .aoi_cbf_eval_harness import sha256_file, verify_source_checkout


AUTHOR_SEMANTICS = "author_test_py_delay_aware_false_predictor_not_loaded"


def parse_author_test_log(path: str | Path) -> dict[str, Any]:
    """Parse the single aggregate row produced by one frozen ``test.py`` call."""
    log_path = Path(path)
    with log_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) != 1 or len(rows[0]) != 8:
        raise ValueError("author test log must contain exactly one eight-column row")
    row = rows[0]
    return {
        "num_agents": int(row[0]),
        "safe_rate": float(row[1]),
        "mean_length": float(row[2]),
        "mean_error": float(row[3]),
        "episodes": int(row[4]),
        "poisson_coefficient": float(row[5]),
        "communication_radius": float(row[6]),
        "outer_seed": int(row[7]),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"receipt path already exists: {path}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_author_script_fidelity(
    *,
    source_root: Path,
    training_log_path: Path,
    output_root: Path,
    outer_seed: int,
    episodes: int,
    num_agents: int,
    checkpoint_step: int,
    gpu_index: int,
    expected_source_commit: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Run one unmodified author-script arm without mutating training evidence."""
    source_root = source_root.resolve()
    training_log_path = training_log_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"evaluation output root already exists: {output_root}")
    if not training_log_path.is_dir():
        raise FileNotFoundError(f"training log path is missing: {training_log_path}")
    if outer_seed < 0 or episodes <= 0 or num_agents <= 0:
        raise ValueError("seed must be non-negative and episodes/agents must be positive")
    if checkpoint_step <= 0 or gpu_index < 0 or timeout_seconds <= 0:
        raise ValueError("checkpoint, GPU index, and timeout must be positive or non-negative")

    source_before = verify_source_checkout(source_root, expected_source_commit)
    test_script = source_root / "test.py"
    settings = training_log_path / "settings.yaml"
    checkpoint_dir = training_log_path / "models" / f"step_{checkpoint_step}"
    checkpoint_files = {
        name: checkpoint_dir / name for name in ("actor.pkl", "cbf.pkl", "predictor.pkl")
    }
    required = [test_script, settings, *checkpoint_files.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"author evaluation inputs are missing: {missing}")

    evaluation_input = output_root / "eval_input"
    model_parent = evaluation_input / "models"
    output_root.mkdir(parents=True)
    evaluation_input.mkdir()
    model_parent.mkdir()
    shutil.copy2(settings, evaluation_input / "settings.yaml")
    os.symlink(
        checkpoint_dir,
        model_parent / f"step_{checkpoint_step}",
        target_is_directory=True,
    )

    command = [
        sys.executable,
        str(test_script),
        "--path", str(evaluation_input),
        "--epi", str(episodes),
        "-n", str(num_agents),
        "--seed", str(outer_seed),
        "--no-video",
        "--gpu", str(gpu_index),
        "--env", "SimpleCar",
        "--iter", str(checkpoint_step),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    started = time.time()
    timed_out = False
    try:
        process = subprocess.run(
            command,
            cwd=source_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        exit_code = process.returncode
        stdout = process.stdout
        stderr = process.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        exit_code = 124
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
    wall_clock_seconds = time.time() - started
    stdout_path = output_root / "stdout.log"
    stderr_path = output_root / "stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")

    errors: list[str] = []
    result: dict[str, Any] | None = None
    test_log = evaluation_input / "test_log.csv"
    if exit_code != 0:
        errors.append(f"author test.py exit code was {exit_code}")
    if not test_log.is_file():
        errors.append("author test_log.csv is missing")
    else:
        try:
            result = parse_author_test_log(test_log)
        except (ValueError, OSError) as error:
            errors.append(f"author test log parse failed: {error}")
    if result is not None and result != {
        **result,
        "num_agents": num_agents,
        "episodes": episodes,
        "outer_seed": outer_seed,
    }:
        errors.append("author test log contract fields do not match the frozen command")
    try:
        source_after = verify_source_checkout(source_root, expected_source_commit)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        source_after = {"verification_error": str(error)}
        errors.append(f"source checkout changed during evaluation: {error}")

    receipt = {
        "schema_version": 1,
        "status": "completed" if not errors else "failed_no_retry",
        "semantics": AUTHOR_SEMANTICS,
        "claim_generation_allowed": False,
        "sealed_data_used": False,
        "formal_or_g2": False,
        "source": {
            "root": str(source_root),
            "expected_commit": expected_source_commit.lower(),
            "before": source_before,
            "after": source_after,
            "test_py_sha256": sha256_file(test_script),
            "source_modified": False,
        },
        "execution": {
            "command": command,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "timeout_seconds": timeout_seconds,
            "wall_clock_seconds": wall_clock_seconds,
            "gpu_index": gpu_index,
            "sequential": True,
        },
        "evaluation": {
            "delay_aware": False,
            "predictor_loaded": False,
            "checkpoint_step": checkpoint_step,
            "result": result,
        },
        "input_sha256": {
            "settings.yaml": sha256_file(settings),
            **{name: sha256_file(path) for name, path in checkpoint_files.items()},
        },
        "output_sha256": {
            "stdout.log": sha256_file(stdout_path),
            "stderr.log": sha256_file(stderr_path),
            "test_log.csv": sha256_file(test_log) if test_log.is_file() else None,
        },
        "errors": errors,
        "limitations": [
            "The unmodified author test.py fixes delay_aware=False.",
            "predictor.pkl is hashed but is not loaded by this evaluation arm.",
            "One trained policy seed cannot establish seed-level variability.",
            "This unsealed development receipt cannot authorize confirmatory claims.",
        ],
    }
    _atomic_json(output_root / "receipt.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--training-log-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--outer-seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--checkpoint-step", type=int, default=500000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run_author_script_fidelity(
        source_root=args.source_root,
        training_log_path=args.training_log_path,
        output_root=args.output_root,
        outer_seed=args.outer_seed,
        episodes=args.episodes,
        num_agents=args.num_agents,
        checkpoint_step=args.checkpoint_step,
        gpu_index=args.gpu,
        expected_source_commit=args.expected_source_commit,
        timeout_seconds=args.timeout_seconds,
    )
    return 0 if receipt["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
