"""Auditable delay-aware evaluation adapter for the isolated MACBF-GNN source.

The upstream ``test.py`` fixes ``delay_aware=False``.  This adapter does not
modify or copy upstream source code: it imports the isolated package, creates
the same task with ``delay_aware=True``, loads the predictor checkpoint, and
writes an explicit P4-adaptation receipt.  Its output must never be labelled as
an unmodified author-script result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Sequence


ADAPTER_SEMANTICS = "p4_delay_aware_adaptation_not_author_test_script"
CONTROL_ADAPTER_SEMANTICS = "p4_non_delay_aware_adapter_not_author_test_script"


def derive_episode_seeds(outer_seed: int, episodes: int) -> tuple[int, ...]:
    """Match the legacy NumPy seed stream used by the upstream test entry."""
    if not isinstance(outer_seed, int) or outer_seed < 0:
        raise ValueError("outer_seed must be a non-negative integer")
    if not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    import numpy as np

    generator = np.random.RandomState(outer_seed)
    return tuple(int(value) for value in generator.randint(100000, size=episodes))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_episode_seed_artifact(
    path: str | Path,
    *,
    outer_seed: int,
    episodes: int,
) -> dict[str, Any]:
    """Verify that a frozen, unsealed artifact contains the exact seed stream."""
    artifact_path = Path(path).resolve()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    expected = list(derive_episode_seeds(outer_seed, episodes))
    if artifact.get("sealed") is not False:
        raise ValueError("episode seed artifact must be explicitly unsealed")
    if artifact.get("outer_seed") != outer_seed or artifact.get("episodes") != episodes:
        raise ValueError("episode seed artifact contract does not match the evaluation")
    if artifact.get("episode_seeds") != expected:
        raise ValueError("episode seed artifact stream does not match the frozen generator")
    return {
        "path": str(artifact_path),
        "sha256": sha256_file(artifact_path),
        "outer_seed": outer_seed,
        "episodes": episodes,
        "episode_seeds": expected,
    }


def verify_source_checkout(
    source_root: str | Path,
    expected_source_commit: str,
) -> dict[str, Any]:
    """Bind an evaluation to an exact, clean upstream Git checkout."""
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"source root is missing: {root}")
    if re.fullmatch(r"[0-9a-fA-F]{40}", expected_source_commit) is None:
        raise ValueError("expected_source_commit must be a full hexadecimal Git commit")
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Git is required for source provenance verification") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError("source provenance Git command failed") from error

    actual_commit = revision.stdout.strip().lower()
    if actual_commit != expected_source_commit.lower():
        raise RuntimeError(
            f"source commit mismatch: expected {expected_source_commit.lower()}, "
            f"observed {actual_commit}"
        )
    if status.stdout.strip():
        raise RuntimeError("source checkout is not clean")
    return {
        "actual_commit": actual_commit,
        "worktree_clean": True,
        "verification": "git_rev_parse_and_status_porcelain_v1",
    }


def _scalar(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def evaluate_delay_aware(
    *,
    source_root: Path,
    log_path: Path,
    output_path: Path,
    outer_seeds: Sequence[int],
    episodes_per_seed: int,
    num_agents: int,
    checkpoint_step: int,
    gpu_index: int,
    expected_source_commit: str,
    delay_aware: bool = True,
    seed_artifact_path: Path | None = None,
) -> dict[str, Any]:
    """Run one P4 adapter arm and atomically persist its receipt.

    ``delay_aware=False`` provides the paired control semantics.  It is not
    labelled as an unmodified-author-script result because the upstream script
    cannot preserve a precomputed episode-seed stream after its first episode.
    """
    source_root = source_root.resolve()
    log_path = log_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"evaluation receipt already exists: {output_path}")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary evaluation receipt already exists: {temporary}")
    source_provenance = verify_source_checkout(source_root, expected_source_commit)
    if not log_path.is_dir():
        raise FileNotFoundError(f"training log path is missing: {log_path}")
    if not outer_seeds or any(not isinstance(seed, int) or seed < 0 for seed in outer_seeds):
        raise ValueError("outer_seeds must contain non-negative integers")
    if len(set(outer_seeds)) != len(outer_seeds):
        raise ValueError("outer_seeds must be unique")
    if episodes_per_seed <= 0 or num_agents <= 0 or checkpoint_step <= 0:
        raise ValueError("episodes, agents, and checkpoint step must be positive")
    if gpu_index < 0:
        raise ValueError("gpu_index must be non-negative")
    seed_artifact = None
    if seed_artifact_path is not None:
        if len(outer_seeds) != 1:
            raise ValueError("a frozen seed artifact requires exactly one outer seed")
        seed_artifact = verify_episode_seed_artifact(
            seed_artifact_path,
            outer_seed=outer_seeds[0],
            episodes=episodes_per_seed,
        )
    checkpoint_dir = log_path / "models" / f"step_{checkpoint_step}"
    checkpoint_files = {
        name: checkpoint_dir / name for name in ("actor.pkl", "cbf.pkl", "predictor.pkl")
    }
    missing = [str(path) for path in checkpoint_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint files are missing: {missing}")

    sys.path.insert(0, str(source_root))
    original_cwd = Path.cwd()
    os.chdir(source_root)
    started = time.time()
    try:
        import numpy as np
        import torch

        from macbf_gnn.algo import make_algo
        from macbf_gnn.env import make_env
        from macbf_gnn.trainer.utils import eval_ctrl_epi, read_settings

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the frozen evaluation contract")
        if gpu_index >= torch.cuda.device_count():
            raise RuntimeError("requested GPU index is not visible")
        device = torch.device(f"cuda:{gpu_index}")
        settings = read_settings(str(log_path))
        environment_name = str(settings.get("env", "SimpleCar"))
        algorithm_name = str(settings.get("algo", "macbfgnn"))

        env = make_env(
            environment_name,
            num_agents,
            device,
            delay_aware=delay_aware,
        )
        env.test()
        use_all_data = True
        algo = make_algo(
            algorithm_name,
            env,
            num_agents,
            env.node_dim,
            env.edge_dim,
            env.state_dim,
            env.action_dim,
            device,
            hyperparams=settings.get("hyper_params"),
            use_all_data=use_all_data,
        )
        algo.load(str(checkpoint_dir), delay_aware=delay_aware)

        episode_records: list[dict[str, Any]] = []
        for outer_seed in outer_seeds:
            for episode_seed in derive_episode_seeds(outer_seed, episodes_per_seed):
                reward, length, error, _video, info = eval_ctrl_epi(
                    algo.act,
                    use_all_data,
                    env,
                    episode_seed,
                    False,
                    False,
                    verbose=False,
                )
                episode_records.append({
                    "outer_seed": int(outer_seed),
                    "episode_seed": int(episode_seed),
                    "reward": _scalar(reward),
                    "length": _scalar(length),
                    "error": _scalar(error),
                    "safe": bool(info.get("safe", False)),
                })

        safe_count = sum(int(record["safe"]) for record in episode_records)
        source_after = verify_source_checkout(source_root, expected_source_commit)
        receipt = {
            "schema_version": 1,
            "status": "completed",
            "semantics": (
                ADAPTER_SEMANTICS if delay_aware else CONTROL_ADAPTER_SEMANTICS
            ),
            "claim_generation_allowed": False,
            "sealed_data_used": False,
            "formal_or_g2": False,
            "source": {
                "root": str(source_root),
                "expected_commit": expected_source_commit,
                "before": source_provenance,
                "after": source_after,
                "source_modified": False,
            },
            "evaluation": {
                "environment": environment_name,
                "algorithm": algorithm_name,
                "delay_aware": delay_aware,
                "predictor_loaded": delay_aware,
                "use_all_data": use_all_data,
                "num_agents": num_agents,
                "checkpoint_step": checkpoint_step,
                "outer_seeds": list(outer_seeds),
                "episodes_per_seed": episodes_per_seed,
                "total_episodes": len(episode_records),
                "safe_episodes": safe_count,
                "safe_rate": safe_count / len(episode_records),
                "mean_reward": float(np.mean([r["reward"] for r in episode_records])),
                "mean_length": float(np.mean([r["length"] for r in episode_records])),
                "mean_error": float(np.mean([r["error"] for r in episode_records])),
            },
            "seed_artifact": seed_artifact,
            "checkpoint_sha256": {
                name: sha256_file(path) for name, path in checkpoint_files.items()
            },
            "runtime": {
                "gpu_index": gpu_index,
                "cuda_device_name": torch.cuda.get_device_name(gpu_index),
                "wall_clock_seconds": time.time() - started,
            },
            "episodes": episode_records,
            "limitations": [
                "This is a P4 evaluation adapter, not the upstream test.py result.",
                "A single trained policy seed cannot establish seed-level performance variability.",
                "Development evidence cannot generate confirmatory or sealed-data claims.",
            ],
        }
    finally:
        os.chdir(original_cwd)
        try:
            sys.path.remove(str(source_root))
        except ValueError:
            pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output_path)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--episodes-per-seed", type=int, required=True)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--checkpoint-step", type=int, default=500000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--seed-artifact", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--delay-aware", dest="delay_aware", action="store_true")
    mode.add_argument("--no-delay-aware", dest="delay_aware", action="store_false")
    parser.set_defaults(delay_aware=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluate_delay_aware(
        source_root=args.source_root.resolve(),
        log_path=args.log_path.resolve(),
        output_path=args.output.resolve(),
        outer_seeds=tuple(args.outer_seeds),
        episodes_per_seed=args.episodes_per_seed,
        num_agents=args.num_agents,
        checkpoint_step=args.checkpoint_step,
        gpu_index=args.gpu,
        expected_source_commit=args.expected_source_commit,
        delay_aware=args.delay_aware,
        seed_artifact_path=args.seed_artifact.resolve() if args.seed_artifact else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
