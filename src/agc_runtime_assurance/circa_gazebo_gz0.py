"""Claim-ineligible CIRCA Gazebo GZ0 scene-development runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from .assurance_case import verify_assurance_case
from .dynamic_l0 import _assurance_bundle
from .environment import CompoundShift
from .gazebo_second_system import GazeboAirGroundEnv
from .predictor import NominalRolloutHorizonPredictor
from .sandbox_baselines import SandboxBaselineInfeasible, SandboxNominalCBFAdapter
from .sandbox_fallback import sandbox_backup_equilibrium, sandbox_backup_gain
from .sandbox_task import SandboxComparisonTask


class CircaGazeboGZ0Error(RuntimeError):
    pass


@dataclass(frozen=True)
class SceneCandidate:
    candidate_id: str
    family_id: str
    half_gap: float
    closing_speed: float
    lateral_offset: float
    position_jitter_scale: float
    observation_delay_steps: int
    communication_delay_steps: int
    uav_mass: float
    uav_drag: float
    ugv_friction: float
    actuator_lag: float
    sensor_bias: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SceneCandidate":
        item = cls(**value)
        numeric = np.asarray([
            item.half_gap, item.closing_speed, item.lateral_offset,
            item.position_jitter_scale, item.uav_mass, item.uav_drag,
            item.ugv_friction, item.actuator_lag, item.sensor_bias,
        ], dtype=float)
        if not item.candidate_id or not item.family_id or not np.all(np.isfinite(numeric)):
            raise CircaGazeboGZ0Error("candidate identifiers and parameters must be finite")
        if item.half_gap <= 0.5 or item.closing_speed < 0.0 or item.position_jitter_scale < 0.0:
            raise CircaGazeboGZ0Error("invalid gap, speed, or jitter")
        if item.observation_delay_steps < 0 or item.communication_delay_steps < 0:
            raise CircaGazeboGZ0Error("delay steps must be non-negative")
        item.shift().validate()
        return item

    def shift(self) -> CompoundShift:
        return CompoundShift(
            uav_mass=self.uav_mass,
            uav_drag=self.uav_drag,
            ugv_friction=self.ugv_friction,
            actuator_lag=self.actuator_lag,
            sensor_bias=self.sensor_bias,
        )

    def initial_state(self) -> np.ndarray:
        return np.array([
            -self.half_gap, self.lateral_offset, 2.0,
            self.closing_speed, 0.0, 0.0,
            self.half_gap, -self.lateral_offset,
            -self.closing_speed, 0.0,
        ], dtype=float)


@dataclass(frozen=True)
class RegimeSummary:
    candidate_id: str
    family_id: str
    scenario_seed: int
    regime: str
    first_violation: int
    first_violation_step: int | None
    intervened: int
    intervention_steps: int
    completed_steps: int


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest(manifest: dict[str, Any], root: Path) -> tuple[Path, tuple[SceneCandidate, ...]]:
    required = {
        "stage": "claim_ineligible_gazebo_gz0_scene_development",
        "status": "authorized_exactly_once",
        "authorized": True,
        "scientific_claim_allowed": False,
        "circa_gz1_authorized": False,
        "retry_allowed": False,
        "sealed_data_authorized": False,
        "gpu_count": 0,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise CircaGazeboGZ0Error(f"manifest {key} must equal {expected!r}")
    world = (root / manifest["world_path"]).resolve()
    if root not in world.parents or not world.is_file() or _sha256(world) != manifest["world_sha256"]:
        raise CircaGazeboGZ0Error("world source lock failed")
    for relative, expected in manifest.get("source_files", {}).items():
        source = (root / relative).resolve()
        if root not in source.parents or not source.is_file() or _sha256(source) != expected:
            raise CircaGazeboGZ0Error(f"source lock failed for {relative}")
    candidates = tuple(SceneCandidate.from_dict(value) for value in manifest["candidates"])
    if not candidates or len({item.candidate_id for item in candidates}) != len(candidates):
        raise CircaGazeboGZ0Error("candidate IDs must be non-empty and unique")
    families = tuple(manifest["families"])
    if set(item.family_id for item in candidates) != set(families):
        raise CircaGazeboGZ0Error("every registered family must have candidates")
    per_family = int(manifest["candidates_per_family"])
    if any(sum(item.family_id == family for item in candidates) != per_family for family in families):
        raise CircaGazeboGZ0Error("candidate count differs across families")
    if manifest["seeds_per_candidate"] <= 0 or manifest["horizon_steps"] <= 0:
        raise CircaGazeboGZ0Error("seed and horizon counts must be positive")
    expected = len(candidates) * int(manifest["seeds_per_candidate"]) * 2
    if expected != manifest["max_regime_rollouts"]:
        raise CircaGazeboGZ0Error("rollout budget does not match the frozen grid")
    return world, candidates


def _run_regime(candidate: SceneCandidate, seed: int, regime: str, world: Path, horizon: int) -> RegimeSummary:
    if regime not in {"active_assurance", "shadow_no_override"}:
        raise CircaGazeboGZ0Error("unknown regime")
    env = GazeboAirGroundEnv(world, candidate.shift(), horizon=horizon)
    env.reset(
        seed=seed,
        initial_state=candidate.initial_state(),
        position_jitter_scale=candidate.position_jitter_scale,
    )
    task = SandboxComparisonTask()
    cbf = SandboxNominalCBFAdapter(task)
    backup_center, backup_gain = sandbox_backup_equilibrium(), sandbox_backup_gain()
    predictor = NominalRolloutHorizonPredictor(CompoundShift(), max_steps=40)
    states, applied = [env.state.copy()], [env._applied_action.copy()]
    intervened = intervention_steps = 0
    first_violation_step: int | None = None
    for step in range(horizon):
        index = max(0, len(states) - 1 - candidate.observation_delay_steps)
        observed = states[index].copy()
        observed[[0, 1, 2, 6, 7]] += candidate.sensor_bias
        augmented = np.concatenate([observed, applied[index]])
        nominal = task.nominal_action(augmented)
        true_augmented = np.concatenate([env.state, env._applied_action])
        backup = np.clip(backup_gain @ (true_augmented - backup_center), task.action_lower, task.action_upper)
        try:
            filtered = cbf.decide(augmented, fallback_action=backup).action
        except (SandboxBaselineInfeasible, ValueError):
            filtered = backup.copy()
        age = (candidate.observation_delay_steps + candidate.communication_delay_steps) * env.dt + 0.04
        predicted = predictor.predict(observed, filtered, previous_applied_action=applied[index], step_index=step)
        duration = max(0.0, predicted - 0.30 - age - 0.05)
        bundle = _assurance_bundle(filtered, step * env.dt, duration, f"circa-gz0-{candidate.candidate_id}-{seed}-{step}")
        verification = verify_assurance_case(bundle)
        active_action = filtered if duration > 0.0 and verification.accepted else backup
        action = active_action if regime == "active_assurance" else nominal
        changed = not np.allclose(active_action, nominal)
        intervened |= int(changed)
        intervention_steps += int(changed)
        _, _, terminated, truncated, info = env.step(action)
        states.append(env.state.copy())
        applied.append(env._applied_action.copy())
        if info["constraint_violation"] and first_violation_step is None:
            first_violation_step = step + 1
        if terminated or truncated:
            break
    return RegimeSummary(
        candidate.candidate_id, candidate.family_id, seed, regime,
        int(first_violation_step is not None), first_violation_step,
        int(intervened), intervention_steps, env.step_index,
    )


def _select_candidates(rows: list[RegimeSummary], families: tuple[str, ...], target: float) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for family in families:
        ids = sorted({row.candidate_id for row in rows if row.family_id == family})
        scored = []
        for candidate_id in ids:
            subset = [row for row in rows if row.candidate_id == candidate_id]
            active = [row for row in subset if row.regime == "active_assurance"]
            shadow = [row for row in subset if row.regime == "shadow_no_override"]
            shadow_rate = float(np.mean([row.first_violation for row in shadow]))
            active_rate = float(np.mean([row.first_violation for row in active]))
            intervention_rate = float(np.mean([row.intervened for row in active]))
            score = abs(shadow_rate - target) + max(0.0, active_rate - 0.25) + max(0.0, 0.10 - intervention_rate)
            scored.append({
                "candidate_id": candidate_id,
                "shadow_violation_rate": shadow_rate,
                "active_violation_rate": active_rate,
                "active_intervention_rate": intervention_rate,
                "development_score": score,
            })
        scored.sort(key=lambda value: (value["development_score"], value["candidate_id"]))
        selected[family] = {"selected": scored[0], "all_candidates": scored}
    return selected


def run_gz0(manifest_path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest_raw = manifest_file.read_bytes()
    manifest = json.loads(manifest_raw)
    world, candidates = validate_manifest(manifest, root)
    output = (root / manifest["output_path"]).resolve()
    if root not in output.parents or output.exists():
        raise CircaGazeboGZ0Error("exactly-once output path already exists or is unsafe")
    schedule = []
    for candidate in candidates:
        for index in range(int(manifest["seeds_per_candidate"])):
            digest = hashlib.sha256(f"{manifest['master_seed']}|{candidate.candidate_id}|{index}".encode()).digest()
            seed = int.from_bytes(digest[:8], "big") % (2**31 - 1)
            schedule.extend([(candidate, seed, "active_assurance"), (candidate, seed, "shadow_no_override")])
    random.Random(int(manifest["schedule_seed"])).shuffle(schedule)
    start = time.perf_counter()
    rows = []
    for candidate, seed, regime in schedule:
        if time.perf_counter() - start > manifest["wall_time_seconds_max"]:
            raise CircaGazeboGZ0Error("GZ0 wall-time budget exceeded")
        rows.append(_run_regime(candidate, seed, regime, world, int(manifest["horizon_steps"])))
    elapsed = time.perf_counter() - start
    result = {
        "result_id": manifest["manifest_id"],
        "claim_eligible": False,
        "circa_gz1_run": False,
        "sealed_data_used": False,
        "formal_experiment_run": False,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "world_sha256": manifest["world_sha256"],
        "rows": [asdict(row) for row in rows],
        "selection_diagnostic": _select_candidates(rows, tuple(manifest["families"]), float(manifest["target_shadow_violation_rate"])),
        "elapsed_seconds": elapsed,
        "boundary": "claim-ineligible scene development and budget measurement only",
    }
    body = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if len(body) > manifest["output_bytes_max"]:
        raise CircaGazeboGZ0Error("GZ0 output budget exceeded")
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_bytes(body)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    result = run_gz0(args.manifest, args.repo_root)
    print(json.dumps({
        "result_id": result["result_id"],
        "rows": len(result["rows"]),
        "elapsed_seconds": result["elapsed_seconds"],
        "claim_eligible": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
