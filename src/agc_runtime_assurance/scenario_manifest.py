"""Fail-closed loader for unsealed G0 baseline-compatibility scenarios."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .environment import AirGroundRuntimeEnv, CompoundShift


class ScenarioManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeTimingScenario:
    observation_age_s: float
    communication_delay_s: float
    compute_delay_s: float
    actuation_delay_s: float


@dataclass(frozen=True)
class G0Scenario:
    name: str
    reset_seed: int
    horizon: int
    shift: CompoundShift
    timing: RuntimeTimingScenario
    initial_state: np.ndarray | None
    initial_applied_action: np.ndarray | None

    def instantiate(self) -> tuple[AirGroundRuntimeEnv, np.ndarray]:
        """Instantiate only the declared development sandbox state."""
        env = AirGroundRuntimeEnv(self.shift, horizon=self.horizon)
        env.reset(seed=self.reset_seed)
        if self.initial_state is not None:
            env.state = self.initial_state.copy()
        if self.initial_applied_action is not None:
            env._applied_action = self.initial_applied_action.copy()
        return env, np.concatenate([env.state, env._applied_action])


@dataclass(frozen=True)
class G0ScenarioManifest:
    manifest_id: str
    scenarios: tuple[G0Scenario, ...]
    fingerprint: str


def load_g0_scenario_manifest(path: str | Path) -> G0ScenarioManifest:
    raw = Path(path).read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    if manifest.get("stage") != "g0_baseline_compatibility":
        raise ScenarioManifestError("only g0_baseline_compatibility manifests are accepted")
    if manifest.get("development_only") is not True:
        raise ScenarioManifestError("G0 scenarios must be marked development_only")
    if manifest.get("authorized_execution") is not False:
        raise ScenarioManifestError("scenario description must not grant execution authority")
    if manifest.get("sealed_data_referenced") is not False:
        raise ScenarioManifestError("sealed data must not be referenced")
    records = manifest.get("scenarios")
    if not isinstance(records, list) or not records:
        raise ScenarioManifestError("at least one G0 scenario is required")
    scenarios = tuple(_parse_scenario(record) for record in records)
    names = [scenario.name for scenario in scenarios]
    if len(set(names)) != len(names):
        raise ScenarioManifestError("scenario names must be unique")
    manifest_id = manifest.get("manifest_id")
    if not isinstance(manifest_id, str) or not manifest_id:
        raise ScenarioManifestError("manifest_id is missing")
    return G0ScenarioManifest(
        manifest_id, scenarios, hashlib.sha256(raw).hexdigest(),
    )


def _parse_scenario(record: Any) -> G0Scenario:
    if not isinstance(record, dict):
        raise ScenarioManifestError("scenario records must be objects")
    name = record.get("name")
    seed = record.get("reset_seed")
    horizon = record.get("horizon")
    if not isinstance(name, str) or not name:
        raise ScenarioManifestError("scenario name is missing")
    if not isinstance(seed, int):
        raise ScenarioManifestError(f"scenario {name} reset_seed must be an integer")
    if not isinstance(horizon, int) or not 1 <= horizon <= 200:
        raise ScenarioManifestError(f"scenario {name} horizon must lie in [1, 200]")
    shift_record = record.get("shift")
    if not isinstance(shift_record, dict):
        raise ScenarioManifestError(f"scenario {name} shift is missing")
    try:
        shift = CompoundShift(**shift_record)
        shift.validate()
    except (TypeError, ValueError) as error:
        raise ScenarioManifestError(f"scenario {name} has invalid shift: {error}") from error
    timing_record = record.get("runtime_timing")
    timing = _parse_timing(name, timing_record)
    state = _optional_vector(record.get("initial_state"), 10, name, "initial_state")
    applied = _optional_vector(
        record.get("initial_applied_action"), 5, name, "initial_applied_action",
    )
    if (state is None) != (applied is None):
        raise ScenarioManifestError(
            f"scenario {name} must specify both initial state and applied action or neither"
        )
    return G0Scenario(name, seed, horizon, shift, timing, state, applied)


def _parse_timing(name: str, record: Any) -> RuntimeTimingScenario:
    fields = (
        "observation_age_s", "communication_delay_s",
        "compute_delay_s", "actuation_delay_s",
    )
    if not isinstance(record, dict) or set(record) != set(fields):
        raise ScenarioManifestError(f"scenario {name} runtime timing fields are incomplete")
    values = np.asarray([record[field] for field in fields], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ScenarioManifestError(f"scenario {name} runtime timings must be non-negative")
    return RuntimeTimingScenario(*map(float, values))


def _optional_vector(
    value: Any, size: int, name: str, field: str,
) -> np.ndarray | None:
    if value is None:
        return None
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ScenarioManifestError(f"scenario {name} {field} must have {size} finite values")
    return vector
