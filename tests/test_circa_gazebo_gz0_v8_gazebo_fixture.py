from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from agc_runtime_assurance.circa_gazebo_gz0 import SceneCandidate
from agc_runtime_assurance.circa_gazebo_gz0_v3 import derive_operational_envelope
from agc_runtime_assurance.circa_gazebo_gz0_v8 import _scenario, build_filter
from agc_runtime_assurance.gazebo_second_system_v3 import GazeboAirGroundEnvV3
from agc_runtime_assurance.gazebo_second_system_v4 import GazeboAirGroundEnvV4
from agc_runtime_assurance.gazebo_timestamp_aligned_set_filter import (
    align_async_state_history,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "experiments/manifests/circa_gazebo_gz0_v8_DESIGN_ONLY.json"
WORLD = ROOT / "sim/gazebo/tavh_1u1g.sdf"


def _fixture(driver: str) -> tuple[int, list[str]]:
    manifest = json.loads(DESIGN.read_text(encoding="utf-8"))
    candidate = SceneCandidate.from_dict(manifest["candidates"][9])
    envelope = derive_operational_envelope(
        candidate, manifest["operational_envelope_assumptions"]
    )
    if driver == "planar_speed_projected_v4":
        env = GazeboAirGroundEnvV4(
            WORLD,
            candidate.shift(),
            horizon=2,
            uav_planar_speed_limit_mps=0.5 * envelope.design_relative_speed_mps,
            ugv_planar_speed_limit_mps=0.5 * envelope.design_relative_speed_mps,
        )
    else:
        env = GazeboAirGroundEnvV3(WORLD, candidate.shift(), horizon=2)
    scenario = _scenario(candidate, manifest, hazard_active=True)
    env.reset(seed=9801, initial_state=scenario.initial_state, position_jitter_scale=0.0)
    safety_filter = build_filter(candidate, manifest)
    states = [env.state.copy()]
    applied = [env._applied_action.copy()]
    reasons = []
    for _ in range(2):
        aligned = align_async_state_history(
            states,
            applied,
            candidate.observation_delay_steps,
            candidate.communication_delay_steps,
            safety_filter.plant,
            safety_filter.alignment,
            observed_common_mode_bias_m=candidate.sensor_bias,
        )
        nominal = scenario.task.nominal_action(
            np.concatenate([aligned.center, applied[-1]])
        )
        decision = safety_filter.decide(aligned, applied[-1], nominal)
        assert decision.certificate_emitted is True
        reasons.append(decision.reason)
        env.step(decision.action)
        states.append(env.state.copy())
        applied.append(env._applied_action.copy())
    return env.step_index, reasons


def test_v8_wires_both_frozen_gazebo_drivers_without_scientific_output() -> None:
    if importlib.util.find_spec("gz") is None or importlib.util.find_spec("gz.sim8") is None:
        pytest.skip("optional Gazebo Sim 8 Python bindings are unavailable locally")
    results = [_fixture(driver) for driver in (
        "command_persistent_unbounded_v3",
        "planar_speed_projected_v4",
    )]
    assert all(step_count == 2 for step_count, _ in results)
    assert all(len(reasons) == 2 for _, reasons in results)
    manifest = json.loads(DESIGN.read_text(encoding="utf-8"))
    assert not (ROOT / manifest["output_namespace_reserved"]).exists()
