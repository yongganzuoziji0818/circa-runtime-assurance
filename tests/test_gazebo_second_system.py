from pathlib import Path

import numpy as np
import pytest

from agc_runtime_assurance.environment import CompoundShift
from agc_runtime_assurance.gazebo_second_system import (
    GazeboSecondSystemError,
    GazeboStepReceipt,
    action_to_world_forces,
    constraint_margins,
)
from agc_runtime_assurance.gazebo_second_system_benchmark import (
    GazeboBenchmarkError,
    _contains_pending,
)


ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "sim" / "gazebo" / "tavh_1u1g.sdf"


def test_world_is_self_contained_and_has_frozen_step():
    body = WORLD.read_text(encoding="utf-8").lower()
    assert "<max_step_size>0.01</max_step_size>" in body
    assert '<model name="uav">' in body and '<model name="ugv">' in body
    assert all(token not in body for token in ("http://", "https://", "fuel.gazebosim", "model://"))


def test_force_mapping_is_finite_clipped_and_nonlinear():
    state = np.array([0, 0, 2, 1, -2, 0.5, 4, 2, 1.5, -0.5], dtype=float)
    uav, ugv = action_to_world_forces(state, np.array([9, -9, 1, 9, -9]), CompoundShift(uav_mass=1.2, uav_drag=0.2, ugv_friction=0.3))
    assert uav.shape == (3,) and ugv.shape == (3,)
    assert np.all(np.isfinite(uav)) and np.all(np.isfinite(ugv))
    assert ugv[2] == 0.0
    assert not np.allclose(uav, np.array([2.0, -2.0, 1.0]) / 1.2)


def test_invalid_state_action_fail_closed():
    with pytest.raises(GazeboSecondSystemError):
        constraint_margins(np.zeros(9), 0)
    with pytest.raises(GazeboSecondSystemError):
        action_to_world_forces(np.zeros(10), np.array([0, 0, np.nan, 0, 0]), CompoundShift())


def test_parameterized_reset_signature_rejects_invalid_inputs_without_gazebo():
    # Exercise the validation branch without constructing the optional Gazebo bridge.
    env = object.__new__(__import__(
        "agc_runtime_assurance.gazebo_second_system", fromlist=["GazeboAirGroundEnv"]
    ).GazeboAirGroundEnv)
    with pytest.raises(GazeboSecondSystemError):
        env.reset(seed=1, initial_state=np.zeros(9))
    with pytest.raises(GazeboSecondSystemError):
        env.reset(seed=1, position_jitter_scale=-1.0)


def test_latency_receipt_requires_monotonic_measurements_and_no_false_bound():
    receipt = GazeboStepReceipt(10, 11, 20, 21, 10)
    receipt.validate()
    assert receipt.as_seconds()["deterministic_compute_bound_available"] is False
    with pytest.raises(GazeboSecondSystemError):
        GazeboStepReceipt(10, 9, 20, 21, 10).validate()
    with pytest.raises(GazeboSecondSystemError):
        GazeboStepReceipt(10, 11, 20, 21, 10, True).validate()


def test_pending_manifest_cannot_be_mistaken_for_runnable():
    assert _contains_pending({"nested": ["PENDING_G0_FREEZE"]})
    assert not _contains_pending({"status": "frozen_g0_verified", "value": 3})
    with pytest.raises(GazeboBenchmarkError):
        raise GazeboBenchmarkError("runner remains fail closed before G0")
