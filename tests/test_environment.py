import numpy as np

from agc_runtime_assurance.environment import AirGroundRuntimeEnv, CompoundShift


def test_deterministic_replay():
    action = np.array([0.1, 0.2, 0.0, -0.1, 0.1], dtype=np.float32)
    env_a = AirGroundRuntimeEnv(CompoundShift(actuator_lag=0.2))
    env_b = AirGroundRuntimeEnv(CompoundShift(actuator_lag=0.2))
    obs_a, _ = env_a.reset(seed=7)
    obs_b, _ = env_b.reset(seed=7)
    assert np.allclose(obs_a, obs_b)
    for _ in range(5):
        out_a = env_a.step(action)
        out_b = env_b.step(action)
        assert np.allclose(out_a[0], out_b[0])
        assert out_a[1:4] == out_b[1:4]


def test_environment_exposes_compound_shift_and_team_margins():
    env = AirGroundRuntimeEnv(CompoundShift(uav_mass=1.5, sensor_bias=0.1))
    _, info = env.reset(seed=1)
    assert info["development_only"] is True
    assert info["margins"].as_array().shape == (4,)
