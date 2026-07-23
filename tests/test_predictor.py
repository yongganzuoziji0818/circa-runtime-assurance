import numpy as np

from agc_runtime_assurance.environment import CompoundShift
from agc_runtime_assurance.predictor import NominalRolloutHorizonPredictor


def _safe_state():
    return np.array([-4.0, -2.0, 2.0, 0.0, 0.0, 0.0, 4.0, 2.0, 0.0, 0.0])


def test_nominal_rollout_predictor_is_deterministic_and_versioned():
    predictor = NominalRolloutHorizonPredictor(CompoundShift(), max_steps=20)
    action = np.zeros(5)
    a = predictor.predict(_safe_state(), action)
    b = predictor.predict(_safe_state(), action)
    assert a == b == 2.0
    assert len(predictor.fingerprint()) == 64


def test_predictor_returns_first_team_constraint_failure():
    predictor = NominalRolloutHorizonPredictor(CompoundShift(), max_steps=30)
    state = _safe_state()
    state[0] = 9.9
    horizon = predictor.predict(state, np.array([2.0, 0.0, 0.0, 0.0, 0.0]))
    assert 0.0 < horizon < 3.0


def test_model_mismatch_changes_predicted_horizon_but_not_interface():
    state = _safe_state()
    state[0] = 9.5
    action = np.array([2.0, 0.0, 0.0, 0.0, 0.0])
    nominal = NominalRolloutHorizonPredictor(CompoundShift(uav_mass=1.0), max_steps=30)
    heavy = NominalRolloutHorizonPredictor(CompoundShift(uav_mass=2.0), max_steps=30)
    assert heavy.predict(state, action) >= nominal.predict(state, action)
