"""Frozen point-prediction interface for team first-violation horizons."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json

import numpy as np

from .environment import AirGroundRuntimeEnv, CompoundShift


class NominalRolloutHorizonPredictor:
    """Roll out a fixed nominal model under a repeated candidate joint action.

    This deliberately simple predictor is a shared G0 baseline. It is not the
    proposed novelty and must be kept fixed across assurance comparisons.
    """

    def __init__(self, model_shift: CompoundShift, *, max_steps: int = 100):
        model_shift.validate()
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.model_shift = model_shift
        self.max_steps = int(max_steps)

    def predict(
        self,
        observed_state: np.ndarray,
        candidate_action: np.ndarray,
        *,
        previous_applied_action: np.ndarray | None = None,
        step_index: int = 0,
    ) -> float:
        state = np.asarray(observed_state, dtype=float)
        action = np.asarray(candidate_action, dtype=float)
        previous = (
            np.zeros(5, dtype=float)
            if previous_applied_action is None
            else np.asarray(previous_applied_action, dtype=float)
        )
        if state.shape != (10,) or action.shape != (5,) or previous.shape != (5,):
            raise ValueError("state/action dimensions must be 10/5/5")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
            raise ValueError("predictor inputs must be finite")
        if step_index < 0:
            raise ValueError("step_index must be non-negative")

        env = AirGroundRuntimeEnv(self.model_shift, horizon=step_index + self.max_steps + 1)
        env.state = state.copy()
        env._applied_action = previous.copy()
        env.step_index = int(step_index)
        for offset in range(1, self.max_steps + 1):
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                return float(offset * env.dt)
        return float(self.max_steps * env.dt)

    def fingerprint(self) -> str:
        payload = {
            "type": type(self).__name__, "shift": asdict(self.model_shift),
            "max_steps": self.max_steps,
        }
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha256(body.encode()).hexdigest()
