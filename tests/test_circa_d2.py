import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/agc_runtime_assurance"))

from circa_d2 import audit_compiled_family, compile_family  # noqa: E402


def micro_spec():
    return {
        "horizon": 30,
        "decision_step": 3,
        "separation_threshold": 0.35,
        "support": {
            "uav_distance": [1.25],
            "ugv_distance": [1.25],
            "uav_speed": [0.08],
            "ugv_speed": [0.08],
            "lateral_wind": [0],
            "sensor_error": [0],
            "recovery_delay": [0],
            "audit_coin": [1],
        },
    }


def micro_family():
    return {
        "id": "TEST_ONLY_D2_MICRO",
        "role": "test_only",
        "uav_distance_probs": [1.0],
        "ugv_distance_probs": [1.0],
        "uav_speed_probs": [1.0],
        "ugv_speed_probs": [1.0],
        "wind_probs": [1.0],
        "sensor_probs": [1.0],
        "delay_probs": [1.0],
        "audit_coin_probs": [1.0],
        "position_error_scale": 0.0,
        "speed_error_scale": 0.0,
        "trigger_threshold": 0.55,
        "climb_strength": 1.0,
        "brake_fraction": 0.6,
        "coordination_delay": 0,
        "randomized_overlap": False,
        "verifier_radius": 0.01,
    }


class CircaD2EngineeringTests(unittest.TestCase):
    def test_second_dynamics_micro_fixture_is_sound(self):
        compiled = compile_family(micro_spec(), micro_family())
        audit = audit_compiled_family(compiled)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["support_points"], 1)
        self.assertAlmostEqual(float(compiled["probability"].sum()), 1.0)

    def test_radius_sensitivity_preserves_exogenous_support_and_outcomes(self):
        base = compile_family(micro_spec(), micro_family(), 1.0)
        conservative = compile_family(micro_spec(), micro_family(), 1.5)
        self.assertTrue(np.array_equal(base["probability"], conservative["probability"]))
        self.assertTrue(np.array_equal(base["y0"], conservative["y0"]))
        self.assertTrue(np.array_equal(base["y1"], conservative["y1"]))

    def test_active_and_shadow_share_registered_support(self):
        compiled = compile_family(micro_spec(), micro_family())
        self.assertEqual(compiled["r"].shape, compiled["y0"].shape)
        self.assertEqual(compiled["r"].shape, compiled["y1"].shape)
        self.assertEqual(compiled["r"].shape, compiled["witness"].shape)


if __name__ == "__main__":
    unittest.main()
