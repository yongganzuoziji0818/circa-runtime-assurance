import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/agc_runtime_assurance"))

from circa_d1 import (  # noqa: E402
    METHODS,
    SCHEMA_DTYPE,
    audit_compiled_family,
    compile_family,
    schema_only_array,
    verify_lossless_arrays,
    write_lossless_arrays,
)


def micro_spec():
    return {
        "horizon": 24,
        "decision_step": 5,
        "risk_threshold": 1.0,
        "support": {
            "hazard": [4],
            "gust": [0],
            "congestion": [0],
            "sensor_error": [0],
            "delay": [0],
            "audit_coin": [1],
        },
    }


def micro_family():
    return {
        "id": "TEST_ONLY_NOT_D1",
        "hazard_probs": [1.0],
        "gust_probs": [1.0],
        "congestion_probs": [1.0],
        "sensor_probs": [1.0],
        "delay_probs": [1.0],
        "audit_coin_probs": [1.0],
        "sensor_scale": 0.0,
        "registered_coupling": 0.0,
        "shield_strength": 0.7,
        "transfer_penalty": 0.0,
        "trigger_threshold": 0.8,
        "randomized_overlap": False,
        "verifier_radius": 0.01,
    }


class CircaD1EngineeringTests(unittest.TestCase):
    def test_schema_and_lossless_round_trip(self):
        records = schema_only_array(13)
        records["family"] = b"TEST_ONLY_NOT_D1"
        records["method"] = METHODS[-1].encode("ascii")
        records["status"] = b"VALID_BOUND"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d1_arrays.npz"
            write_lossless_arrays(path, records)
            result = verify_lossless_arrays(path, records)
        self.assertEqual(result["records"], 13)
        self.assertEqual(result["dtype_itemsize"], SCHEMA_DTYPE.itemsize)

    def test_micro_dynamic_family_has_sound_witnesses(self):
        compiled = compile_family(micro_spec(), micro_family())
        audit = audit_compiled_family(compiled)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["support_points"], 1)
        self.assertEqual(float(compiled["probability"].sum()), 1.0)

    def test_conservative_radius_does_not_change_support_order(self):
        a = compile_family(micro_spec(), micro_family(), 1.0)
        b = compile_family(micro_spec(), micro_family(), 1.5)
        self.assertTrue(np.array_equal(a["probability"], b["probability"]))
        self.assertTrue(np.array_equal(a["y0"], b["y0"]))
        self.assertTrue(np.array_equal(a["y1"], b["y1"]))


if __name__ == "__main__":
    unittest.main()
