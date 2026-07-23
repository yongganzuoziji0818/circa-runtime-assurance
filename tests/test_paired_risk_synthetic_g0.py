import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "agc_runtime_assurance"
    / "paired_risk_synthetic_g0.py"
)
SPEC = importlib.util.spec_from_file_location("p4_paired_risk_synthetic_g0_isolated", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load paired risk synthetic G0 module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PairedRiskSyntheticG0Tests(unittest.TestCase):
    def _manifest(self):
        return {
            "seed": 17,
            "alpha": 0.05,
            "rho": 0.5,
            "replications": 1000,
            "sample_sizes": [20, 100],
            "minimum_empirical_coverage": 0.90,
            "maximum_absolute_bias": 0.10,
            "cases": [
                {
                    "family": "positive",
                    "nominal": [0.94, 0.03, 0.01, 0.02],
                    "surrogate": [0.20, 0.50, 0.20, 0.10],
                },
                {
                    "family": "negative",
                    "nominal": [0.95, 0.005, 0.025, 0.02],
                    "surrogate": [0.10, 0.20, 0.60, 0.10],
                },
            ],
        }

    def test_simulation_is_deterministic_and_covers_signed_directions(self):
        first = MODULE.simulate_coverage(self._manifest())
        second = MODULE.simulate_coverage(self._manifest())
        self.assertEqual(first, second)
        truths = [
            family["true_risk_reduction"]
            for family in first["rows"][0]["family_results"]
        ]
        self.assertGreater(truths[0], 0.0)
        self.assertLess(truths[1], 0.0)
        self.assertTrue(first["coverage_gate_passed"])

    def test_probability_contract_rejects_zero_support_entry(self):
        manifest = self._manifest()
        manifest["cases"][0]["surrogate"][0] = 0.0
        manifest["cases"][0]["surrogate"][1] += 0.20
        with self.assertRaisesRegex(MODULE.SyntheticCoverageError, "positive"):
            MODULE.simulate_coverage(manifest)


if __name__ == "__main__":
    unittest.main()
