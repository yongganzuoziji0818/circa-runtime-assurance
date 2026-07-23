import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "agc_runtime_assurance"
    / "paired_risk_reduction.py"
)
SPEC = importlib.util.spec_from_file_location("p4_paired_risk_reduction_isolated", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load paired risk-reduction module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PairedRiskReductionError = MODULE.PairedRiskReductionError
family_risk_reduction_certificate = MODULE.family_risk_reduction_certificate
importance_weights_from_log_densities = MODULE.importance_weights_from_log_densities
simultaneous_risk_reduction_certificate = MODULE.simultaneous_risk_reduction_certificate


class PairedRiskReductionTests(unittest.TestCase):
    def test_log_density_weights_enforce_defensive_bound(self):
        weights = importance_weights_from_log_densities(
            np.log([0.2, 0.4]), np.log([0.4, 0.4]), rho=0.5
        )
        self.assertTrue(np.allclose(weights, [0.5, 1.0]))
        with self.assertRaisesRegex(PairedRiskReductionError, "exceeds"):
            importance_weights_from_log_densities([0.0], [-math.log(3.0)], rho=0.5)

    def test_family_certificate_uses_signed_paired_difference_and_exact_counts(self):
        result = family_risk_reduction_certificate(
            "interaction",
            np.ones(6),
            [1, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0],
            rho=1.0,
            family_alpha=0.05,
        )
        self.assertAlmostEqual(result.estimate, 1.0 / 6.0)
        self.assertAlmostEqual(result.sample_variance, np.var([1, -1, 0, 1, 0, 0], ddof=1))
        self.assertEqual(result.baseline_only_failures, 2)
        self.assertEqual(result.full_only_failures, 1)
        self.assertEqual(result.shared_failures, 1)
        self.assertEqual(result.shared_safe, 2)
        self.assertAlmostEqual(result.generic_weight_ess, 6.0)
        self.assertLessEqual(-1.0, result.empirical_bernstein_lower)
        self.assertLessEqual(result.empirical_bernstein_lower, result.estimate)

    def test_simultaneous_certificate_refuses_missing_registered_family(self):
        samples = {"dynamics": (np.ones(3), [1, 0, 0], [0, 0, 0])}
        with self.assertRaisesRegex(PairedRiskReductionError, "exactly match"):
            simultaneous_risk_reduction_certificate(
                ["dynamics", "communication"],
                samples,
                rho=1.0,
                alpha=0.05,
                minimum_relevant_reduction=0.0,
            )

    def test_simultaneous_certificate_uses_worst_family_lower_bound(self):
        n = 5000
        samples = {
            "dynamics": (np.ones(n), np.ones(n), np.zeros(n)),
            "communication": (np.ones(n), np.ones(n), np.zeros(n)),
        }
        result = simultaneous_risk_reduction_certificate(
            ["dynamics", "communication"],
            samples,
            rho=1.0,
            alpha=0.05,
            minimum_relevant_reduction=0.9,
        )
        expected = 1.0 - 14.0 * math.log(2.0 / (0.05 / 2.0)) / (3.0 * (n - 1))
        self.assertAlmostEqual(result.empirical_bernstein_lower_min, expected)
        self.assertTrue(result.certified)

    def test_family_certificate_fails_closed_on_invalid_inputs(self):
        cases = [
            ([1.0], [1], [0], "at least two"),
            ([2.1, 1.0], [1, 0], [0, 0], "defensive-mixture"),
            ([1.0, 1.0], [2, 0], [0, 0], "binary"),
            ([1.0, float("nan")], [1, 0], [0, 0], "non-finite"),
        ]
        for weights, baseline, full, match in cases:
            with self.subTest(match=match):
                with self.assertRaisesRegex(PairedRiskReductionError, match):
                    family_risk_reduction_certificate(
                        "family",
                        weights,
                        baseline,
                        full,
                        rho=0.5,
                        family_alpha=0.05,
                    )


if __name__ == "__main__":
    unittest.main()
