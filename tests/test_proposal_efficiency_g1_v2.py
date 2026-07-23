import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "src" / "agc_runtime_assurance" / "proposal_efficiency_g1_v2.py"
SPEC = importlib.util.spec_from_file_location("p4_proposal_efficiency_g1_v2_test", PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load v2 proposal-efficiency runner")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ProposalEfficiencyG1V2Tests(unittest.TestCase):
    def test_exact_nominal_boundary_is_supported(self):
        result = MODULE.hedged_bounded_mean_interval_allow_nominal(
            np.ones(100),
            np.array([1.0, -1.0, 0.0, 1.0, 0.0] * 20),
            rho=1.0,
            family_alpha=0.0125,
        )
        self.assertEqual(result.rho, 1.0)
        self.assertLessEqual(result.delta_lower, 0.2)
        self.assertGreaterEqual(result.delta_upper, 0.2)

    def test_nonboundary_path_remains_exactly_delegated(self):
        weights = np.array([1.2, 0.8, 1.0, 1.1, 0.9] * 20)
        differences = np.array([1.0, -1.0, 0.0, 1.0, 0.0] * 20)
        expected = MODULE.ORIGINAL_HEDGED(
            weights, differences, rho=0.5, family_alpha=0.0125
        )
        observed = MODULE.hedged_bounded_mean_interval_allow_nominal(
            weights, differences, rho=0.5, family_alpha=0.0125
        )
        self.assertEqual(observed, expected)

    def test_nominal_boundary_rejects_weights_above_one(self):
        with self.assertRaisesRegex(
            MODULE.BASE.BETTING.PairedRiskBettingError, "nominal rho=1"
        ):
            MODULE.hedged_bounded_mean_interval_allow_nominal(
                [1.01, 1.0], [1.0, 0.0], rho=1.0, family_alpha=0.0125
            )


if __name__ == "__main__":
    unittest.main()
