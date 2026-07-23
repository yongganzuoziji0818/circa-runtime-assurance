import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/agc_runtime_assurance"))

from circa import (  # noqa: E402
    INEVITABLE_VIOLATION,
    NOMINAL_SAFETY,
    NO_WITNESS,
    OBSERVED_CONSISTENCY,
    clopper_pearson_lower,
    clopper_pearson_upper,
    hoeffding_radius,
    manski_bounds,
    structured_bounds,
    verify_evidence_contract,
)


class CircaCoreTests(unittest.TestCase):
    def test_structured_bounds_pin_only_valid_witnesses(self):
        r = [0, 1, 1, 1]
        y1 = [1, 0, 0, 0]
        w = [OBSERVED_CONSISTENCY, INEVITABLE_VIOLATION, NOMINAL_SAFETY, NO_WITNESS]
        lm, um = manski_bounds(r, y1)
        lower, upper = structured_bounds(r, y1, w)
        np.testing.assert_array_equal(lm, [1, 0, 0, 0])
        np.testing.assert_array_equal(um, [1, 1, 1, 1])
        np.testing.assert_array_equal(lower, [1, 1, 0, 0])
        np.testing.assert_array_equal(upper, [1, 1, 0, 1])

    def test_invalid_witness_fails_closed(self):
        with self.assertRaises(ValueError):
            structured_bounds([0], [0], [NO_WITNESS])

    def test_registered_hoeffding_radius(self):
        expected = math.sqrt(2.0 * math.log(8 / 0.05) / 1000)
        self.assertAlmostEqual(hoeffding_radius(1000, 0.05), expected, places=15)

    def test_clopper_pearson_reference_values(self):
        self.assertAlmostEqual(clopper_pearson_lower(4750, 5000), 0.944632, places=6)
        self.assertAlmostEqual(clopper_pearson_upper(250, 5000), 0.055368, places=6)

    def test_fail_closed_priority_and_codes(self):
        self.assertEqual(verify_evidence_contract(interference_registered=False).status, "INTERFERENCE_UNMODELED")
        self.assertEqual(verify_evidence_contract(outcome_complete=False).status, "OUTCOME_CENSORED_INVALIDLY")
        self.assertEqual(verify_evidence_contract(witness_hash_matches=False).status, "INVALID_WITNESS")
        self.assertEqual(verify_evidence_contract(policy_hash_matches=False).status, "INVALID_PROVENANCE")
        self.assertTrue(verify_evidence_contract().numeric_certificate_allowed)


if __name__ == "__main__":
    unittest.main()
