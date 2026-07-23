import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "src" / "agc_runtime_assurance" / "proposal_efficiency_g1_v3.py"
SPEC = importlib.util.spec_from_file_location("p4_proposal_efficiency_g1_v3_test", PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load V3 proposal-efficiency runner")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ProposalEfficiencyG1V3Tests(unittest.TestCase):
    def test_lossless_npz_roundtrip_preserves_full_schema(self):
        arrays = MODULE.schema_only_arrays(37)
        arrays["estimate"] = np.linspace(-0.2, 0.2, 37, dtype="<f8")
        arrays["proposal_sha256"] = np.asarray([b"a" * 64] * 37, dtype="S64")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evidence.npz"
            MODULE.write_lossless_arrays(path, arrays)
            receipt = MODULE.verify_lossless_arrays(path, 37)
            self.assertEqual(receipt["row_count"], 37)
            with np.load(path, allow_pickle=False) as loaded:
                self.assertTrue(np.array_equal(loaded["estimate"], arrays["estimate"]))
                self.assertTrue(
                    np.array_equal(loaded["proposal_sha256"], arrays["proposal_sha256"])
                )

    def test_analysis_split_removes_only_replication_records(self):
        record = {
            "method": "paired_nominal_mc",
            "family": "family",
            "replication": 0,
            "estimate": 0.1,
            "lower": 0.0,
            "upper": 0.2,
            "width": 0.2,
            "covered": True,
            "positive_lower": False,
            "ess_fraction": 1.0,
            "maximum_weight": 1.0,
            "screen_paths": 0,
            "evaluation_paths": 10,
            "simulator_calls": 20,
            "proposal_sha256": "a" * 64,
            "method_order_sha256": "b" * 64,
        }
        analysis = {
            "budget_results": [
                {
                    "total_path_budget": 10,
                    "cell_summaries": [{"mean": 0.1}],
                    "replication_records": [record],
                }
            ],
            "coverage_gate_passed": True,
        }
        arrays, summary = MODULE.arrays_from_analysis(
            analysis, ["paired_nominal_mc"], ["family"]
        )
        self.assertEqual(summary["replication_record_count"], 1)
        self.assertNotIn("replication_records", summary["budget_results"][0])
        self.assertEqual(summary["budget_results"][0]["cell_summaries"], [{"mean": 0.1}])
        self.assertAlmostEqual(float(arrays["estimate"][0]), 0.1)
        self.assertEqual(arrays["proposal_sha256"][0], b"a" * 64)

    def test_schema_preflight_uses_exact_registered_cardinality(self):
        receipt = MODULE.schema_capacity_preflight(24000, 1048576)
        self.assertEqual(receipt["record_count"], 24000)
        self.assertEqual(receipt["field_count"], len(MODULE.ARRAY_DTYPES))
        self.assertLess(receipt["projected_total_bytes"], 16777216)


if __name__ == "__main__":
    unittest.main()
