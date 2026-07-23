import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "agc_runtime_assurance" / "proposal_efficiency_g1.py"
SPEC = importlib.util.spec_from_file_location("p4_proposal_efficiency_g1_isolated", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load proposal-efficiency G1 runner")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ProposalEfficiencyG1Tests(unittest.TestCase):
    def test_bootstrap_upper_is_deterministic_and_above_geometric_mean(self):
        values = np.linspace(-0.4, -0.1, 100)
        first = MODULE._bootstrap_upper(
            values,
            confidence=0.9875,
            resamples=1000,
            rng=np.random.default_rng(7),
        )
        second = MODULE._bootstrap_upper(
            values,
            confidence=0.9875,
            resamples=1000,
            rng=np.random.default_rng(7),
        )
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, float(np.exp(np.mean(values))))
        self.assertLess(first, 1.0)

    def test_launch_receipt_must_bind_manifest_and_exact_command(self):
        raw = b'{"frozen":true}\n'
        receipt = {
            "authorization_id": "p4-proposal-efficiency-g1-v1-one-shot-launch",
            "authorized": True,
            "retry_allowed": False,
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "manifest_path": "experiments/manifests/proposal_efficiency_g1_v1.json",
            "authorized_command": (
                "C:/Users/liaoy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe "
                "src/agc_runtime_assurance/proposal_efficiency_g1.py "
                "--manifest experiments/manifests/proposal_efficiency_g1_v1.json "
                "--authorization experiments/manifests/proposal_efficiency_g1_v1_launch_authorization.json "
                "--repo-root . --output experiments/results/proposal_efficiency_g1_v1/result.json"
            ),
        }
        MODULE.validate_launch_authorization(
            receipt, raw, Path("experiments/manifests/proposal_efficiency_g1_v1.json")
        )
        receipt["retry_allowed"] = True
        with self.assertRaisesRegex(MODULE.ProposalEfficiencyG1Error, "permits retry"):
            MODULE.validate_launch_authorization(
                receipt, raw, Path("experiments/manifests/proposal_efficiency_g1_v1.json")
            )

    def test_invalid_bootstrap_input_fails_closed(self):
        with self.assertRaisesRegex(MODULE.ProposalEfficiencyG1Error, "invalid"):
            MODULE._bootstrap_upper(
                np.ones(5),
                confidence=0.95,
                resamples=100,
                rng=np.random.default_rng(1),
            )


if __name__ == "__main__":
    unittest.main()
