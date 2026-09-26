"""Deterministic conformance fixtures, not simulator or effectiveness evidence."""
from copy import deepcopy
import unittest
from evidence_gate import Registry, canonical, GATES


def fixture():
    e = dict(evidence_id='e1', subject_id='system', edge_id='edge', issuer='producer',
        model_sha256='a'*64, config_sha256='b'*64, source_sha256='c'*64, scope_id='scope',
        issued_at='2026-09-01T00:00:00Z', valid_from='2026-09-01T00:00:00Z',
        valid_until='2026-09-30T00:00:00Z', lower_bound=0.2, upper_bound=0.8,
        witness_valid=True, outcome_complete=True, interference_modeled=True)
    registry = Registry()
    digest = registry.register(e)
    request = {k: e[k] for k in ('subject_id', 'edge_id', 'scope_id', 'model_sha256', 'config_sha256', 'source_sha256')}
    request.update(minimum_lower_bound=0.1, observed_at='2026-09-12T00:00:00Z', expected_digest=digest)
    return e, registry, request


class GateTests(unittest.TestCase):
    def test_valid_and_no_operational_authority(self):
        e, r, q = fixture()
        self.assertEqual(r.evaluate(e, q), dict(status='ADMISSIBLE', operational_authorization=False))

    def test_numeric_malformed(self):
        for key in ('lower_bound', 'upper_bound'):
            for value in (float('nan'), float('inf'), -float('inf'), True, '0.2', None, 2.0, -2.0):
                with self.subTest(key=key, value=str(value)):
                    e, r, q = fixture(); e[key] = value
                    self.assertEqual(r.evaluate(e, q)['status'], 'REFUSE_INVALID_SCHEMA')

    def test_invalid_threshold(self):
        for value in (float('nan'), float('inf'), True, None, '0.1', 2):
            e, r, q = fixture(); q['minimum_lower_bound'] = value
            self.assertEqual(r.evaluate(e, q)['status'], 'REFUSE_INVALID_SCHEMA')

    def test_gate_types(self):
        for key in GATES:
            for value in ('false', 'true', 1, 0, None):
                e, r, q = fixture(); e[key] = value
                self.assertEqual(r.evaluate(e, q)['status'], 'REFUSE_INVALID_SCHEMA')

    def test_missing_and_extra_fields(self):
        e, r, q = fixture()
        for key in list(e):
            bad = deepcopy(e); del bad[key]
            self.assertEqual(r.evaluate(bad, q)['status'], 'REFUSE_INVALID_SCHEMA')
        e['extra'] = 1
        self.assertEqual(r.evaluate(e, q)['status'], 'REFUSE_INVALID_SCHEMA')

    def test_schema_order_and_timezone(self):
        for key, value in [('lower_bound', 0.9), ('issued_at', '2026-09-05T00:00:00Z'),
                           ('valid_until', '2026-08-01T00:00:00Z'), ('valid_from', 'invalid'),
                           ('valid_until', '2026-09-30T00:00:00'), ('model_sha256', 'model')]:
            e, r, q = fixture(); e[key] = value
            self.assertEqual(r.evaluate(e, q)['status'], 'REFUSE_INVALID_SCHEMA')

    def test_digest_and_source_pins(self):
        e, r, q = fixture(); e['upper_bound'] = 0.7
        self.assertEqual(r.evaluate(e, q)['status'], 'REFUSE_PROVENANCE_MISMATCH')
        e, r, q = fixture(); q['expected_digest'] = 'd'*64
        self.assertEqual(r.evaluate(e, q)['status'], 'REFUSE_PROVENANCE_MISMATCH')
        e, r, q = fixture(); q['source_sha256'] = 'd'*64
        self.assertEqual(r.evaluate(e, q)['status'], 'HOLD_SCOPE_MISMATCH')

    def test_all_false_gates(self):
        for key in GATES:
            e, _, q = fixture(); e[key] = False
            r = Registry(); q['expected_digest'] = r.register(e)
            self.assertEqual(r.evaluate(e, q)['status'], 'REFUSE_' + key.upper())

    def test_scope_fields(self):
        for key in ('subject_id', 'edge_id', 'scope_id', 'model_sha256', 'config_sha256'):
            e, r, q = fixture(); q[key] = 'd'*64
            self.assertEqual(r.evaluate(e, q)['status'], 'HOLD_SCOPE_MISMATCH')

    def test_expiry_and_inclusive_endpoints(self):
        for when, status in [('2026-09-01T00:00:00Z', 'ADMISSIBLE'), ('2026-09-30T00:00:00Z', 'ADMISSIBLE'),
                             ('2026-08-31T23:59:59Z', 'STALE_EXPIRED'), ('2026-09-30T00:00:01Z', 'STALE_EXPIRED')]:
            e, r, q = fixture(); q['observed_at'] = when
            self.assertEqual(r.evaluate(e, q)['status'], status)

    def test_no_identity_repair_or_reactivation(self):
        for state in ('STALE', 'REVOKED', 'SUPERSEDED'):
            e, r, q = fixture(); r.invalidate('e1', state)
            self.assertEqual(r.evaluate(e, q)['status'], 'REFUSE_' + state)
            with self.assertRaises(ValueError): r.register(e)
            with self.assertRaises(ValueError): r.invalidate('e1', 'ACTIVE')
            self.assertEqual(len(r.history), 2)
            e['evidence_id'] = 'e2'; q['expected_digest'] = r.register(e)
            self.assertEqual(r.evaluate(e, q)['status'], 'ADMISSIBLE')
        e, r, q = fixture(); e['upper_bound'] = 0.7
        with self.assertRaises(ValueError): r.register(e)

    def test_no_implicit_registration(self):
        e, _, q = fixture()
        self.assertEqual(Registry().evaluate(e, q)['status'], 'REFUSE_UNREGISTERED')

    def test_history_and_digest_determinism(self):
        e, r, q = fixture()
        self.assertEqual(canonical(e), canonical(dict(reversed(list(e.items())))))
        r.register(e)
        self.assertEqual(len(r.history), 1)
        self.assertIsInstance(r.history, tuple)

    def test_claim_contraction_grid(self):
        # 21 baseline intervals; all 126 ordered widening pairs; 5 thresholds.
        grid = [-1.0, -0.5, 0.0, 0.25, 0.5, 1.0]
        pairs = 0
        for lo in grid:
            for hi in grid:
                if lo > hi: continue
                for wider_lo in grid:
                    for wider_hi in grid:
                        if not wider_lo <= lo <= hi <= wider_hi: continue
                        for threshold in [-0.75, -0.25, 0, 0.25, 0.75]:
                            statuses = []
                            for lower, upper in [(lo, hi), (wider_lo, wider_hi)]:
                                e, _, q = fixture(); e.update(lower_bound=lower, upper_bound=upper)
                                r = Registry(); q['expected_digest'] = r.register(e); q['minimum_lower_bound'] = threshold
                                statuses.append(r.evaluate(e, q)['status'] == 'ADMISSIBLE')
                            self.assertFalse(statuses[1] and not statuses[0])
                            pairs += 1
        self.assertEqual(pairs, 630)


if __name__ == '__main__':
    unittest.main(verbosity=2)
