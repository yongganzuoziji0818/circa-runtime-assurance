"""Illustration only: static fixture dates; no statistical or runtime validation."""
from copy import deepcopy
import json
from test_evidence_gate import fixture

e, registry, q = fixture()
changed_request = deepcopy(q)
changed_request['config_sha256'] = 'd' * 64
changed_object = deepcopy(e)
changed_object['upper_bound'] = 0.7
decisions = {
    'valid': registry.evaluate(e, q),
    'changed_requested_configuration': registry.evaluate(e, changed_request),
    'changed_payload': registry.evaluate(changed_object, q),
}
registry.invalidate(e['evidence_id'], 'REVOKED')
decisions['revoked'] = registry.evaluate(e, q)
assert [v['status'] for v in decisions.values()] == [
    'ADMISSIBLE', 'HOLD_SCOPE_MISMATCH', 'REFUSE_PROVENANCE_MISMATCH', 'REFUSE_REVOKED']
assert all(v['operational_authorization'] is False for v in decisions.values())
print(json.dumps({'kind':'ILLUSTRATION_NOT_EMPIRICAL_EVIDENCE', 'object':e,
                 'request':q, 'decisions':decisions}, sort_keys=True, indent=2))
