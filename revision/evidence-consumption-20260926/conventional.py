"""Independent conventional rule comparator, no imports from CIRCA."""
import datetime as dt
import hashlib
import json
import math
import re

IDENTITY = ('subject_id', 'edge_id', 'scope_id', 'model_sha256', 'config_sha256', 'source_sha256')
HASHES = ('model_sha256', 'config_sha256', 'source_sha256')
FLAGS = ('witness_valid', 'outcome_complete', 'interference_modeled')
TIMES = ('issued_at', 'valid_from', 'valid_until')
FIELDS = set(IDENTITY + FLAGS + TIMES + ('evidence_id', 'issuer', 'lower_bound', 'upper_bound'))
REQUEST = set(IDENTITY + ('minimum_lower_bound', 'observed_at', 'expected_digest'))

def timestamp(value):
    if type(value) is not str:
        raise ValueError('timestamp type')
    t = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if t.tzinfo is None:
        raise ValueError('timezone')
    return t

def number(value):
    return type(value) in (int, float) and math.isfinite(value)

def digest(e):
    return hashlib.sha256(json.dumps(e, ensure_ascii=True, allow_nan=False,
        sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()

def schema(e, q):
    if type(e) is not dict or set(e) != FIELDS or type(q) is not dict or set(q) != REQUEST:
        return False
    for key in FIELDS - set(FLAGS) - {'lower_bound', 'upper_bound'}:
        if type(e[key]) is not str or not e[key].strip():
            return False
    for key in IDENTITY:
        if type(q[key]) is not str or not q[key].strip():
            return False
    for value in [e[k] for k in HASHES] + [q[k] for k in HASHES + ('expected_digest',)]:
        if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
            return False
    if any(type(e[k]) is not bool for k in FLAGS):
        return False
    lo, hi, threshold = e['lower_bound'], e['upper_bound'], q['minimum_lower_bound']
    if not all(number(x) for x in (lo, hi, threshold)) or not -1 <= lo <= hi <= 1 or not -1 <= threshold <= 1:
        return False
    try:
        i, s, end = (timestamp(e[k]) for k in TIMES)
        timestamp(q['observed_at'])
        return i <= s <= end
    except (ValueError, TypeError, OverflowError):
        return False

def consume(case, arm):
    e, q = case['object'], case['request']
    def out(accepted, status):
        return {'admissible': bool(accepted), 'status': status, 'operational_authorization': False}
    if arm == 'numeric_report':
        lo, t = e.get('lower_bound'), q.get('minimum_lower_bound')
        yes = number(lo) and number(t) and lo >= t
        return out(yes, 'NUMERIC_PASS' if yes else 'NUMERIC_REFUSE')
    if not schema(e, q):
        return out(False, 'SCHEMA_REFUSE')
    if arm == 'typed_report':
        return out(e['lower_bound'] >= q['minimum_lower_bound'], 'TYPED_NUMERIC')
    if digest(e) != q['expected_digest']:
        return out(False, 'PIN_REFUSE')
    if any(e[k] != q[k] for k in IDENTITY):
        return out(False, 'SCOPE_REFUSE')
    if not all(e[k] for k in FLAGS):
        return out(False, 'PREMISE_REFUSE')
    if not timestamp(e['valid_from']) <= timestamp(q['observed_at']) <= timestamp(e['valid_until']):
        return out(False, 'TIME_REFUSE')
    if e['lower_bound'] < q['minimum_lower_bound']:
        return out(False, 'BOUND_REFUSE')
    if arm == 'stateful_rules':
        registry = {}
        for obj in case['registered']:
            eid, pin = obj['evidence_id'], digest(obj)
            if eid in registry and registry[eid] != (pin, 'ACTIVE'):
                return out(False, 'REGISTER_CONFLICT')
            registry[eid] = (pin, 'ACTIVE')
        for event in case['events']:
            if event['id'] not in registry or event['state'] not in ('STALE', 'REVOKED', 'SUPERSEDED'):
                return out(False, 'INVALID_EVENT')
            pin, prior = registry[event['id']]
            if prior not in ('ACTIVE', event['state']):
                return out(False, 'INVALID_TRANSITION')
            registry[event['id']] = (pin, event['state'])
        if registry.get(e['evidence_id']) != (q['expected_digest'], 'ACTIVE'):
            return out(False, 'REGISTRY_REFUSE')
    return out(True, 'ADMISSIBLE')
