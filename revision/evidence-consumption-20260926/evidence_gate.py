"""Revision-only evidence consumer. No controller, simulator, network or persistence.

The caller must authenticate the producer, pin the expected digest and serialize
registry access externally. An in-memory history is NOT a durable audit log.
"""
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import re


FIELDS = frozenset(('evidence_id', 'subject_id', 'edge_id', 'issuer',
    'model_sha256', 'config_sha256', 'source_sha256', 'scope_id', 'issued_at',
    'valid_from', 'valid_until', 'lower_bound', 'upper_bound', 'witness_valid',
    'outcome_complete', 'interference_modeled'))
HASH_FIELDS = ('model_sha256', 'config_sha256', 'source_sha256')
GATES = ('witness_valid', 'outcome_complete', 'interference_modeled')
IDENTITIES = ('subject_id', 'edge_id', 'scope_id') + HASH_FIELDS
STATES = frozenset(('ACTIVE', 'STALE', 'REVOKED', 'SUPERSEDED'))


def instant(value):
    if type(value) is not str:
        raise ValueError('timestamp must be text')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('timezone required')
    return parsed.astimezone(timezone.utc)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def canonical(evidence):
    """Strict schema and deterministic local JSON, not a cross-language standard."""
    e = asdict(evidence) if is_dataclass(evidence) else evidence
    if type(e) is not dict or set(e) != FIELDS:
        raise ValueError('missing or extra evidence field')
    for key in FIELDS - set(GATES) - {'lower_bound', 'upper_bound'}:
        if type(e[key]) is not str or not e[key].strip():
            raise ValueError('nonempty string required: ' + key)
    for key in HASH_FIELDS:
        if re.fullmatch(r'[0-9a-f]{64}', e[key]) is None:
            raise ValueError('invalid SHA-256 encoding')
    if any(type(e[key]) is not bool for key in GATES):
        raise ValueError('gate flags must be booleans')
    lo, hi = e['lower_bound'], e['upper_bound']
    if not finite_number(lo) or not finite_number(hi) or not -1 <= lo <= hi <= 1:
        raise ValueError('invalid risk-reduction interval')
    issued, start, end = [instant(e[k]) for k in ('issued_at', 'valid_from', 'valid_until')]
    if not issued <= start <= end:
        raise ValueError('invalid time ordering')
    return json.dumps(e, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('utf-8')


class Registry:
    """Single-process reference state; trusted callers own mutation authority."""
    def __init__(self):
        self._objects = {}
        self._history = []

    @property
    def history(self):
        return tuple(self._history)

    def register(self, evidence):
        payload = canonical(evidence)
        e = json.loads(payload)
        identity = e['evidence_id']
        digest = sha256(payload).hexdigest()
        if identity in self._objects:
            old_digest, state = self._objects[identity]
            if old_digest == digest and state == 'ACTIVE':
                return digest
            raise ValueError('identity is immutable and cannot be reactivated')
        self._objects[identity] = (digest, 'ACTIVE')
        self._history.append((identity, 'ACTIVE', digest))
        return digest

    def invalidate(self, identity, state):
        if state not in STATES - {'ACTIVE'}:
            raise ValueError('only terminal invalidation is allowed')
        digest, old_state = self._objects[identity]
        if old_state != 'ACTIVE':
            if old_state == state:
                return
            raise ValueError('terminal identities cannot transition again')
        self._objects[identity] = (digest, state)
        self._history.append((identity, state, digest))

    def evaluate(self, evidence, request):
        def result(status):
            return {'status': status, 'operational_authorization': False}
        try:
            payload = canonical(evidence)
            e = json.loads(payload)
            if type(request) is not dict:
                raise ValueError('request must be mapping')
            required = set(IDENTITIES) | {'minimum_lower_bound', 'observed_at', 'expected_digest'}
            if set(request) != required:
                raise ValueError('invalid request schema')
            if any(type(request[k]) is not str or not request[k].strip() for k in IDENTITIES):
                raise ValueError('invalid request identity')
            for key in HASH_FIELDS + ('expected_digest',):
                if type(request[key]) is not str or re.fullmatch(r'[0-9a-f]{64}', request[key]) is None:
                    raise ValueError('invalid request digest')
            threshold = request['minimum_lower_bound']
            if not finite_number(threshold) or not -1 <= threshold <= 1:
                raise ValueError('invalid request threshold')
            observed = instant(request['observed_at'])
        except (ValueError, TypeError, OverflowError, KeyError):
            return result('REFUSE_INVALID_SCHEMA')
        identity = e['evidence_id']
        if identity not in self._objects:
            return result('REFUSE_UNREGISTERED')
        digest, state = self._objects[identity]
        if digest != sha256(payload).hexdigest() or digest != request['expected_digest']:
            return result('REFUSE_PROVENANCE_MISMATCH')
        if state != 'ACTIVE':
            return result('REFUSE_' + state)
        if any(e[key] != request[key] for key in IDENTITIES):
            return result('HOLD_SCOPE_MISMATCH')
        for key in GATES:
            if not e[key]:
                return result('REFUSE_' + key.upper())
        if not instant(e['valid_from']) <= observed <= instant(e['valid_until']):
            return result('STALE_EXPIRED')
        if e['lower_bound'] < threshold:
            return result('HOLD_BELOW_REQUIRED_BOUND')
        return result('ADMISSIBLE')
