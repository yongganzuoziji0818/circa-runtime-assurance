"""Prepare fresh, locally committed software cases; does not evaluate arms."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent
BASE = PACKAGE.parent / 'CIRCA_IST_revision_20260926_r1'

def raw(v):
    return (json.dumps(v, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(',', ':'))+'\n').encode()

def digest(e):
    return hashlib.sha256(json.dumps(e,sort_keys=True,ensure_ascii=True,allow_nan=False,separators=(',', ':')).encode()).hexdigest()

def fresh(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(data)

def base():
    e = dict(evidence_id='e-case',subject_id='robot-controller',edge_id='assurance-edge',issuer='trusted-producer',
        model_sha256='a'*64,config_sha256='b'*64,source_sha256='c'*64,scope_id='binary-first-violation',
        issued_at='2026-09-01T00:00:00Z',valid_from='2026-09-01T00:00:00Z',valid_until='2026-09-30T00:00:00Z',
        lower_bound=0.2,upper_bound=0.8,witness_valid=True,outcome_complete=True,interference_modeled=True)
    q = {k:e[k] for k in ('subject_id','edge_id','scope_id','model_sha256','config_sha256','source_sha256')}
    q.update(minimum_lower_bound=0.1,observed_at='2026-09-12T00:00:00Z',expected_digest=digest(e))
    return dict(object=e,request=q,registered=[deepcopy(e)],events=[])

def repin(c):
    c['request']['expected_digest']=digest(c['object'])
    c['registered']=[deepcopy(c['object'])]

def main():
    for name in ('evidence_gate.py','test_evidence_gate.py','example_consumer.py'):
        fresh(ROOT/name,(BASE/'code'/name).read_bytes())
    cases=[]
    def add(name,group,expected,edit=None,boundary=None):
        c=base()
        if edit: edit(c)
        c.update(case_id=name,family=group,expected_admissible=expected,
                 scope='external_boundary' if boundary else 'supported_contract',boundary=boundary)
        cases.append(c)
    add('valid','legitimate',True)
    add('threshold_equal','legitimate',True,lambda c:c['request'].update(minimum_lower_bound=.2))
    for label,when in [('window_start','2026-09-01T00:00:00Z'),('window_end','2026-09-30T00:00:00Z'),
                       ('timezone_equivalent','2026-09-12T08:00:00+08:00')]:
        add(label,'legitimate',True,lambda c,w=when:c['request'].update(observed_at=w))
    add('key_order','legitimate',True,lambda c:c.update(object=dict(reversed(list(c['object'].items())))))
    def legitimate_refresh(c):
        c['events']=[{'id':c['object']['evidence_id'],'state':'REVOKED'}]
        c['object']['evidence_id']='fresh-identity'
        c['object']['config_sha256']='d'*64
        c['request']['config_sha256']='d'*64
        c['request']['expected_digest']=digest(c['object'])
        c['registered'].append(deepcopy(c['object']))
    add('fresh_after_revocation','legitimate',True,legitimate_refresh)
    add('repeat_identical_registration','legitimate',True,lambda c:c['registered'].append(deepcopy(c['object'])))
    def negative_supported(c):
        c['object'].update(lower_bound=-.4,upper_bound=-.1)
        c['request']['minimum_lower_bound']=-.5
        repin(c)
    add('negative_interval_negative_threshold','legitimate',True,negative_supported)
    add('unicode_identity','legitimate',True,lambda c:(c['object'].update(subject_id='控制器α'),c['request'].update(subject_id='控制器α'),repin(c)))
    defects=[('string_bound','lower_bound','0.2'),('boolean_bound','lower_bound',True),
             ('out_of_range','upper_bound',2),('reversed_interval','lower_bound',.9),
             ('string_boolean','witness_valid','true'),('empty_issuer','issuer',''),
             ('bad_hash','model_sha256','not-a-hash'),('naive_timestamp','valid_until','2026-09-30T00:00:00'),
             ('time_order','issued_at','2026-09-15T00:00:00Z')]
    for name,key,value in defects:
        add(name,'schema',False,lambda c,k=key,v=value:c['object'].update({k:v}))
    add('missing_object_field','schema',False,lambda c:c['object'].pop('edge_id'))
    add('extra_object_field','schema',False,lambda c:c['object'].update(extra=1))
    add('missing_request_field','schema',False,lambda c:c['request'].pop('edge_id'))
    add('extra_request_field','schema',False,lambda c:c['request'].update(extra=1))
    add('boolean_threshold','schema',False,lambda c:c['request'].update(minimum_lower_bound=True))
    for key in ('subject_id','edge_id','scope_id','model_sha256','config_sha256','source_sha256'):
        add('changed_'+key,'scope',False,lambda c,k=key:c['request'].update({k:'d'*64}))
    add('mutated_payload','provenance',False,lambda c:c['object'].update(upper_bound=.7))
    add('wrong_pin','provenance',False,lambda c:c['request'].update(expected_digest='d'*64))
    for key in ('witness_valid','outcome_complete','interference_modeled'):
        add('false_'+key,'premise',False,lambda c,k=key:(c['object'].update({k:False}),repin(c)))
    for label,when in [('not_yet_valid','2026-08-31T23:59:59Z'),('expired','2026-09-30T00:00:01Z')]:
        add(label,'time',False,lambda c,w=when:c['request'].update(observed_at=w))
    add('insufficient_bound','threshold',False,lambda c:c['request'].update(minimum_lower_bound=.3))
    add('unregistered','lifecycle',False,lambda c:c.update(registered=[]))
    for state in ('STALE','REVOKED','SUPERSEDED'):
        add(state.lower(),'lifecycle',False,lambda c,s=state:c['events'].append({'id':c['object']['evidence_id'],'state':s}))
    add('forged_trusted_producer','external',None,None,'Producer supplied false but well-formed scientific premises; no consumer authenticates their truth.')
    add('restart_lost_revocation','external',None,None,'Input emulates a restarted empty in-memory service re-registering a formerly revoked object; no durable history is supplied.')
    for c in cases:
        fresh(ROOT/'cases'/(c['case_id']+'.json'),raw(c))
    fresh(ROOT/'CASES_INDEX.json',raw([{'id':c['case_id'],'family':c['family'],'expected':c['expected_admissible'],'scope':c['scope']} for c in cases]))
    sources=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            b=p.read_bytes(); sources.append({'path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
    import datetime
    fresh(ROOT/'COMMITMENT.json',raw({'committed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status':'LOCAL_PRECOMMITMENT_BEFORE_NEW_EVALUATION_NOT_PUBLIC_PREREGISTRATION',
        'cases':len(cases),'supported':sum(c['scope']=='supported_contract' for c in cases),'sources':sources}))
    print(json.dumps({'cases':len(cases),'supported':len(cases)-2,'committed':True}))

if __name__=='__main__':
    main()
