"""One fresh local software comparison. No network or scientific simulation."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import os
import platform
import statistics
import time
import conventional
from evidence_gate import Registry

ROOT=Path(__file__).resolve().parent
ARMS=('numeric_report','typed_report','stateless_rules','stateful_rules','circa')

def fresh(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=True,allow_nan=False,sort_keys=True,indent=2)
        f.write('\n')

def workflow(path,arm):
    case=json.loads(path.read_bytes())
    exception=None
    try:
        if arm=='circa':
            r=Registry()
            for obj in case['registered']: r.register(obj)
            for event in case['events']: r.invalidate(event['id'],event['state'])
            decision=r.evaluate(case['object'],case['request'])
            decision['admissible']=decision['status']=='ADMISSIBLE'
        else:
            decision=conventional.consume(case,arm)
    except Exception as e:
        exception=type(e).__name__+': '+str(e)
        decision={'admissible':False,'status':'WORKFLOW_EXCEPTION','operational_authorization':False}
    report={'case':case['case_id'],'arm':arm,'expected':case['expected_admissible'],
            'scope':case['scope'],'family':case['family'],'decision':decision,'exception':exception}
    encoded=json.dumps(report,ensure_ascii=True,allow_nan=False,sort_keys=True,separators=(',', ':')).encode()
    return json.loads(encoded)

def main():
    commitment=json.loads((ROOT/'COMMITMENT.json').read_bytes())
    for row in commitment['sources']:
        b=(ROOT/row['path']).read_bytes()
        assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'],row['path']
    out=ROOT/'results'
    out.mkdir(exist_ok=False)
    paths=sorted((ROOT/'cases').glob('*.json'))
    rows=[workflow(p,a) for p in paths for a in ARMS]
    fresh(out/'decisions.json',rows)
    timings=[]
    for block in range(25):
        order=ARMS[block%len(ARMS):]+ARMS[:block%len(ARMS)]
        for p in paths:
            for arm in order:
                t=time.perf_counter_ns(); result=workflow(p,arm); elapsed=time.perf_counter_ns()-t
                timings.append({'block':block,'case':p.stem,'arm':arm,'milliseconds':elapsed/1e6})
    fresh(out/'timings.json',timings)
    summary=[]
    for arm in ARMS:
        selected=[r for r in rows if r['arm']==arm and r['scope']=='supported_contract']
        neg=[r for r in selected if r['expected'] is False]
        pos=[r for r in selected if r['expected'] is True]
        values=[t['milliseconds'] for t in timings if t['arm']==arm]
        summary.append({'arm':arm,'cases':len(selected),'eligible':len(pos),'ineligible':len(neg),
            'mistaken_acceptances':sum(r['decision']['admissible'] for r in neg),
            'unwarranted_refusals':sum(not r['decision']['admissible'] for r in pos),
            'exceptions':sum(r['exception'] is not None for r in selected),
            'median_ms':statistics.median(values),'p95_ms':sorted(values)[int(.95*(len(values)-1))],
            'timing_repetitions':len(values)})
    fresh(out/'summary.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor(),
        'logical_cpus':os.cpu_count(),'commitment_sha256':hashlib.sha256((ROOT/'COMMITMENT.json').read_bytes()).hexdigest(),
        'status':'COMPLETE_LOCAL_AUTHOR_CONSTRUCTED_COMPARISON_NOT_FIELD_VALIDATION',
        'arms':summary,'scientific_runs':0,'remote_actions':0,
        'external_boundary_decisions':[r for r in rows if r['scope']=='external_boundary']})
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
