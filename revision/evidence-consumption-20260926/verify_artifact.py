"""Verify committed inputs, preserved outputs and deterministic decisions.

Does not run timing measurements, simulations, networks or scientific attempts.
"""
from pathlib import Path
import collections
import hashlib
import json
import statistics
import run_evaluation as runner

ROOT = Path(__file__).resolve().parent


def main():
    commitment_raw = (ROOT / 'COMMITMENT.json').read_bytes()
    commitment = json.loads(commitment_raw)
    for row in commitment['sources']:
        b = (ROOT / row['path']).read_bytes()
        assert len(b) == row['bytes'], row['path']
        assert hashlib.sha256(b).hexdigest() == row['sha256'], row['path']
    paths = sorted((ROOT / 'cases').glob('*.json'))
    assert len(paths) == 44
    cases = [json.loads(p.read_bytes()) for p in paths]
    assert len({c['case_id'] for c in cases}) == 44
    supported = [c for c in cases if c['scope'] == 'supported_contract']
    assert len(supported) == 42
    assert sum(c['expected_admissible'] is True for c in supported) == 10
    assert sum(c['expected_admissible'] is False for c in supported) == 32
    preserved = json.loads((ROOT / 'results/decisions.json').read_bytes())
    replay = [runner.workflow(p, a) for p in paths for a in runner.ARMS]
    assert replay == preserved, 'deterministic decision mismatch'
    assert all(r['decision']['operational_authorization'] is False for r in replay)
    times = json.loads((ROOT / 'results/timings.json').read_bytes())
    assert len(times) == 5500
    counts = collections.Counter((t['block'], t['case'], t['arm']) for t in times)
    assert len(counts) == 5500 and set(counts.values()) == {1}
    summary = json.loads((ROOT / 'results/summary.json').read_bytes())
    assert summary['commitment_sha256'] == hashlib.sha256(commitment_raw).hexdigest()
    for row in summary['arms']:
        chosen = [r for r in preserved if r['arm'] == row['arm'] and r['scope'] == 'supported_contract']
        values = [t['milliseconds'] for t in times if t['arm'] == row['arm']]
        assert row['mistaken_acceptances'] == sum(r['expected'] is False and r['decision']['admissible'] for r in chosen)
        assert row['unwarranted_refusals'] == sum(r['expected'] is True and not r['decision']['admissible'] for r in chosen)
        assert row['exceptions'] == sum(r['exception'] is not None for r in chosen)
        assert row['median_ms'] == statistics.median(values)
        assert row['p95_ms'] == sorted(values)[int(.95 * (len(values) - 1))]
    print(json.dumps({'status': 'PASS_ARTIFACT_IDENTITY_AND_DETERMINISTIC_REPLAY',
        'committed_sources': len(commitment['sources']), 'decisions_replayed': len(replay),
        'timing_rows_recomputed_not_remeasured': len(times),
        'scientific_runs': 0, 'network_actions': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
