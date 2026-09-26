# CIRCA evidence-consumption comparison, 2026-09-26

This standard-library-only software artifact accompanies the working CIRCA IST
revision. It is a local, author-constructed conformance comparison, **not** an
external benchmark, a human study, a simulator experiment, or deployment approval.
It does not replace the historical scientific code or evidence archive.

## Inspect and reproduce

Use Python 3.11 or later (the recorded run used CPython 3.14.5 on Windows 11).
From this directory:

```text
python -B verify_artifact.py
python -B -m unittest -v test_evidence_gate
python -B example_consumer.py
```

The verifier checks the committed source/case identities, replays all 220
case/arm decisions and exactly compares them with the retained decisions. It
also recalculates every summary count and timing statistic from 5,500 raw timing
rows. It does not repeat timing measurements or write into the recorded results.
The unit suite has 14 test methods, including 630 interval/threshold combinations.
The illustrative consumer always disables operational authorization.

To repeat the timing experiment, make a **separate** copy of this directory
without its `results` subdirectory, retaining all other committed files, then run
`python -B run_evaluation.py` there. It refuses to overwrite an existing results
directory. Results from another machine are a new measurement, not replacements
for the recorded run. `prepare.py` documents original case construction from the
local r1 revision; it is not needed for replay and should not be run here.

## Design and result

`PROTOCOL.md` and `COMMITMENT.json` were saved before measurement. This is a local
timestamped commitment, not a public preregistration or independent attestation.
`CASES_INDEX.json` declares expected decisions. Forty-two supported-scope cases
comprise 10 eligible and 32 ineligible workflows. Two additional illustrative
trust/persistence boundaries have no correctness denominator.

| Author-implemented arm | Mistaken acceptance / 32 | Unwarranted refusal / 10 | Median / p95 ms |
|---|---:|---:|---:|
| Numeric report | 28 | 0 | 0.0941 / 0.1394 |
| Typed schema plus numeric report | 17 | 0 | 0.1012 / 0.1491 |
| Stateless contract rules | 4 | 0 | 0.1104 / 0.1618 |
| Stateful contract rules | 0 | 0 | 0.1142 / 0.1792 |
| CIRCA reference | 0 | 0 | 0.1404 / 0.2080 |

The strong stateful comparator is separately implemented without importing
CIRCA. **Parity**, not superiority, is the functional result. The weaker arms
are explanatory ablations, not competitors representing mature assurance tools.
All cases and all failures are retained. Timing uses 25 rotating paired blocks
over all 44 cases (1,100 observations per arm), includes JSON disk input and
in-memory serialized report creation, but excludes output-file publication.
Repeated timings are technical measurements, not independent study units.

## Explicit limitations and post-run audit notes

- All inputs and expected decisions were constructed by the authors from the
  contract. Zero errors on this set is not a population error-rate guarantee.
- The `key_order` case was reordered before preparation, but sorted-key JSON
  publication normalized that order. It is an input-normalization duplicate,
  **not a distinct parser/key-order challenge**. Its retained result is not
  silently deleted; no distinct-fault-coverage claim uses it.
- The two external-boundary inputs only illustrate false trusted assertions and
  lost history through supplied inputs. They do not execute a forgery attack,
  process crash or restart. Every arm admits them. This demonstrates the limits
  of the supplied model, not measured authentication or persistence resilience.
- Producers, pins and mutation authority are trusted; state is in memory.
  Authentication, durable history, concurrency, witness truth and scientific
  budget justification remain external obligations.
- The schema comparator is handwritten; this is not a JSON Schema library test.
- Warm-cache local latency is not service overhead, throughput, integration
  effort, developer productivity or independent practical validation.
- Fixed historical simulator attempts and outcomes were not run or changed.

See `results/summary.json` for runtime metadata and unrounded values. The
repository's MIT license covers these author-generated sources and fixtures.
