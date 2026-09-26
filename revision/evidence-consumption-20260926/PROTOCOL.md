# CIRCA local evidence-consumption comparison v1

Date: 2026-09-26. Scope: new author-constructed, local software evaluation. This
does not rerun or replace any historical simulator/scientific attempt. No network,
controller, training, seed top-up, operational authorization or scientific output
is involved. It is not a publicly preregistered or independent field study.

## Question and estimand

For fixed JSON evidence-consumption workflows, which checks prevent an
ineligible evidence object from being marked admissible? The unit is one named
scenario, not a timing repetition. Every arm receives the same serialized inputs,
trusted digest, lifecycle events and decision threshold. The expected binary
admissibility is declared when constructing each case, independently of the
tested consumer's return value. All scenario results, including failures, will be
retained. No p-values or population-generalizing accuracy confidence intervals.

## Arms (all are author implementations, not replicas of named external tools)

1. **Numeric report**: accepts a finite lower bound meeting the request threshold.
2. **Typed schema + numeric report**: validates exact field sets, scalar types,
   ranges, hash encodings and timestamps, then applies the threshold. This is a
   transparent handwritten schema checker, NOT a JSON Schema library benchmark.
3. **Stateless contract rules**: schema, numeric threshold, scope, pinned content,
   Boolean validity premises and validity window; no lifecycle registry.
4. **Stateful contract rules**: the same conventional checks plus registration and
   terminal invalidation. Implemented independently without importing CIRCA.
   This is the strong functional comparator; parity is an informative result.
5. **CIRCA reference**: the unchanged r1 evidence_gate.py and its actual Registry.

This nested comparison isolates additional obligations rather than comparing
CIRCA against an intentionally broken implementation. Weak arms are explanatory
ablations, not evidence of superiority to assurance frameworks or human review.

## Input path and outcomes

Each case passes through actual disk JSON input, JSON decoding, arm-specific
registration/event processing, consumption and serialized claim-report output.
The report contains admissible, status and operational_authorization=false.
The harness must not reinterpret an exception as a positive decision. Record
exceptions separately. Expected-reject cases test mistaken acceptance; expected-
accept cases test unwarranted refusal. Preserve counts by fault family and case.

Positive cases include legitimate identity refresh, timezone equivalence, key
order, inclusive time endpoints, and threshold equality. Negative cases include
schema defects, all identity roles, mutated content, false premises, insufficient
bounds, expiration, unregistered objects and each terminal lifecycle state.
Additional explicitly out-of-contract cases expose false trusted producer
assertions and state loss after restarting an in-memory registry. These must stay
visible and must NOT be included in the supported-scope success denominator.

## Cost measurements and repetitions

After one untimed correctness pass, run 25 complete paired blocks; rotate arm
order deterministically by block index to reduce position bias. Measure complete
disk-read to serialized-report latency in milliseconds for each arm/case. Report
median and empirical p95, with raw timings, Python/OS/CPU metadata and case count.
No inferential timing claims: these are warm-cache Windows microbenchmarks, not
independent replications, production throughput, developer effort or maintenance
cost. Do not select a fastest subset. No random scientific seed is generated.

## Commitment, execution and interpretation

Before measurement, preserve protocol, cases, both implementations and runner
SHA-256 in a fresh commitment file. Keep original sources byte-identical.
Implementation defects may be repaired before the committed run using disjoint
smoke checks; after an observed run, retain all evidence and label any correction
as a new evaluation version, never silently replace results.

Success criterion for the narrow conformance claim is zero mistaken acceptances
and zero unwarranted refusals on the fixed supported-scope set, with actual status
outputs disclosed. A nonzero count is a visible limitation, not permission to
change scenarios. Broader practical benefit still requires external workloads,
independent systems/engineers, deployment-quality authentication/persistence and
appropriate comparative study design.
