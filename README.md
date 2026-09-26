# CIRCA Runtime Assurance

Research code for Counterfactual-Interval Runtime Control Assurance (CIRCA),
witness-structured partial identification, finite-sample runtime-assurance
certificates, and simulator adapters for air-ground coordination studies.

The July 2026 update adds the versioned Gazebo diversity worlds, independent
adapter, V9 feasible-initial-domain implementation, segmented SCI-S3 runner,
frozen analysis code, and non-running boundary/schema tests. It also adds the
V10 bounded-loss theory utilities, typed offline assurance-case implementation,
assumption-matched comparison code, exploratory mechanism analysis, and
one-command public-artifact verifiers. The corresponding outcome-locked
evidence is released separately as a versioned Zenodo archive.

The historical release is code-only. The September 2026 addition below also
includes author-constructed software fixtures and their local comparison output;
it does not add simulator outcomes. Manuscripts, author contact records,
scientific seeds, private infrastructure configuration and internal governance
files remain excluded.

## September 2026 evidence-consumption artifact

[revision/evidence-consumption-20260926](revision/evidence-consumption-20260926)
contains a separate reference consumer, a locally precommitted comparison with
four author-implemented conventional rule variants, 44 JSON workflows, retained
decisions/timings and a standard-library reproducibility verifier. The complete
stateful comparator and CIRCA both have zero mistaken acceptances on the fixed
32 ineligible cases and zero unwarranted refusals on the 10 eligible cases.
This is functional parity, **not superiority over existing assurance tools**.
Two extra trust/persistence examples expose responsibilities outside the model.

This versioned addition does not alter the historical implementation or results.
It is not an authenticated, persistent or concurrent service; operational
authorization is always disabled. See its README for limitations and commands.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -e .
```

For development and unit tests:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The optional low-resource PyBullet route requires:

```bash
python -m pip install -e ".[pybullet]"
```

Gazebo and Isaac Sim adapters require compatible simulator installations
provided separately by their vendors.

## Repository layout

```text
src/agc_runtime_assurance/  CIRCA and runtime-assurance implementation
tests/                      Unit, invariant, schema, and fail-closed tests
sim/gazebo/                 Gazebo world model
sim/isaac/                  Isaac Sim USD world models
scripts/                    Schema, preflight, seed-freeze, and frozen-analysis tools
experiments/manifests/      Public frozen design contracts required by tests
```

## Scope

The repository includes theoretical and computational research software,
schema builders, simulator adapters, and tests. Publishing this source does
not authorize a scientific rerun, change any frozen experimental contract, or
constitute evidence of real-world effectiveness, deployment safety, or
certification.

## License

Released under the [MIT License](LICENSE).
