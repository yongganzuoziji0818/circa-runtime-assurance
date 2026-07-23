from dataclasses import fields
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/agc_runtime_assurance"))

from circa import (  # noqa: E402
    INEVITABLE_VIOLATION,
    NOMINAL_SAFETY,
    NO_WITNESS,
    OBSERVED_CONSISTENCY,
)
from circa_gazebo import (  # noqa: E402
    GazeboEvidenceContract,
    GazeboObservedRecord,
    GazeboOracleRecord,
    GazeboRegimeTrace,
    GazeboWitnessInput,
    attach_oracle_audit,
    compile_observed_scenario,
    compile_oracle_scenario,
    derive_registered_witness,
    derive_scenario_seed,
    evaluate_observed_family,
    first_violation_from_margins,
    intervention_from_actions,
    schema_capacity_summary,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _contract() -> GazeboEvidenceContract:
    return GazeboEvidenceContract(("GZE1",), HASH_A, HASH_B, HASH_C, HASH_D, 80)


def _record(seed: int, r: int, y1: int, witness: int, **overrides) -> GazeboObservedRecord:
    values = dict(
        scenario_seed=seed,
        family_id="GZE1",
        intervened=r,
        active_first_violation=y1,
        witness_code=witness,
        outcome_complete=True,
        interference_registered=True,
        policy_sha256=HASH_A,
        constraint_sha256=HASH_B,
        witness_model_sha256=HASH_C,
        interaction_graph_sha256=HASH_D,
        horizon_steps=80,
    )
    values.update(overrides)
    return GazeboObservedRecord(**values)


def test_witness_input_has_no_oracle_outcome_and_derives_only_robust_codes():
    assert "shadow_first_violation" not in {field.name for field in fields(GazeboWitnessInput)}
    factual = GazeboWitnessInput(0, 0, 8.0, 0.1, 1.0)
    crossing = GazeboWitnessInput(1, 0, 8.0, 0.1, 1.0, 6.0, None, 1.0, 0.5)
    safe = GazeboWitnessInput(1, 0, 8.0, 0.1, 1.0, None, 0.2)
    unresolved = GazeboWitnessInput(1, 0, 8.0, 0.3, 1.0, None, 0.2)
    assert derive_registered_witness(factual).code == OBSERVED_CONSISTENCY
    assert derive_registered_witness(crossing).code == INEVITABLE_VIOLATION
    assert derive_registered_witness(safe).code == NOMINAL_SAFETY
    assert derive_registered_witness(unresolved).code == NO_WITNESS


def test_trace_adapters_create_binary_first_violation_and_intervention():
    assert first_violation_from_margins([[1.0, 0.2], [0.4, -0.01]]) == 1
    assert first_violation_from_margins([[1.0, 0.2], [0.4, 0.01]]) == 0
    nominal = np.zeros((3, 5))
    active = nominal.copy()
    active[1, 2] = 0.1
    assert intervention_from_actions(nominal, active) == 1
    assert intervention_from_actions(nominal, nominal) == 0


def test_active_and_shadow_tables_are_compiled_on_separate_paths():
    nominal = ((0.0, 0.0), (0.0, 0.0))
    active = GazeboRegimeTrace(7, "GZE1", 2, ((1.0, 0.5), (0.2, 0.1)), nominal, True)
    witness = GazeboWitnessInput(0, 0, 0.2, 0.0, 1.0)
    observed = compile_observed_scenario(
        active,
        nominal,
        witness,
        policy_sha256=HASH_A,
        constraint_sha256=HASH_B,
        witness_model_sha256=HASH_C,
        interaction_graph_sha256=HASH_D,
    )
    shadow = GazeboRegimeTrace(7, "GZE1", 2, ((1.0, 0.5), (0.2, -0.1)), nominal, True)
    oracle = compile_oracle_scenario(shadow)
    assert observed.active_first_violation == 0
    assert not hasattr(observed, "shadow_first_violation")
    assert oracle.shadow_first_violation == 1


def test_family_compilation_and_oracle_audit_keep_tables_separate():
    observed = [
        _record(1, 0, 0, OBSERVED_CONSISTENCY),
        _record(2, 1, 0, INEVITABLE_VIOLATION),
        _record(3, 1, 0, NO_WITNESS),
        _record(4, 1, 0, NOMINAL_SAFETY),
    ]
    result = evaluate_observed_family(observed, _contract())
    assert result["status"] == "VALID"
    assert result["circa_identification_interval"] == [0.25, 0.5]
    assert result["manski_identification_interval"] == [0.0, 0.75]
    audit = attach_oracle_audit(
        observed,
        [
            GazeboOracleRecord(1, "GZE1", 0),
            GazeboOracleRecord(2, "GZE1", 1),
            GazeboOracleRecord(3, "GZE1", 1),
            GazeboOracleRecord(4, "GZE1", 0),
        ],
    )
    assert audit["paired_oracle_sample_delta"] == 0.5
    assert audit["all_registered_witnesses_sound_on_paired_gazebo_sample"] is True


def test_corrupt_provenance_and_witness_semantics_refuse_numeric_output():
    corrupt = [_record(1, 1, 0, INEVITABLE_VIOLATION, policy_sha256="e" * 64)]
    result = evaluate_observed_family(corrupt, _contract())
    assert result["status"] == "INVALID_PROVENANCE"
    assert result["numeric_certificate_allowed"] is False
    malformed = [_record(1, 0, 0, NO_WITNESS)]
    result = evaluate_observed_family(malformed, _contract())
    assert result["status"] == "INVALID_WITNESS"
    assert result["circa_identification_interval"] is None


def test_seed_derivation_and_capacity_are_reproducible_and_precise_enough():
    seeds = [derive_scenario_seed(2026072001, "GZE1", index) for index in range(1200)]
    assert len(set(seeds)) == 1200
    summary = schema_capacity_summary(
        families=("GZC1", "GZC2", "GZE1", "GZE2", "GZE3", "GZE4"),
        scenarios_per_family=1200,
        endpoint_count=12,
        alpha=0.05,
        estimated_bytes_per_paired_scenario=1024,
    )
    assert summary["independent_paired_scenarios"] == 7200
    assert summary["gazebo_regime_rollouts"] == 14400
    assert summary["simultaneous_hoeffding_radius"] < 0.10
