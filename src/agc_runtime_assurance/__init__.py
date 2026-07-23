"""G0 runtime-assurance primitives for the P4 research line."""

from .acofi import ACoFiDecision, ACoFiRuntimeAdapter
from .action_cells import CellConditionalValidityBank, FirstPassageCellSpec
from .assurance_case import (
    AssuranceCaseVerification,
    build_valid_assurance_case,
    inject_assurance_fault,
    verify_assurance_case,
)
from .backup_invariant import BackupInvariantResult, LinearFeedbackInvariantBoxVerifier
from .baseline_evidence import (
    BaselineEvidenceError,
    BaselineEvidenceFinding,
    BaselineMatrixEvidenceReport,
    FrozenComparisonBudget,
    verify_baseline_reproduction_manifest,
)
from .censored_validity import (
    FirstPassageObservation,
    FirstPassageObservationKind,
    NaiveCensoredValidityCertificate,
)
from .contracts import ActionEnvelope, AgentObservation, ExpiredActionError
from .deadline_metrics import (
    DeadlineAuditRow,
    DeadlineCoverageSummary,
    summarize_deadline_coverage,
)
from .development_analysis import (
    DeadlineAggregate,
    DevelopmentAnalysisError,
    DevelopmentAnalysisReport,
    GuardrailAggregate,
    PairedSeedEffect,
    SeedFamilyResult,
    analyze_development_results,
)
from .conformal_cbf import MultiAgentConformalCBF, conformal_cbf_interval_loss
from .fallback_mpc import (
    FallbackMPCSolution,
    FallbackTubeResult,
    InvariantBoundFallbackResult,
    InvariantBoundFallbackTubeVerifier,
    LinearBoxFallbackSafeMPC,
    LinearBoxFallbackSafeMPCQP,
    LinearBoxFallbackTubeVerifier,
)
from .fallback_monitor import ConformalStoppingTimeMonitor, FallbackMonitorDecision
from .horizon_solver import (
    SelfConsistentHorizon,
    solve_censored_horizon,
    solve_crossing_horizon,
)
from .latency import HandoverLatencyCertificate
from .paper_readiness import (
    PaperReadinessReport,
    ReadinessFinding,
    ReadinessTier,
    audit_paper_readiness,
)
from .power_planning import (
    PowerPlanningError,
    SeedPowerPlan,
    SeedPowerPoint,
    plan_seed_power_from_manifest,
    simulate_seed_power_plan,
)
from .risk import ConformalQuantileCertificate, ConstraintMargins, team_safety_score
from .sandbox_model import (
    ModelUncertaintyEnvelope,
    ShiftParameterBox,
    air_ground_augmented_matrices,
    sandbox_axis_aligned_state_constraints,
    sandbox_constraint_margin_lower_bound,
    sandbox_parameter_uncertainty_envelope,
)
from .sandbox_task import AffineConstraintBundle, SandboxComparisonTask
from .sandbox_baselines import (
    SandboxACoFiAdapter,
    SandboxACoFiDecision,
    SandboxBaselineDecision,
    SandboxBaselineInfeasible,
    SandboxConformalCBFAdapter,
    SandboxConformalCBFDecision,
    SandboxNominalCBFAdapter,
)
from .sandbox_fallback import (
    SandboxFallbackDecision,
    SandboxFallbackSafeMPCAdapter,
    sandbox_backup_equilibrium,
    sandbox_backup_gain,
    sandbox_backup_invariant_radius,
)
from .scenario_manifest import (
    G0Scenario,
    G0ScenarioManifest,
    RuntimeTimingScenario,
    ScenarioManifestError,
    load_g0_scenario_manifest,
)
from .sensitivity import FirstPassageSensitivityBound, first_passage_sensitivity_bound
from .simultaneous_cells import SimultaneousCellCertificate
from .transversality import (
    ConstraintPassageEvidence,
    ConstraintTransportResult,
    TeamTransportResult,
    transport_constraint_horizon,
    transport_team_horizon,
)
from .recoverability import (
    AssuredRecoverabilityResult,
    MarginErosionRecoverabilityGate,
    RecoverabilityResult,
    VerifiedBackupRecoverabilityGate,
)
from .predictor import NominalRolloutHorizonPredictor
from .paired_risk_reduction import (
    FamilyRiskReductionCertificate,
    PairedRiskReductionError,
    SimultaneousRiskReductionCertificate,
    family_risk_reduction_certificate,
    importance_weights_from_log_densities,
    simultaneous_risk_reduction_certificate,
)
from .state_machine import AssuranceMode, RuntimeAssuranceStateMachine
from .supervisor import AssuranceDecision, RuntimeAssuranceSupervisor
from .validity import ActionValidityCertificate, first_violation_time
from .weighted_validity import WeightedActionValidityCertificate

__all__ = [
    "ActionEnvelope",
    "ACoFiDecision",
    "ACoFiRuntimeAdapter",
    "ActionValidityCertificate",
    "AffineConstraintBundle",
    "BackupInvariantResult",
    "BaselineEvidenceError",
    "BaselineEvidenceFinding",
    "BaselineMatrixEvidenceReport",
    "FrozenComparisonBudget",
    "WeightedActionValidityCertificate",
    "CellConditionalValidityBank",
    "FirstPassageObservation",
    "FirstPassageObservationKind",
    "NaiveCensoredValidityCertificate",
    "AgentObservation",
    "AssuranceMode",
    "AssuranceDecision",
    "AssuranceCaseVerification",
    "AssuredRecoverabilityResult",
    "ConformalQuantileCertificate",
    "MultiAgentConformalCBF",
    "ConstraintMargins",
    "DeadlineAuditRow",
    "DeadlineCoverageSummary",
    "DeadlineAggregate",
    "DevelopmentAnalysisError",
    "DevelopmentAnalysisReport",
    "ExpiredActionError",
    "FallbackTubeResult",
    "InvariantBoundFallbackResult",
    "InvariantBoundFallbackTubeVerifier",
    "FirstPassageCellSpec",
    "FirstPassageSensitivityBound",
    "FallbackMPCSolution",
    "FallbackMonitorDecision",
    "G0Scenario",
    "G0ScenarioManifest",
    "HandoverLatencyCertificate",
    "GuardrailAggregate",
    "PaperReadinessReport",
    "PairedRiskReductionError",
    "FamilyRiskReductionCertificate",
    "SimultaneousRiskReductionCertificate",
    "PowerPlanningError",
    "PairedSeedEffect",
    "ReadinessFinding",
    "ReadinessTier",
    "SelfConsistentHorizon",
    "LinearBoxFallbackSafeMPC",
    "LinearBoxFallbackSafeMPCQP",
    "LinearFeedbackInvariantBoxVerifier",
    "ConformalStoppingTimeMonitor",
    "LinearBoxFallbackTubeVerifier",
    "MarginErosionRecoverabilityGate",
    "ModelUncertaintyEnvelope",
    "NominalRolloutHorizonPredictor",
    "RecoverabilityResult",
    "VerifiedBackupRecoverabilityGate",
    "RuntimeAssuranceStateMachine",
    "RuntimeAssuranceSupervisor",
    "RuntimeTimingScenario",
    "ScenarioManifestError",
    "ShiftParameterBox",
    "SeedFamilyResult",
    "SeedPowerPlan",
    "SeedPowerPoint",
    "SandboxComparisonTask",
    "SandboxACoFiAdapter",
    "SandboxACoFiDecision",
    "SandboxBaselineDecision",
    "SandboxBaselineInfeasible",
    "SandboxConformalCBFAdapter",
    "SandboxConformalCBFDecision",
    "SandboxNominalCBFAdapter",
    "SandboxFallbackDecision",
    "SandboxFallbackSafeMPCAdapter",
    "air_ground_augmented_matrices",
    "analyze_development_results",
    "audit_paper_readiness",
    "first_violation_time",
    "first_passage_sensitivity_bound",
    "load_g0_scenario_manifest",
    "SimultaneousCellCertificate",
    "ConstraintPassageEvidence",
    "ConstraintTransportResult",
    "TeamTransportResult",
    "transport_constraint_horizon",
    "transport_team_horizon",
    "verify_baseline_reproduction_manifest",
    "verify_assurance_case",
    "build_valid_assurance_case",
    "inject_assurance_fault",
    "conformal_cbf_interval_loss",
    "team_safety_score",
    "summarize_deadline_coverage",
    "sandbox_axis_aligned_state_constraints",
    "sandbox_constraint_margin_lower_bound",
    "sandbox_backup_equilibrium",
    "sandbox_backup_gain",
    "sandbox_backup_invariant_radius",
    "sandbox_parameter_uncertainty_envelope",
    "plan_seed_power_from_manifest",
    "family_risk_reduction_certificate",
    "importance_weights_from_log_densities",
    "simultaneous_risk_reduction_certificate",
    "solve_censored_horizon",
    "solve_crossing_horizon",
    "simulate_seed_power_plan",
]
