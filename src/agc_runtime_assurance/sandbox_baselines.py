"""Shared-task runtime adapters for G0 baseline compatibility.

These adapters bind decisions to ``SandboxComparisonTask`` fingerprints.  They
do not replace original-task reproduction or learned components from papers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .acofi import ACoFiDecision, ACoFiRuntimeAdapter
from .conformal_cbf import ConformalCBFStep, MultiAgentConformalCBF
from .filtering import AffineSafetyFilter, FilterResult
from .sandbox_task import AffineConstraintBundle, SandboxComparisonTask


class SandboxBaselineInfeasible(RuntimeError):
    pass


@dataclass(frozen=True)
class SandboxBaselineDecision:
    action: np.ndarray
    nominal_action: np.ndarray
    filter_result: FilterResult
    constraint_bundle: AffineConstraintBundle
    nominal_policy_fingerprint: str
    constraint_contract_fingerprint: str


@dataclass(frozen=True)
class SandboxConformalCBFDecision:
    action: np.ndarray
    nominal_action: np.ndarray
    conformal_step: ConformalCBFStep
    constraint_bundle: AffineConstraintBundle
    nominal_policy_fingerprint: str
    constraint_contract_fingerprint: str


@dataclass(frozen=True)
class SandboxACoFiDecision:
    action: np.ndarray
    acofi_decision: ACoFiDecision
    safe_decision: SandboxBaselineDecision
    current_team_margin: float
    exact_next_step_postcheck: bool
    nominal_policy_fingerprint: str
    constraint_contract_fingerprint: str


class SandboxNominalCBFAdapter:
    """Minimal-intervention affine filter on the frozen shared task."""

    def __init__(self, task: SandboxComparisonTask | None = None):
        self.task = task or SandboxComparisonTask()
        self.filter = AffineSafetyFilter(self.task.action_lower, self.task.action_upper)

    def decide(
        self, augmented_state: np.ndarray, *, fallback_action: np.ndarray,
    ) -> SandboxBaselineDecision:
        nominal = self.task.nominal_action(augmented_state)
        bundle = self.task.next_state_constraints(
            augmented_state, separation_reference_action=nominal,
        )
        result = self.filter.filter(nominal, bundle.A, bundle.b, fallback_action)
        if not self.task.postcheck_next_state(augmented_state, result.action):
            raise SandboxBaselineInfeasible(
                "nominal CBF result failed the exact shared-task postcheck"
            )
        return SandboxBaselineDecision(
            result.action.copy(), nominal, result, bundle,
            self.task.nominal_policy_fingerprint,
            self.task.constraint_contract_fingerprint,
        )


class SandboxConformalCBFAdapter:
    """Bind the multi-agent conformal-CBF online kernel to the shared task."""

    def __init__(
        self,
        *,
        target_loss: float,
        learning_rate: float,
        initial_value: float,
        task: SandboxComparisonTask | None = None,
    ):
        self.task = task or SandboxComparisonTask()
        self.controller = MultiAgentConformalCBF(
            safety_filter=AffineSafetyFilter(
                self.task.action_lower, self.task.action_upper,
            ),
            target_loss=target_loss,
            learning_rate=learning_rate,
            initial_value=initial_value,
        )

    def decide(
        self, augmented_state: np.ndarray, *, fallback_action: np.ndarray,
    ) -> SandboxConformalCBFDecision:
        nominal = self.task.nominal_action(augmented_state)
        bundle = self.task.next_state_constraints(
            augmented_state, separation_reference_action=nominal,
        )
        # Shared contract A u <= b becomes G u + d + lambda >= 0 with
        # G=-A and d=b.  Negative lambda is therefore more conservative.
        step = self.controller.filter_action(
            nominal_action=nominal,
            control_coefficients=-bundle.A,
            predicted_offsets=bundle.b,
            fallback_action=fallback_action,
        )
        action = step.filter_result.action
        if not self.task.postcheck_next_state(augmented_state, action):
            raise SandboxBaselineInfeasible(
                "conformal-CBF result failed the exact shared-task postcheck"
            )
        return SandboxConformalCBFDecision(
            action.copy(), nominal, step, bundle,
            self.task.nominal_policy_fingerprint,
            self.task.constraint_contract_fingerprint,
        )


class SandboxACoFiAdapter:
    """Bind ACoFi switching to the shared task and shared nominal-CBF backup.

    Learned Q/HJ/world-model values remain external.  If ACoFi selects its task
    policy, the action is not silently post-filtered; the exact next-step check
    is recorded so evaluation can count the resulting violation risk honestly.
    """

    def __init__(
        self,
        *,
        target_alpha: float,
        learning_rate: float,
        gamma: float,
        safety_threshold: float,
        task: SandboxComparisonTask | None = None,
    ):
        self.task = task or SandboxComparisonTask()
        self.safe_adapter = SandboxNominalCBFAdapter(self.task)
        self.controller = ACoFiRuntimeAdapter(
            target_alpha=target_alpha,
            learning_rate=learning_rate,
            gamma=gamma,
            safety_threshold=safety_threshold,
        )

    def decide(
        self,
        augmented_state: np.ndarray,
        *,
        step_index: int,
        predicted_task_q: float,
        fallback_action: np.ndarray,
    ) -> SandboxACoFiDecision:
        nominal = self.task.nominal_action(augmented_state)
        safe = self.safe_adapter.decide(
            augmented_state, fallback_action=fallback_action,
        )
        margins = self.task.point_constraint_margins(
            augmented_state, step_index=step_index,
        )
        team_margin = float(np.min(margins.as_array()))
        decision = self.controller.decide(
            predicted_task_q=predicted_task_q,
            current_local_margin=team_margin,
            task_action=nominal,
            safe_action=safe.action,
        )
        return SandboxACoFiDecision(
            decision.action.copy(), decision, safe, team_margin,
            self.task.postcheck_next_state(augmented_state, decision.action),
            self.task.nominal_policy_fingerprint,
            self.task.constraint_contract_fingerprint,
        )

    def observe_transition(
        self,
        *,
        previous_predicted_q: float,
        previous_local_margin: float,
        next_learned_value: float,
    ) -> tuple[float, bool]:
        return self.controller.observe_transition(
            previous_predicted_q=previous_predicted_q,
            previous_local_margin=previous_local_margin,
            next_learned_value=next_learned_value,
        )
