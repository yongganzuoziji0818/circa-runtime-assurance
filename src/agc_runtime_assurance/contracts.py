"""Versioned runtime contracts shared by the environment and assurance layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


class ContractError(ValueError):
    """Raised when a runtime message violates the frozen interface contract."""


class ExpiredActionError(ContractError):
    """Raised when an action is used after its validity period."""


@dataclass(frozen=True)
class AgentObservation:
    agent_id: str
    agent_kind: Literal["uav", "ugv"]
    monotonic_time: float
    position: np.ndarray
    velocity: np.ndarray
    neighbor_age: float
    local_risk: float
    interaction_risk: float
    confidence: float
    communication_delay: float
    packet_loss: float
    compute_budget: float

    def validate(self) -> None:
        if not self.agent_id:
            raise ContractError("agent_id must be non-empty")
        if self.monotonic_time < 0 or self.neighbor_age < 0:
            raise ContractError("timestamps and ages must be non-negative")
        for name in ("local_risk", "interaction_risk", "confidence", "packet_loss"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ContractError(f"{name} must lie in [0, 1]")
        if self.communication_delay < 0 or self.compute_budget < 0:
            raise ContractError("delay and compute budget must be non-negative")


@dataclass(frozen=True)
class ActionEnvelope:
    action: np.ndarray
    issued_at: float
    valid_until: float
    source: str
    constraint_state: str = "unchecked"

    def checked_action(self, now: float) -> np.ndarray:
        if not np.isfinite(self.issued_at) or not np.isfinite(self.valid_until):
            raise ContractError("action timestamps must be finite")
        if self.issued_at < 0:
            raise ContractError("issued_at must be non-negative")
        if self.valid_until <= self.issued_at:
            raise ExpiredActionError("action has no positive validity interval")
        if now < self.issued_at:
            raise ContractError("action cannot be consumed before it is issued")
        if now > self.valid_until:
            raise ExpiredActionError(
                f"action from {self.source!r} expired at {self.valid_until:.6f}; now={now:.6f}"
            )
        action = np.asarray(self.action, dtype=float)
        if action.ndim != 1 or not np.all(np.isfinite(action)):
            raise ContractError("action must be a finite one-dimensional vector")
        return action.copy()
