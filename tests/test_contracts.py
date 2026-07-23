import numpy as np
import pytest

from agc_runtime_assurance.contracts import (
    ActionEnvelope,
    AgentObservation,
    ContractError,
    ExpiredActionError,
)


def test_expired_action_is_rejected():
    envelope = ActionEnvelope(np.zeros(2), issued_at=1.0, valid_until=2.0, source="nominal")
    with pytest.raises(ExpiredActionError):
        envelope.checked_action(2.01)


def test_risk_and_confidence_are_separate_and_bounded():
    obs = AgentObservation(
        "u1", "uav", 1.0, np.zeros(3), np.zeros(3), 0.1,
        local_risk=0.3, interaction_risk=0.8, confidence=0.6,
        communication_delay=0.02, packet_loss=0.1, compute_budget=1.0,
    )
    obs.validate()
    invalid = AgentObservation(**{**obs.__dict__, "confidence": 1.1})
    with pytest.raises(ContractError):
        invalid.validate()
