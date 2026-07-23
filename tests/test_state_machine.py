from agc_runtime_assurance.state_machine import AssuranceMode, RuntimeAssuranceStateMachine


def test_invalid_certificate_fails_closed():
    machine = RuntimeAssuranceStateMachine()
    transition = machine.step(
        certificate_valid=False, nominal_safe=True, filter_feasible=True,
        backup_recoverable=True, observations_fresh=True,
    )
    assert transition.current == AssuranceMode.BACKUP


def test_recovery_requires_hysteresis():
    machine = RuntimeAssuranceStateMachine(recovery_hold_steps=2)
    machine.step(certificate_valid=False, nominal_safe=False, filter_feasible=False, backup_recoverable=True)
    first = machine.step(certificate_valid=True, nominal_safe=True, filter_feasible=True, backup_recoverable=True)
    second = machine.step(certificate_valid=True, nominal_safe=True, filter_feasible=True, backup_recoverable=True)
    assert first.current == AssuranceMode.RECOVERY
    assert second.current == AssuranceMode.NOMINAL


def test_recoverability_gate_switches_before_verified_region_is_lost():
    machine = RuntimeAssuranceStateMachine()
    transition = machine.step(
        certificate_valid=True, nominal_safe=True, filter_feasible=True,
        backup_recoverable=False, observations_fresh=True,
    )
    assert transition.current == AssuranceMode.BACKUP
    assert transition.reason == "recoverability_gate_triggered"
