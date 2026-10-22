"""Les privilèges du décideur n'augmentent pas ceux du garde commun."""

from dataclasses import replace
import pytest
from types import SimpleNamespace

from tests.unit.test_recovery_prerequisites import view
from cav_recovery.recovery.causal import FixedSensorGuard, global_view, AuditPolicy
from cav_recovery.recovery.contracts import Action, RecoveryContract
from cav_recovery.recovery.safety import LocalGuard
from cav_recovery.recovery.protocol import Endpoint


def test_global_decision_state_cannot_fill_the_local_guards_unknown_space():
    original = view()
    original.scan = dict(original.scan, ranges=[0.01] * 360)
    omniscient = replace(original, scan=dict(original.scan, ranges=[1.0] * 360))
    receipt = FixedSensorGuard(LocalGuard(), original).check(omniscient, Action(1.0))
    assert not receipt["accepted"]
    assert receipt["reason"] == LocalGuard().check(original, Action(1.0))["reason"]


def test_global_snapshot_is_current_declared_and_does_not_mutate_local_view():
    own, other = view(), view()
    other.ego = "V1"
    other.own = dict(other.own, position=[45.0, 0.0])
    result = global_view(own, {"V0": own, "V1": other}, [], RecoveryContract())
    assert not own.messages
    assert result.messages[0]["position"] == [45.0, 0.0]
    assert result.messages[0]["sent_s"] == own.at_s
    assert result.graph["scope"].startswith("privileged")


def test_warmup_uses_only_own_history_and_keeps_the_nominal_prefix():
    local = view()
    config = dict(
        strategy="current", information="local", trigger="local", not_before_s=69
    )
    policy = AuditPolicy("V0", Endpoint("V0"), RecoveryContract(), config, {})
    action = Action(1.0)
    local.diagnostic = {"status": "CONFIRMED_OBSTRUCTION"}
    local.graph = {"edges": ["unavailable global data must not be read"]}
    assert policy.decide(local, action, LocalGuard()) == action
    assert not policy.outbox
    assert policy.policy.progress_history == [(0.0, 0.0)]


def test_diagnostic_trigger_does_not_override_the_physical_guard():
    local = view()
    local.diagnostic = dict(status="CONFIRMED_OBSTRUCTION")
    local.route = SimpleNamespace(line=None)
    policy = AuditPolicy(
        "V0",
        Endpoint("V0"),
        RecoveryContract(),
        dict(
            strategy="current", information="local", trigger="d01_metro", not_before_s=1
        ),
        {},
    )
    # Avant le seuil temporel commun, même une confirmation n'autorise rien.
    assert policy.decide(local, Action(1.0), LocalGuard()).speed_mps == 1.0


def test_tcn_trigger_refuses_silent_execution_without_its_frozen_model(tmp_path):
    from cav_recovery.recovery.causal import run_condition

    with pytest.raises(ValueError, match="artefact gelé"):
        run_condition({}, tmp_path / "missing", dict(trigger="tcn"))
