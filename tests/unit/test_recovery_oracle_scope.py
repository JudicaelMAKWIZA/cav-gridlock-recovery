"""Le garde idéal expérimental ne supprime pas les contraintes non perceptives."""

from types import SimpleNamespace
from tests.unit.test_recovery_prerequisites import view
from cav_recovery.recovery.contracts import Action, RecoveryContract
from cav_recovery.recovery.feasibility import GlobalBodyGuard
from cav_recovery.recovery.safety import LocalGuard


def test_global_occupancy_is_explicit_and_does_not_override_limits_or_missing_scan():
    local = view()
    guard = GlobalBodyGuard(
        SimpleNamespace(views={"V0": local}, contract=RecoveryContract())
    )
    assert guard.check(local, Action(3.0))["reason"] == "ACTION_LIMIT"
    local.scan["missing"] = True
    assert guard.check(local, Action(1.0))["reason"] == "PERCEPTION_UNAVAILABLE"


def test_global_bodies_cannot_remove_finite_horizon_reverse_budget():
    local = view()
    local.remaining_horizon_s = 100
    local.reverse_used_m = 7
    guard = GlobalBodyGuard(
        SimpleNamespace(views={"V0": local}, contract=RecoveryContract())
    )
    assert guard.check(local, Action(-0.6))["reason"] == "REVERSE_STOPPING_BUDGET"


def test_occlusion_relaxation_is_a_declared_privilege_not_a_local_certificate():
    local = view()
    local.scan["ranges"] = [0.1] * 360
    guard = GlobalBodyGuard(
        SimpleNamespace(views={"V0": local}, contract=RecoveryContract())
    )
    assert not LocalGuard().check(local, Action(1.0))["accepted"]
    receipt = guard.check(local, Action(1.0))
    assert receipt["accepted"]
    assert receipt["scope"] == "privileged_global_upper_bound_not_operational"
