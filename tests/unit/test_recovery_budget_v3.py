"""Le budget couvre le freinage puis le suivi, sans effacer le recul mesuré."""

from dataclasses import replace
from tests.unit.test_recovery_prerequisites import view
from cav_recovery.recovery.contracts import Action, RecoveryContract
from cav_recovery.recovery.safety import LocalGuard
import pytest


def test_future_stationary_drift_consumes_a_declared_budget_reserve():
    local = view()
    local.remaining_horizon_s = 100
    local.reverse_used_m = 7
    action = Action(-0.6)
    assert (
        LocalGuard().check(local, action, require_agreement=False)["reason"]
        == "REVERSE_STOPPING_BUDGET"
    )
    historical = replace(RecoveryContract(), version="recovery-physical-2")
    assert LocalGuard(historical).check(local, action, require_agreement=False)[
        "accepted"
    ]


def test_settling_reserve_cannot_be_removed_by_an_unknown_episode_horizon():
    local = view()
    local.remaining_horizon_s = float("nan")
    assert LocalGuard().check(local, Action(-0.6))["reason"] == "HORIZON_UNAVAILABLE"


def test_training_cannot_reuse_a_proof_from_a_different_physical_contract(tmp_path):
    from cav_recovery.recovery.pilot import train
    from cav_recovery.laboratory.storage import atomic_json

    atomic_json(
        tmp_path / "proof.json",
        dict(intervention=dict(success=True), nominal=dict(success=False)),
    )
    atomic_json(
        tmp_path / "cooperative/summary.json",
        dict(method="cooperative", global_audit_in_actor=False),
    )
    old = replace(RecoveryContract(), version="recovery-physical-2")
    atomic_json(tmp_path / "cooperative/contract.json", old.record())
    with pytest.raises(ValueError, match="domaine physique courant"):
        train(
            tmp_path / "absent-config.json",
            tmp_path / "new-pilot",
            proof_path=tmp_path / "proof.json",
        )
