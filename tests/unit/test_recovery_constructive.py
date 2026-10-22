"""Régressions des causes physiques et informationnelles de l'impasse initiale."""

from types import SimpleNamespace
import copy
import pytest
from cav_recovery.recovery.contracts import Action, RecoveryContract
from cav_recovery.recovery.protocol import Endpoint, validate


@pytest.mark.parametrize("speed,target", [(-0.003, 2.0), (0.003, -0.6)])
def test_numerical_rest_does_not_latch_braking(speed, target):
    from cav_recovery.recovery.actuation import command

    vehicle = SimpleNamespace(velocity=[speed, 0.0], heading_theta=0.0)
    native = command(vehicle, Action(target), RecoveryContract())
    assert not vehicle.explicit_brake
    assert native[1] * target > 0


@pytest.mark.parametrize("speed,target", [(-0.5, 2.0), (0.5, -0.6)])
def test_real_direction_change_keeps_braking(speed, target):
    from cav_recovery.recovery.actuation import command

    vehicle = SimpleNamespace(velocity=[speed, 0.0], heading_theta=0.0)
    assert command(vehicle, Action(target), RecoveryContract())[1] == -1.0
    assert vehicle.explicit_brake


def test_request_contains_only_sender_declared_space_not_oracle_membership():
    endpoint = Endpoint("V0")
    message = endpoint.send(
        "CLEAR_REQUEST",
        "own:1",
        [0.0, 0.0],
        0.0,
        target="V1",
        payload=dict(space_wkt="POLYGON ((0 0, 2 0, 2 1, 0 1, 0 0))", progress_m=1.0),
    )
    assert message["version"] == "REC-V2V-2"
    validate(message, 0.2)
    altered = copy.deepcopy(message)
    altered["payload"]["global_group"] = ["V0", "V1"]
    with pytest.raises(ValueError):
        validate(altered, 0.2)


def test_distant_intent_does_not_stop_an_unrelated_vehicle():
    from tests.unit.test_recovery_prerequisites import view
    from cav_recovery.recovery.cooperative import CooperativePolicy
    from cav_recovery.recovery.safety import LocalGuard

    local = view()
    other = Endpoint("A")
    local.messages = [
        other.send(
            "INTENT",
            "A:1",
            [20.0, 0.0],
            0.0,
            payload=dict(speed_mps=-0.6, steering=0.0, distance_m=4.5),
        )
    ]
    policy = CooperativePolicy("V0", Endpoint("V0"), RecoveryContract())
    action = policy.decide(local, Action(1.0), LocalGuard())
    assert action.speed_mps == 1.0
    assert not policy.outbox


def test_global_planner_is_not_an_observation_dependency():
    import inspect
    from cav_recovery.recovery import learning, cooperative

    assert "feasibility" not in inspect.getsource(learning)
    assert "instance.views" not in inspect.getsource(cooperative.CooperativePolicy)
