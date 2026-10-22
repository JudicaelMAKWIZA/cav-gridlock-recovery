"""La préparation mesurée doit être suivie de l'étape annoncée, pas répétée."""

from types import SimpleNamespace
from cav_recovery.recovery.sequence_audit import SequencedCoordinator
from cav_recovery.recovery.contracts import Action, RecoveryContract
from cav_recovery.recovery.protocol import Endpoint


def scene(at, position):
    views = {
        ego: SimpleNamespace(own=dict(position=xy, speed_mps=0.0))
        for ego, xy in {"V7": position, "V3": [3.0, 0.0]}.items()
    }
    return SimpleNamespace(
        at_s=at,
        views=views,
        contract=RecoveryContract(),
        endpoints={ego: Endpoint(ego) for ego in views},
    )


def coordinator():
    policy = SequencedCoordinator()
    policy.plan = dict(
        ego="V7",
        speed_mps=-0.6,
        steering=0.0,
        distance_m=2.0,
        lookahead_second=dict(ego="V3", steering=-0.3, distance_m=2.0),
    )
    policy.anchor = [0.0, 0.0]
    policy.started = 0.0
    policy.count = 1
    policy.attempt = "global:1"
    return policy


def test_completed_preparation_commits_the_second_vehicle_with_fresh_negotiation():
    policy = coordinator()
    world = scene(10.0, [2.0, 0.0])
    nominal = {ego: Action(1.0) for ego in world.views}
    policy.decide(world, nominal)
    assert policy.plan is None and policy.pending_second["ego"] == "V3"
    world.at_s = 10.2
    actions, _ = policy.decide(world, nominal)
    assert all(a.speed_mps == 0 for a in actions.values())
    world.at_s = 10.4
    policy.decide(world, nominal)
    assert policy.plan["ego"] == "V3" and policy.count == 2
    world.at_s = 10.8
    actions, _ = policy.decide(world, nominal)
    assert actions["V3"].speed_mps == -0.6
    assert actions["V7"].speed_mps == 0
    assert world.views["V7"].own["position"] == [2.0, 0.0]


def test_a_timed_out_incomplete_preparation_does_not_authorize_its_second_step():
    policy = coordinator()
    world = scene(30.0, [0.1, 0.0])
    policy.decide(world, {ego: Action(0.0) for ego in world.views})
    assert policy.pending_second is None
    assert any(r.get("event") == "preparatory_step_incomplete" for r in policy.log)
