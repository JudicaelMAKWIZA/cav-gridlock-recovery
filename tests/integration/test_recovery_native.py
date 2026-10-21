"""Vraies acquisitions arrière et commandes Bullet de récupération."""

import copy
import importlib.util

import pytest

from cav_recovery.experiments.catalog import specification
from cav_recovery.recovery.contracts import Action, RecoveryContract

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("metadrive") is None, reason="MetaDrive natif requis"
)


def two_car_spec():
    spec = specification("nominal-1", seed=21001)
    first = spec["agents"][0]
    first.update(longitude=25.0, departure_s=0.0)
    second = copy.deepcopy(first)
    second.update(id="agent1", longitude=18.0)
    spec.update(agents=[first, second], active_limit=2, horizon_s=12.0)
    return spec


@pytest.mark.integration
def test_native_rear_obstacle_is_seen_and_permission_cannot_be_bypassed(tmp_path):
    from cav_recovery.recovery.environment import RecoveryEnv

    env = RecoveryEnv(two_car_spec(), tmp_path / "rear", RecoveryContract())
    try:
        env.views = env.reset()
        assert set(env.views) == {"agent0", "agent1"}
        assert min(env.views["agent0"].scan["ranges"][150:211]) < 0.3
        result = env.guard.check(env.views["agent0"], Action(-0.6))
        assert not result["accepted"]
        # Sans message, le rayon reste une preuve anonyme, jamais de l'espace libre.
        no_messages = copy.copy(env.views["agent0"])
        no_messages.messages = []
        no_messages.scan = copy.deepcopy(no_messages.scan)
        no_messages.scan["ranges"] = [1.0] * 360
        assert env.guard.check(no_messages, Action(-0.6))["accepted"]
        # La simulation réelle garde les vrais rayons et exige l'accord du CAV connu.
        env.step({"agent0": Action(-0.6), "agent1": Action(0.0)})
        assert env.last_commands["agent0"]["applied"]["speed_mps"] == 0.0
        assert not env.errors
    finally:
        env.close()


@pytest.mark.integration
def test_native_reverse_with_refreshed_hold_stops_before_rear_body(tmp_path):
    from cav_recovery.recovery.environment import RecoveryEnv

    env = RecoveryEnv(two_car_spec(), tmp_path / "braking", RecoveryContract())
    attempt = "agent0:rear-test"
    try:
        env.views = env.reset()
        initial = env.views["agent0"].own["progress_m"]
        for index in range(50):
            rear = env.endpoints["agent1"]
            agreement = rear.send(
                "AGREE_HOLD",
                attempt,
                env.views["agent1"].own["position"],
                env.at_s,
                target="agent0",
                payload={"until_s": env.at_s + 0.6},
            )
            env.views["agent1"].own_commitment_until_s = rear.committed_until_s
            env.step(
                {"agent0": Action(-0.6, attempt=attempt), "agent1": Action(0.0)},
                [agreement],
            )
            assert not env.errors
        movement = env.views["agent0"].own["progress_m"] - initial
        assert movement < -0.2
        assert env.minimum_gap > 0.15
        assert abs(env.views["agent0"].own["speed_mps"]) < 0.05
        assert env.last_commands["agent0"]["applied"]["speed_mps"] == 0.0
    finally:
        env.close()


@pytest.mark.integration
def test_native_moving_leader_braking_and_perception_loss(tmp_path):
    from cav_recovery.recovery.environment import RecoveryEnv

    spec = two_car_spec()
    spec["horizon_s"] = 24.0
    env = RecoveryEnv(spec, tmp_path / "moving-front", RecoveryContract())
    try:
        env.reset()
        start = env.views["agent1"].own["progress_m"]
        for index in range(90):
            leader = Action(1.0 if index < 20 else 0.0)
            if index == 10:
                env.views["agent1"].scan["missing"] = True
            env.step({"agent0": leader, "agent1": Action(2.0)})
            if index == 10:
                assert env.last_commands["agent1"]["applied"]["speed_mps"] == 0.0
                assert (
                    env.last_commands["agent1"]["guard"]["reason"]
                    == "PERCEPTION_UNAVAILABLE"
                )
            assert not env.errors
        assert env.views["agent1"].own["progress_m"] > start + 0.2
        assert env.minimum_gap > 0.15
        assert abs(env.views["agent1"].own["speed_mps"]) < 0.05
    finally:
        env.close()


@pytest.mark.integration
def test_native_diagnostic_acquisition_preserves_physics_and_local_scope(tmp_path):
    from cav_recovery.recovery.environment import RecoveryEnv
    from cav_recovery.recovery.learning import observation

    trajectories = []
    for enabled in [False, True]:
        env = RecoveryEnv(
            two_car_spec(),
            tmp_path / str(enabled),
            measurement_contract="REC-METRO-1" if enabled else None,
        )
        measured = []
        try:
            env.reset()
            for _ in range(12):
                if enabled:
                    for ego in env.views:
                        assert len(env.views[ego].scan["ranges"]) == 360
                        assert env.views[ego].measurement_contract == "REC-METRO-1"
                        projected = observation(env.views[ego])
                        assert projected["diagnostic"]["ego_id"] == ego
                        assert (
                            projected["graph"]["scope"]
                            == "partial_one_hop_received_reports"
                        )
                env.step({"agent0": Action(0.8), "agent1": Action(0.4)})
                measured.append(
                    {ego: list(v.own["position"]) for ego, v in env.views.items()}
                )
                assert not env.errors
            trajectories.append(measured)
        finally:
            env.close()
    assert trajectories[0] == trajectories[1]
