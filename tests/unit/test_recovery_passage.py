"""Engagements locaux vérifiables et géométrie sélectionnée du garde."""

from shapely.geometry import box

from cav_recovery.recovery.contracts import Action, RecoveryContract
from cav_recovery.recovery.passage_policy import Passage, PassagePolicy
from cav_recovery.recovery.protocol import Endpoint
from cav_recovery.recovery.safety import LocalGuard
from tests.unit.test_recovery_prerequisites import view


def prepared_policy():
    policy = PassagePolicy("V0", Endpoint("V0"), RecoveryContract())
    policy.passages = {"V3": Passage("V3", box(-1, -1, 1, 1), 0.0, 0.0, 10.0)}
    policy.lease_started_s = 10.0
    policy.lease_until_s = 30.0
    policy.lease_attempt = "V0:9.0"
    return policy


def received_state(now, position, progress):
    return Endpoint("V3").send(
        "STATE",
        "state",
        position,
        now,
        payload=dict(
            speed_mps=1.0,
            heading_rad=0.0,
            length_m=4.515,
            width_m=1.852,
            progress_m=progress,
        ),
    )


def test_yielder_waits_for_fresh_beneficiary_passage_and_then_releases():
    policy = prepared_policy()
    local = view()
    local.at_s = 20.0
    local.messages = [received_state(20.0, [0.0, 0.0], 3.0)]
    assert policy._passage_action(local).reason == "passage_commitment"
    assert policy.outbox[-1]["kind"] == "AGREE_HOLD"
    assert policy.outbox[-1]["target"] == "V3"
    local.at_s = 20.2
    local.messages = [received_state(20.2, [10.0, 0.0], 7.0)]
    assert policy._passage_action(local) is None
    assert not policy.passages
    assert policy.outbox[-1]["payload"]["reason"] == "beneficiaries_passed"


def test_hold_agreement_matches_the_beneficiary_current_intent():
    policy = prepared_policy()
    local = view()
    local.at_s = 20.0
    peer = Endpoint("V3")
    local.messages = [
        received_state(20.0, [0.0, 0.0], 1.0),
        peer.send(
            "INTENT",
            "V3:19.8",
            [0.0, 0.0],
            20.0,
            payload=dict(speed_mps=-0.6, steering=0.0, distance_m=2.0),
        ),
    ]
    assert policy._passage_action(local).reason == "passage_commitment"
    assert policy.outbox[-1]["attempt"] == "V3:19.8"


def test_stale_messages_do_not_prove_passage_and_lease_has_a_bound():
    policy = prepared_policy()
    local = view()
    local.at_s = 20.0
    local.messages = [received_state(19.0, [20.0, 0.0], 20.0)]
    assert policy._passage_action(local).reason == "passage_commitment"
    local.at_s = 30.0
    local.messages = []
    assert policy._passage_action(local) is None
    assert policy.outbox[-1]["payload"]["reason"] == "bounded_passage_timeout"


def test_progress_outside_requested_resource_does_not_prove_crossing():
    policy = prepared_policy()
    local = view()
    local.at_s = 20.0
    local.messages = [received_state(20.0, [10.0, 0.0], 8.0)]
    assert policy._passage_action(local).reason == "passage_commitment"
    assert not policy.passages["V3"].entered


def test_progress_can_renew_once_but_not_beyond_absolute_limit():
    policy = prepared_policy()
    local = view()
    local.at_s = 29.8
    local.messages = [received_state(29.8, [0.0, 0.0], 0.5)]
    assert policy._passage_action(local).reason == "passage_commitment"
    local.at_s = 30.0
    local.messages = [received_state(30.0, [0.1, 0.0], 0.8)]
    assert policy._passage_action(local).reason == "passage_commitment"
    assert policy.lease_until_s == 50.0
    local.at_s = 50.0
    local.messages = [received_state(50.0, [0.2, 0.0], 1.0)]
    assert policy._passage_action(local) is None


def test_reversal_end_does_not_announce_passage_before_beneficiary_crosses():
    policy = prepared_policy()
    local = view()
    local.at_s = 12.0
    policy.action = Action(-0.6)
    policy.requesters = {"V3": (box(-1, -1, 1, 1), 0.0)}
    policy.report_option_end(local, "announced_space_clear")
    assert policy.outbox == []
    policy.action = Action(0.4)
    policy.report_option_end(local, "bounded_local_option_finished")
    assert policy.outbox[-1]["kind"] == "DONE"


def test_guard_uses_requested_sweep_without_changing_default():
    calls = []

    def sweep(own, action, distance):
        calls.append((action.speed_mps, distance))
        return box(-2.3, -1.0, 2.3, 1.0)

    receipt = LocalGuard(sweep_predictor=sweep).check(view(), Action(0.4))
    assert receipt["accepted"] and calls
