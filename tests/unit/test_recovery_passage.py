"""Engagements locaux vérifiables et géométrie sélectionnée du garde."""

import copy

import pytest
from shapely.geometry import box

from cav_recovery.recovery.contracts import Action, RecoveryContract
from cav_recovery.metadrive.observation import rectangle
from cav_recovery.recovery.passage_policy import Passage, PassagePolicy
from cav_recovery.recovery.protocol import Endpoint, validate
from cav_recovery.recovery.safety import LocalGuard
from cav_recovery.recovery.verified_passage_policy import (
    FreshPartnerPolicy, RevalidatedPassagePolicy, VerifiedPassagePolicy,
    PrioritizedPassagePolicy, ContinuousPassagePolicy, OfferedPassage,
)
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


def test_former_partner_out_of_range_does_not_block_a_fresh_agreement():
    policy = FreshPartnerPolicy("V4", Endpoint("V4"), RecoveryContract())
    policy.attempt = "V4:0.0"
    policy.started = 0.0
    policy.anchor = [0.0, 0.0]
    policy.action = Action(-0.6)
    policy.partners = {"V0", "V3"}
    local = view()
    local.at_s = 1.0
    local.scan["at_s"] = 1.0
    peer = Endpoint("V0")
    local.messages = [
        peer.send(
            "STATE", "state", [8.0, 0.0], 1.0,
            payload=dict(speed_mps=0.0, heading_rad=0.0,
                         length_m=4.515, width_m=1.852, progress_m=0.0),
        ),
        peer.send(
            "AGREE_HOLD", policy.attempt, [8.0, 0.0], 1.0,
            target="V4", payload={"until_s": 1.6},
        ),
    ]
    action = policy.decide(local, Action(0.0), LocalGuard())
    assert policy.partners == {"V0"}
    assert action.reason == "cooperative_release"
    receipt = LocalGuard().check(local, action)
    assert receipt["accepted"], receipt


def test_versioned_passage_report_is_targeted_fresh_and_has_no_oracle_fields():
    beneficiary = Endpoint("V0")
    yielder = Endpoint("V3")
    offer = yielder.send(
        "PASSAGE_OFFER", "V3:12.0", [0.0, 0.0], 12.0,
        target="V0", payload={"space_wkt": box(0, 0, 2, 2).wkt,
                              "initial_progress_m": 5.0,
                              "purpose": "NOMINAL_PASSAGE"},
    )
    assert offer["version"] == "REC-V2V-3"
    assert beneficiary.receive(offer, 12.2)
    assert not Endpoint("V7").receive(offer, 12.2)
    report = beneficiary.send(
        "PASSAGE_REPORT", offer["attempt"], [0.5, 0.5], 12.2,
        target="V3", payload={
            "phase": "ENTERED", "resource_status": "CLEAR",
            "guard_reason": "LOCAL_ENVELOPE",
            "policy_reason": "own_route", "required_wkt": box(0, 0, 1, 1).wkt,
            "progress_m": 6.0, "purpose": "NOMINAL_PASSAGE",
        },
    )
    assert yielder.receive(report, 12.4)
    forbidden = copy.deepcopy(report)
    forbidden["payload"]["oracle_blocker"] = "V7"
    with pytest.raises(ValueError):
        validate(forbidden, 12.4)
    with pytest.raises(ValueError):
        validate(report, 13.0)


def test_verified_completion_requires_both_local_exit_and_fresh_peer_report():
    policy = VerifiedPassagePolicy("V3", Endpoint("V3"), RecoveryContract())
    policy.lease_attempt = "V3:12.0"
    passage = Passage("V0", box(0, 0, 2, 2), 0.0, 7.0, 20.0,
                      entered=True, crossed=True)
    local = view()
    local.at_s = 20.0
    assert not policy.passage_complete(local, passage)
    report = Endpoint("V0").send(
        "PASSAGE_REPORT", policy.lease_attempt, [9.0, 0.0], 20.0,
        target="V3", payload={
            "phase": "CLEARED", "resource_status": "CLEAR",
            "guard_reason": "LOCAL_ENVELOPE",
            "policy_reason": "own_route", "required_wkt": box(0, 0, 1, 1).wkt,
            "progress_m": 7.0, "purpose": "NOMINAL_PASSAGE",
        },
    )
    local.messages = [report]
    assert policy.passage_complete(local, passage)
    local.at_s = 20.3
    assert not policy.passage_complete(local, passage)


def test_beneficiary_reports_guard_and_actual_resource_exit_separately():
    policy = VerifiedPassagePolicy("V0", Endpoint("V0"), RecoveryContract())
    local = view()
    offer = Endpoint("V3").send(
        "PASSAGE_OFFER", "V3:12.0", [0.0, 0.0], 12.0,
        target="V0", payload={"space_wkt": box(-1, -1, 1, 1).wkt,
                              "initial_progress_m": 0.0,
                              "purpose": "NOMINAL_PASSAGE"},
    )
    local.at_s = 12.0
    local.scan["at_s"] = 12.0
    peer = Endpoint("V3")
    yielder_state = peer.send(
        "STATE", "state", [8.0, 0.0], 12.0,
        payload=dict(speed_mps=0.0, heading_rad=0.0,
                     length_m=4.515, width_m=1.852, progress_m=1.0),
    )
    local.messages = [offer, yielder_state]
    policy._report_offers(local, Action(0.4), Action(0.0, reason="policy_hold"), LocalGuard())
    first = policy.outbox[-1]
    assert first["kind"] == "PASSAGE_REPORT"
    assert first["payload"]["phase"] == "ADMISSIBLE"
    assert first["payload"]["policy_reason"] == "policy_hold"
    local.at_s = 12.2
    local.scan["at_s"] = 12.2
    local.messages = [copy.deepcopy(offer), copy.deepcopy(yielder_state)]
    local.messages[0]["sent_s"] = 12.2
    local.messages[0]["expires_s"] = 12.8
    local.messages[1]["sent_s"] = 12.2
    local.messages[1]["expires_s"] = 12.8
    local.own["position"] = [8.0, 0.0]
    local.own["progress_m"] = 8.0
    policy._report_offers(local, Action(0.4), Action(0.4, reason="own_route"), LocalGuard())
    assert policy.outbox[-1]["payload"]["phase"] == "CLEARED"


def test_verified_lease_timeout_is_a_cancellation_not_a_passage_success():
    policy = VerifiedPassagePolicy("V3", Endpoint("V3"), RecoveryContract())
    policy.passages = {"V0": Passage("V0", box(-1, -1, 1, 1), 0.0, 0.0, 10.0)}
    policy.lease_started_s = 10.0
    policy.lease_until_s = 30.0
    policy.lease_attempt = "V3:9.0"
    local = view()
    local.at_s = 30.0
    assert policy._passage_action(local) is None
    assert policy.outbox[-1]["kind"] == "CANCEL"
    assert policy.outbox[-1]["payload"]["reason"] == "passage_unconfirmed_timeout"


def test_obsolete_unexecuted_reversal_is_cancelled_only_when_nominal_is_safe():
    policy = RevalidatedPassagePolicy("V4", Endpoint("V4"), RecoveryContract())
    policy.attempt = "V4:1.0"
    policy.started = 1.0
    policy.action = Action(-0.6)
    policy.last_request_s = 1.0
    local = view()
    local.at_s = 4.0
    local.scan["at_s"] = 4.0
    nominal = Action(0.4, reason="own_route")
    assert policy.decide(local, nominal, LocalGuard()) == nominal
    assert policy.outbox[-1]["kind"] == "CANCEL"
    assert policy.outbox[-1]["payload"]["reason"] == "request_expired_nominal_safe"
    assert policy.attempt is None

    unsafe = RevalidatedPassagePolicy("V4", Endpoint("V4"), RecoveryContract())
    unsafe.attempt = "V4:1.0"
    unsafe.started = 1.0
    unsafe.anchor = [0.0, 0.0]
    unsafe.action = Action(-0.6)
    unsafe.last_request_s = 1.0
    local.scan["missing"] = True
    unsafe.decide(local, nominal, LocalGuard())
    assert unsafe.attempt is not None
    assert not any(message["kind"] == "CANCEL" for message in unsafe.outbox)


def test_verified_contract_does_not_inherit_the_negative_revalidation_factor():
    policy = VerifiedPassagePolicy("V4", Endpoint("V4"), RecoveryContract())
    policy.attempt = "V4:1.0"
    policy.started = 1.0
    policy.anchor = [0.0, 0.0]
    policy.action = Action(-0.6)
    local = view()
    local.at_s = 4.0
    local.scan["at_s"] = 4.0
    policy.decide(local, Action(0.4, reason="own_route"), LocalGuard())
    assert policy.attempt == "V4:1.0"
    assert not any(message["kind"] == "CANCEL" for message in policy.outbox)


def test_upstream_passage_priority_lets_existing_hold_expire_before_motion():
    policy = PrioritizedPassagePolicy("V0", Endpoint("V0"), RecoveryContract())
    local = view()
    local.at_s = 20.0
    local.scan["at_s"] = 20.0
    local.own_commitment_until_s = 20.6
    peer = Endpoint("V4")
    offer = Endpoint("V3").send(
        "PASSAGE_OFFER", "V3:19.0", [8.0, 0.0], 20.0,
        target="V0", payload={"space_wkt": box(-1, -1, 1, 1).wkt,
                              "initial_progress_m": 0.0,
                              "purpose": "NOMINAL_PASSAGE"},
    )
    intent = peer.send(
        "INTENT", "V4:19.0", [9.5, 0.0], 20.0,
        payload={"speed_mps": -0.6, "steering": 0.0, "distance_m": 2.0},
    )
    state = peer.send(
        "STATE", "state", [9.5, 0.0], 20.0,
        payload={"speed_mps": 0.0, "heading_rad": 0.0,
                 "length_m": 4.515, "width_m": 1.852,
                 "progress_m": 0.0},
    )
    local.messages = [offer, intent, state]
    nominal = Action(0.4, reason="own_route")
    guard = LocalGuard()
    probe = copy.copy(local)
    probe.own_commitment_until_s = local.at_s
    assert guard.check(probe, nominal)["accepted"]
    assert policy.decide(local, nominal, guard).reason == "honour_hold"
    coordination = [
        message["kind"] for message in policy.outbox
        if message["kind"] in {"REFUSE", "AGREE_HOLD"}
    ]
    assert coordination == ["REFUSE"]

    local.at_s = 20.8
    local.scan["at_s"] = 20.8
    local.messages = []
    local.own_commitment_until_s = 20.6
    assert policy.decide(local, nominal, guard) == nominal

    moving = PrioritizedPassagePolicy("V0", Endpoint("V0"), RecoveryContract())
    local.at_s = 21.0
    local.scan["at_s"] = 21.0
    local.own_commitment_until_s = 21.6
    current_offer = copy.deepcopy(offer)
    current_offer["sent_s"] = 21.0
    current_offer["expires_s"] = 21.6
    current_intent = copy.deepcopy(intent)
    current_intent["sent_s"] = 21.0
    current_intent["expires_s"] = 21.6
    moving_state = copy.deepcopy(state)
    moving_state["sent_s"] = 21.0
    moving_state["expires_s"] = 21.6
    moving_state["payload"]["speed_mps"] = -0.2
    local.messages = [current_offer, current_intent, moving_state]
    assert moving.decide(local, nominal, guard).reason == "relevant_neighbor_hold"
    assert any(message["kind"] == "AGREE_HOLD" for message in moving.outbox)


def test_in_flight_passage_does_not_renew_another_reverse_agreement():
    policy = ContinuousPassagePolicy("V0", Endpoint("V0"), RecoveryContract())
    local = view()
    local.at_s = 20.0
    local.scan["at_s"] = 20.0
    local.own_commitment_until_s = 20.6
    offer = Endpoint("V3").send(
        "PASSAGE_OFFER", "V3:19.0", [8.0, 0.0], 20.0,
        target="V0", payload={"space_wkt": box(-1, -1, 1, 1).wkt,
                              "initial_progress_m": 0.0,
                              "purpose": "NOMINAL_PASSAGE"},
    )
    policy.offers[("V3", "V3:19.0")] = OfferedPassage(
        box(-1, -1, 1, 1), 0.0, "NOMINAL_PASSAGE", entered=True,
    )
    intent = Endpoint("V4").send(
        "INTENT", "V4:19.0", [9.5, 0.0], 20.0,
        payload={"speed_mps": -0.6, "steering": 0.0, "distance_m": 2.0},
    )
    local.messages = [offer, intent]
    action = policy.decide(local, Action(0.4, reason="own_route"), LocalGuard())
    assert action.reason == "honour_hold"
    assert any(message["kind"] == "REFUSE" and
               message["payload"]["reason"] == "in_flight_passage_priority"
               for message in policy.outbox)
    assert not any(message["kind"] == "AGREE_HOLD" for message in policy.outbox)


def test_version_three_request_distinguishes_reversal_from_mission_passage():
    request = Endpoint("V3").send(
        "CLEAR_REQUEST", "V3:109.8", [0.0, 0.0], 109.8,
        target="V7", payload={
            "space_wkt": box(-2, -1, 0, 1).wkt,
            "progress_m": 59.4,
            "purpose": "REVERSE_OPTION",
            "movement_id": "a0_a3_l0",
            "required_wkt": box(-0.5, -1, 0, 1).wkt,
            "guard_reason": "RAY_OCCUPANCY_OR_UNKNOWN",
        },
    )
    assert request["version"] == "REC-V2V-3"
    assert validate(request, 109.8)
    invalid = copy.deepcopy(request)
    invalid["payload"]["purpose"] = "ORACLE_RECOVERY"
    with pytest.raises(ValueError):
        validate(invalid, 109.8)

    policy = VerifiedPassagePolicy("V7", Endpoint("V7"), RecoveryContract())
    passage = Passage("V3", box(-2, -1, 0, 1), 59.4, 59.4, 109.8,
                      purpose="REVERSE_OPTION", entered=True)
    outside = rectangle((8, 0), 0.0, 4.515, 1.852)
    assert policy.passage_crossed(passage, outside, 59.1, 4.515)
    passage.purpose = "NOMINAL_PASSAGE"
    assert not policy.passage_crossed(passage, outside, 59.1, 4.515)


def test_incomplete_reverse_option_is_reported_as_cancellation():
    policy = VerifiedPassagePolicy("V7", Endpoint("V7"), RecoveryContract())
    policy.attempt = "V7:10.0"
    policy.action = Action(-0.6)
    policy.requesters = {"V3": (box(-2, -1, 0, 1), 59.4)}
    local = view()
    policy.report_option_end(local, "bounded_local_option_finished")
    assert policy.outbox[-1]["kind"] == "CANCEL"
    assert policy.outbox[-1]["payload"]["reason"] == "option_ended_without_clearance"


def test_cancelled_option_cannot_open_a_passage_lease():
    policy = VerifiedPassagePolicy("V7", Endpoint("V7"), RecoveryContract())
    policy.attempt = "V7:1.0"
    policy.started = 1.0
    policy.anchor = [0.0, 0.0]
    policy.action = Action(-0.6)
    policy.requesters = {"V3": (box(-2, -1, 0, 1), 59.4)}
    policy.release_space = box(10, 10, 11, 11)
    local = view()
    local.at_s = 42.0
    local.scan["at_s"] = 42.0
    policy.decide(local, Action(0.0), LocalGuard())
    assert policy.passages == {}
    assert any(message["kind"] == "CANCEL" for message in policy.outbox)
    assert not any(message["kind"] == "PASSAGE_OFFER" for message in policy.outbox)


def test_request_contract_preserves_v2_and_explains_v3_geometry():
    local = view()
    space = box(-2, -1, 0, 1)
    old = PassagePolicy("V3", Endpoint("V3"), RecoveryContract())
    old.request_clearance(
        local, space, target="V7", purpose="REVERSE_OPTION", action=Action(-0.6),
    )
    assert old.outbox[-1]["version"] == "REC-V2V-2"
    assert set(old.outbox[-1]["payload"]) == {"space_wkt", "progress_m"}

    new = VerifiedPassagePolicy("V3", Endpoint("V3"), RecoveryContract())
    new._request_guard = LocalGuard()
    local.route = type("OwnRoute", (), {"agent": {"movement_id": "a0_a3_l0"}})()
    new.request_clearance(
        local, space, target="V7", purpose="REVERSE_OPTION", action=Action(-0.6),
    )
    packet = new.outbox[-1]
    assert packet["version"] == "REC-V2V-3"
    assert packet["payload"]["purpose"] == "REVERSE_OPTION"
    assert packet["attempt"] == "V3:clear:REVERSE_OPTION"
    assert packet["payload"]["movement_id"] == "a0_a3_l0"
    assert packet["payload"]["required_wkt"]
