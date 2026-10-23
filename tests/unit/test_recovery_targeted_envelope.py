"""La demande géométrique P4 reste locale et ne change pas le garde."""

from shapely import wkt
from shapely.geometry import box

from cav_recovery.recovery.contracts import Action, RecoveryContract
from cav_recovery.recovery.protocol import Endpoint
from cav_recovery.recovery.safety import LocalGuard
from cav_recovery.recovery.verified_passage_policy import GuardEnvelopePassagePolicy
from tests.unit.test_recovery_prerequisites import view


def test_nominal_request_includes_next_sweep_and_received_body_reserve():
    contract = RecoveryContract()
    policy = GuardEnvelopePassagePolicy("V0", Endpoint("V0"), contract)
    policy._request_guard = LocalGuard(contract)
    local = view()
    local.at_s = 10.0
    local.route = type("OwnRoute", (), {"agent": {"movement_id": "turn_left"}})()
    state = Endpoint("V3").send(
        "STATE", "state", [4.0, 0.0], 10.0,
        payload=dict(speed_mps=0.0, heading_rad=0.0,
                     length_m=4.515, width_m=1.852, progress_m=0.0),
    )
    local.messages = [state]
    original = box(2.0, -1.0, 4.0, 1.0)
    action = Action(0.4, reason="own_route")
    _, _, required = policy._request_guard.action_envelope(local, action)

    policy.request_clearance(
        local, original, target="V3", purpose="NOMINAL_PASSAGE", action=action,
    )
    packet = policy.outbox[-1]
    radius = (contract.margin_m
              + 0.5 * contract.neighbor_acceleration_bound_mps2 * contract.step_s**2)
    expected = original.union(required).buffer(radius)
    assert packet["target"] == "V3"
    assert packet["payload"]["movement_id"] == "turn_left"
    assert wkt.loads(packet["payload"]["required_wkt"]).equals(required)
    assert wkt.loads(packet["payload"]["space_wkt"]).equals(expected)
    assert all("oracle" not in key for key in packet["payload"])


def test_missing_fresh_state_does_not_invent_a_margin_or_a_target():
    policy = GuardEnvelopePassagePolicy("V0", Endpoint("V0"), RecoveryContract())
    policy._request_guard = LocalGuard()
    local = view()
    local.route = type("OwnRoute", (), {"agent": {"movement_id": "straight"}})()
    original = box(2.0, -1.0, 4.0, 1.0)
    policy.request_clearance(
        local, original, target="V3", purpose="NOMINAL_PASSAGE",
        action=Action(0.4),
    )
    assert wkt.loads(policy.outbox[-1]["payload"]["space_wkt"]).equals(original)


def test_moving_neighbors_use_the_same_temporal_reserve_as_the_guard():
    contract = RecoveryContract()
    policy = GuardEnvelopePassagePolicy("V0", Endpoint("V0"), contract)
    policy._request_guard = LocalGuard(contract)
    local = view()
    local.at_s = 10.2
    local.route = type("OwnRoute", (), {"agent": {"movement_id": "left"}})()
    local.messages = [Endpoint("V3").send(
        "STATE", "state", [4.0, 0.0], 10.0,
        payload=dict(speed_mps=2.0, heading_rad=0.0,
                     length_m=4.515, width_m=1.852, progress_m=0.0),
    )]
    original = box(2.0, -1.0, 4.0, 1.0)
    action = Action(0.4, reason="own_route")
    _, _, required = policy._request_guard.action_envelope(local, action)

    policy.request_clearance(
        local, original, target="V3", purpose="NOMINAL_PASSAGE", action=action,
    )

    duration = local.at_s - 10.0 + contract.step_s
    reserve = (contract.margin_m + 2.0 * duration + 2.0**2 / 2
               + 0.5 * contract.neighbor_acceleration_bound_mps2 * duration**2)
    requested = wkt.loads(policy.outbox[-1]["payload"]["space_wkt"])
    expected = original.union(required).buffer(reserve)
    assert requested.symmetric_difference(expected).area < 1e-8


def test_reverse_option_keeps_its_original_resource():
    contract = RecoveryContract()
    policy = GuardEnvelopePassagePolicy("V0", Endpoint("V0"), contract)
    policy._request_guard = LocalGuard(contract)
    local = view()
    local.at_s = 10.2
    local.route = type("OwnRoute", (), {"agent": {"movement_id": "left"}})()
    local.messages = [Endpoint("V3").send(
        "STATE", "state", [4.0, 0.0], 10.0,
        payload=dict(speed_mps=2.0, heading_rad=0.0,
                     length_m=4.515, width_m=1.852, progress_m=0.0),
    )]
    original = box(2.0, -1.0, 4.0, 1.0)

    policy.request_clearance(
        local, original, target="V3", purpose="REVERSE_OPTION",
        action=Action(-0.4, reason="reverse"),
    )

    requested = wkt.loads(policy.outbox[-1]["payload"]["space_wkt"])
    assert requested.equals(original)
