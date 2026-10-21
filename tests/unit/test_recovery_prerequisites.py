"""Frontières locales, engagements et mesures expérimentales de récupération."""

import copy

import pytest

pytest.importorskip("shapely")
from shapely.geometry import box
from cav_recovery.recovery.contracts import Action, RecoveryContract
from cav_recovery.recovery.protocol import Endpoint, deliver, validate
from cav_recovery.recovery.safety import LocalGuard, LocalView
from cav_recovery.recovery.metrology import StationaryMeasurement
from cav_recovery.recovery.baseline import DeterministicPolicy


def view():
    own = dict(
        position=[0.0, 0.0],
        heading_rad=0.0,
        speed_mps=0.0,
        length_m=4.515,
        width_m=1.852,
        wheelbase_m=2.7,
        max_steering_rad=0.7,
        progress_m=0.0,
        desired_speed_mps=2.0,
    )
    return LocalView(
        "V0",
        0.0,
        own,
        dict(at_s=0.0, ranges=[1.0] * 360, range_m=30.0, full_circle=True),
        [],
        box(-50, -10, 50, 10),
        None,
    )


def state(endpoint, xy, now=0.0):
    return endpoint.send(
        "STATE",
        "state",
        xy,
        now,
        payload=dict(
            speed_mps=0.0,
            heading_rad=0.0,
            length_m=4.515,
            width_m=1.852,
            progress_m=0.0,
        ),
    )


def test_transport_range_target_sequence_and_expiry():
    a, b, c = Endpoint("A"), Endpoint("B"), Endpoint("C")
    positions = dict(A=[0.0, 0.0], B=[5.0, 0.0], C=[40.0, 0.0])
    message = state(a, positions["A"])
    events = deliver([message], dict(A=a, B=b, C=c), positions, 0.2, 30.0)
    assert [e["recipient"] for e in events] == ["B"]
    assert not b.receive(message, 0.2)
    assert not b.view(0.8)
    with pytest.raises(ValueError):
        validate(message, 1.0)
    with pytest.raises(ValueError):
        validate(message, -0.1)
    bad = copy.deepcopy(message)
    bad["payload"]["oracle_group"] = ["B", "C"]
    with pytest.raises(ValueError):
        validate(bad, 0.2)


def test_commitments_are_sender_owned_bounded_and_cancellable():
    a, b = Endpoint("A"), Endpoint("B")
    m = b.send(
        "AGREE_HOLD", "A:1", [4.0, 0.0], 0.0, target="A", payload={"until_s": 0.6}
    )
    assert a.receive(m, 0.2)
    bad = copy.deepcopy(m)
    bad["payload"]["until_s"] = 20.0
    with pytest.raises(ValueError):
        validate(bad, 0.2)
    cancel = b.send(
        "CANCEL", "A:1", [4.0, 0.0], 0.2, target="A", payload={"reason": "cancelled"}
    )
    assert a.receive(cancel, 0.4)
    assert not any(r["kind"] == "AGREE_HOLD" for r in a.view(0.4))


@pytest.mark.parametrize("speed", [0.4, 1.0, 2.0, -0.6])
def test_clear_road_admits_bounded_motion(speed):
    result = LocalGuard().check(view(), Action(speed))
    assert result["accepted"], result


@pytest.mark.parametrize(
    "change,reason",
    [
        ("missing", "PERCEPTION_UNAVAILABLE"),
        ("stale", "PERCEPTION_UNAVAILABLE"),
        ("sparse", "PERCEPTION_COVERAGE"),
        ("half", "PERCEPTION_COVERAGE"),
        ("budget", "REVERSE_BUDGET"),
        ("limit", "ACTION_LIMIT"),
    ],
)
def test_safety_fails_closed_on_missing_information(change, reason):
    v = view()
    speed = -0.6
    if change == "missing":
        v.scan["missing"] = True
    if change == "stale":
        v.at_s = 1.0
    if change == "sparse":
        v.scan["ranges"] = [1.0] * 16
    if change == "half":
        v.scan["full_circle"] = False
    if change == "budget":
        v.reverse_used_m = 8.0
    if change == "limit":
        speed = -1.0
    assert LocalGuard().check(v, Action(speed))["reason"] == reason


def test_anonymous_rear_occupancy_blocks_without_identity_or_message():
    v = view()
    # Obstacle mesuré derrière le CAV, sans lui inventer une identité V2V.
    for i in range(150, 211):
        v.scan["ranges"][i] = 2.6 / 30.0
    assert not LocalGuard().check(v, Action(-0.6))["accepted"]


def test_rear_action_requires_agreement_even_if_actor_bypasses_baseline():
    v = view()
    other = Endpoint("V1")
    v.messages = [state(other, [0.0, 7.0])]
    assert (
        LocalGuard().check(v, Action(-0.6))["reason"] == "COOPERATIVE_AGREEMENT_MISSING"
    )
    agreement = other.send(
        "AGREE_HOLD", "V0:1", [0.0, 7.0], 0.0, target="V0", payload={"until_s": 0.6}
    )
    v.messages.append(agreement)
    assert LocalGuard().check(v, Action(-0.6, attempt="V0:1"))["accepted"]
    assert not LocalGuard().check(v, Action(-0.6, attempt="wrong"))["accepted"]


def measurement(t, progress=1.0, speed=-0.003, accepted=0.0):
    return dict(
        decision_at_s=t,
        ego=dict(
            position_xy_m=[progress, 0.0],
            mission_progress_m=progress,
            signed_speed_mps=speed,
        ),
        own_controls=dict(accepted_speed_mps=accepted, requested_speed_mps=2.0),
    )


def test_stationary_adapter_keeps_raw_and_never_edits_position_or_input():
    adapter = StationaryMeasurement()
    raw = measurement(0.0)
    saved = copy.deepcopy(raw)
    adapted, receipt = adapter.adapt(raw)
    assert raw == saved and receipt["raw_speed_mps"] == -0.003
    assert adapted["ego"]["signed_speed_mps"] == 0.0
    assert adapted["ego"]["position_xy_m"] == raw["ego"]["position_xy_m"]
    reverse, _ = adapter.adapt(measurement(0.2, 0.99, -0.6, accepted=-0.6))
    assert reverse["ego"]["signed_speed_mps"] == -0.6


def test_stationary_adapter_does_not_ratchet_away_slow_cumulative_reverse():
    adapter = StationaryMeasurement()
    result = []
    for step in range(61):
        t = step * 0.2
        adapted, receipt = adapter.adapt(measurement(t, 1.0 - 0.005 * t, -0.005))
        result.append(receipt)
    assert not any(r["bounded_stationary_interval"] for r in result[20:])
    assert adapted["ego"]["signed_speed_mps"] < 0
    assert result[-1]["used_progress_m"] < result[20]["used_progress_m"]


def test_policy_honours_commitment_after_intent_disappears():
    c = RecoveryContract()
    local = view()
    local.ego = "V1"
    endpoint = Endpoint("V1")
    policy = DeterministicPolicy("V1", endpoint, c)
    other = Endpoint("V0")
    local.messages = [
        other.send(
            "INTENT",
            "V0:0",
            [5.0, 0.0],
            0.0,
            payload=dict(speed_mps=-0.6, steering=0.0, distance_m=4.0),
        )
    ]
    action = policy.decide(local, Action(1.0), LocalGuard())
    assert action.speed_mps == 0 and policy.outbox[0]["kind"] == "AGREE_HOLD"
    local.at_s = 0.2
    local.messages = []
    assert policy.decide(local, Action(1.0), LocalGuard()).speed_mps == 0
    local.at_s = 1.0
    local.scan["at_s"] = 1.0
    assert policy.decide(local, Action(1.0), LocalGuard()).speed_mps == 1.0


def test_actuator_boundary_rejects_violation_of_own_hold_promise():
    v = view()
    v.own_commitment_until_s = 0.6
    assert LocalGuard().check(v, Action(0.4))["reason"] == "OWN_HOLD_COMMITMENT"
    assert LocalGuard().check(v, Action(0.0))["accepted"]


def test_learning_boundary_never_copies_privileged_dynamic_attributes():
    from cav_recovery.recovery.learning import observation, action_mask

    v = view()
    v.oracle_group = ["secret"]
    v.future_label = True
    projected = observation(v)
    assert "secret" not in repr(projected) and "future_label" not in projected
    options, mask = action_mask(v, LocalGuard())
    assert len(options) == len(mask) == 8 and mask[0]


def test_composed_diagnostics_keep_raw_and_measurement_variant_separate():
    from cav_recovery.recovery.diagnostics import RecoveryDiagnostics
    from tests.support.d01_fixtures import packet

    service = RecoveryDiagnostics(measurement_contract="REC-METRO-1")
    for i in range(3):
        p = packet(i * 0.2, speed=-0.003)
        old = copy.deepcopy(p)
        output = service.observe({"V0": p}, round(i * 0.2, 7))["V0"]
        assert p == old
        assert output["historical_raw"]["reason"] == "NON_NOMINAL_REVERSE"
        assert output["measurement_receipt"]["used_speed_mps"] == 0.0
        assert output["graph"]["scope"] == "partial_one_hop_received_reports"
        assert output["measurement_contract"] == "REC-METRO-1"


@pytest.mark.parametrize("fixture_name", ["fixture_a", "fixture_b", "fixture_c"])
def test_measurement_variant_preserves_analytical_contract_cases(fixture_name):
    from tests.support import d01_fixtures
    from cav_recovery.d01.observer import EgoObserver

    normal = EgoObserver()
    variant = EgoObserver()
    adapter = StationaryMeasurement()
    for packet in getattr(d01_fixtures, fixture_name)():
        raw = normal.update(packet)
        used, _ = adapter.adapt(packet)
        changed = variant.update(used)
        assert raw["status"] == changed["status"]
        assert raw["emitted_at_s"] == changed["emitted_at_s"]


def test_physical_success_requires_progress_and_observed_followup_not_graph_disappearance():
    from cav_recovery.recovery.evaluation import physical_outcome

    contract = dict(recovery_window_s=2.0, relapse_followup_s=3.0, effect_window_s=1.0)
    frames = [
        dict(time_s=float(t), states={"V0": dict(progress_m=float(t))})
        for t in range(7)
    ]
    metadata = dict(vehicles={"V0": dict(length_m=1.0)})
    result = physical_outcome(
        frames, {"events": []}, metadata, {}, {"V0"}, 0.0, contract
    )
    assert result["success"] and result["t_conf_s"] == 2.0
    censored = physical_outcome(
        frames[:3], {"events": []}, metadata, {}, {"V0"}, 0.0, contract
    )
    assert not censored["success"] and censored["status"] == "RIGHT_CENSORED_FOLLOWUP"
    stopped = [
        dict(time_s=f["time_s"], states={"V0": dict(progress_m=0.0)}) for f in frames
    ]
    assert not physical_outcome(
        stopped, {"events": []}, metadata, {}, {"V0"}, 0.0, contract
    )["success"]
    assert not physical_outcome(
        frames, {"events": []}, metadata, {}, set(), 0.0, contract
    )["success"]


def test_explicit_relapse_invalidates_otherwise_progressing_trace():
    from cav_recovery.recovery.evaluation import physical_outcome

    frames = [
        dict(time_s=float(t), states={"V0": dict(progress_m=float(t))})
        for t in range(7)
    ]
    annotation = {
        "events": [
            dict(requester="V0", effect_intervals=[dict(start_s=3.0, end_s=5.0)])
        ]
    }
    result = physical_outcome(
        frames,
        annotation,
        {"vehicles": {"V0": {"length_m": 1.0}}},
        {},
        {"V0"},
        0.0,
        dict(recovery_window_s=2.0, relapse_followup_s=3.0, effect_window_s=1.0),
    )
    assert not result["success"] and result["relapse"]


def test_frozen_future_partitions_keep_parent_and_lineage_variants_together():
    import json
    from pathlib import Path
    from cav_recovery.dataset.contract import digest
    from cav_recovery.dataset.generation import candidate

    path = Path(__file__).resolve().parents[2] / "configs/recovery/partitions-v1.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    parents, lineages, seeds = {}, {}, set()
    counts = dict(train=0, validation=0, test=0)
    for group in manifest["groups"]:
        partition = group["partition"]
        assert group["parent_id"] not in parents
        assert group["lineage_group"] not in lineages
        parents[group["parent_id"]] = partition
        lineages[group["lineage_group"]] = partition
        assert (group["family"] == "asymmetric") == (partition == "test")
        assert len(group["recipes"]) == 4
        for recipe in group["recipes"]:
            spec = candidate(recipe["index"], phase=recipe["phase"])
            assert digest(spec) == recipe["configuration_sha256"]
            assert spec["parent_id"] == group["parent_id"]
            assert spec["lineage_group"] == group["lineage_group"]
            assert recipe["seed"] not in seeds
            seeds.add(recipe["seed"])
            counts[partition] += 1
    assert counts == dict(train=32, validation=16, test=16)


def test_local_release_options_are_available_to_the_future_actor():
    from cav_recovery.recovery.baseline import gentle_forward_options
    from cav_recovery.recovery.learning import action_mask

    local = view()
    policy = DeterministicPolicy("V0", Endpoint("V0"), RecoveryContract())
    policy.stopped_since = 0.0
    local.at_s = 36.0
    local.scan["at_s"] = 36.0
    # Une commande hors domaine est refusée ; les avances douces restent libres.
    action = policy.decide(local, Action(3.0), LocalGuard())
    options, mask = action_mask(local, LocalGuard())
    assert action in gentle_forward_options()
    assert any(
        a.speed_mps == action.speed_mps and a.steering == action.steering and allowed
        for a, allowed in zip(options, mask)
    )


def test_arrival_is_terminal_without_inventing_future_validation_time():
    from cav_recovery.recovery.evaluation import physical_outcome

    frames = [
        dict(time_s=float(t), states={"V0": dict(progress_m=float(t))} if t < 3 else {})
        for t in range(4)
    ]
    result = physical_outcome(
        frames,
        {"events": []},
        {"vehicles": {"V0": {"length_m": 1.0}}},
        {"V0": 3.0},
        {"V0"},
        0.0,
        dict(recovery_window_s=2.0, relapse_followup_s=30.0, effect_window_s=1.0),
    )
    assert result["success"] and result["t_final_validation_s"] == 3.0


def test_contract_cannot_silently_change_metrology_or_native_cadence():
    with pytest.raises(ValueError):
        StationaryMeasurement(speed_error_mps=0.1)
    with pytest.raises(ValueError):
        RecoveryContract(step_s=0.1)


@pytest.mark.parametrize(
    "field,value", [("width_m", -1.0), ("length_m", 0.0), ("target", 42)]
)
def test_protocol_rejects_invalid_body_and_target(field, value):
    message = state(Endpoint("A"), [0.0, 0.0])
    if field == "target":
        message[field] = value
    else:
        message["payload"][field] = value
    with pytest.raises(ValueError):
        validate(message, 0.0)


@pytest.mark.parametrize("speed,target", [(1.0, -0.6), (-0.5, 0.4)])
def test_opposite_direction_requires_braking_before_new_sweep(speed, target):
    local = view()
    local.own["speed_mps"] = speed
    action, receipt = LocalGuard().limit(local, Action(target))
    assert action.speed_mps == 0.0
    assert receipt["reason"] == "SIGN_CHANGE_REQUIRES_STOP"
