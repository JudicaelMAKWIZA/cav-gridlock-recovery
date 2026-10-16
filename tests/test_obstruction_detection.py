"""Protège les verdicts physiques, la censure et l'absence d'action sur SUMO."""

import copy
import json
from pathlib import Path
import shutil
from unittest.mock import Mock

import pytest

from cav_recovery.obstruction_detection import ObstructionDetector, service_cycles
from cav_recovery import cli
from cav_recovery.simulation import traffic_run


LANES = {"a": {"junction": "J"}, "b": {"junction": "J"}, "out": {"junction": "exit"}}


def rows(a=0, b=0, *, lane="a", speed=0):
    return {"A": {"lane": lane, "speed": speed, "distance": a, "length": 5},
            "B": {"lane": "b", "speed": 0, "distance": b, "length": 5}}


def proofs(age, *, permission="permitted", kind="waits_for", cause="receiving_occupancy"):
    return {item: {"low_progress_age_s": age, "signal_permission": permission,
                   "dependencies": [] if kind is None else [{"dependency_id": f"{kind}|{item}|X",
                                    "edge_type": kind, "evidence": cause, "target": "resource:X"}]}
            for item in ("A", "B")}


def sample(detector, time, *, reading=None, evidence=None, snapshot=None, footprints=None):
    detector.observe(time, rows() if reading is None else reading,
                     {"junction_observations": [], "cycle_candidates": []} if snapshot is None else snapshot,
                     proofs(time) if evidence is None else evidence, LANES, footprints=footprints)
    return detector.snapshot()["episodes"][0] if detector.active else detector.finished[-1]


def test_long_red_wait_is_not_an_obstruction_confirmation():
    detector = ObstructionDetector({"J": 90}, observation_interval_s=1)
    for time in range(1, 20):
        result = sample(detector, time, evidence=proofs(time, permission="prohibited"))
    assert result["state"] == "normal_wait"
    assert result["obstruction_verdict"] == "not_confirmed"
    assert not result["persistent"] and result["cause_status"] == "insufficient"


def test_red_at_the_last_instant_does_not_hide_low_progress_across_a_full_cycle():
    detector = ObstructionDetector({"J": 2}, observation_interval_s=1)
    for time in range(1, 5):
        result = sample(detector, time, evidence=proofs(time, permission="prohibited", kind=None))
    assert result["state"] == "persistent_immobility"
    assert result["obstruction_verdict"] == "not_confirmed"
    assert result["signal_wait_members"] == ["A", "B"]
    assert result["identified_members"] == []
    assert result["actor_scope"] == "co_located_only"


def test_collective_obstruction_does_not_need_a_cycle():
    detector = ObstructionDetector({"J": 2}, observation_interval_s=1)
    for time in range(1, 4):
        result = sample(detector, time)
    assert result["state"] == "persistent_obstruction"
    assert result["obstruction_verdict"] == "verified"
    assert result["members"] == result["persistent_members"] == ["A", "B"]
    assert detector.snapshot()["gridlock"] == "not_evaluated"


def test_occupied_conflict_is_not_automatically_a_causal_arc():
    detector = ObstructionDetector({"J": 2}, observation_interval_s=1)
    snap = {"junction_observations": [{"vehicle_id": "A", "foes": [
        {"vehicle_id": "B", "state": "occupied", "active_constraint": False, "native": []}]}]}
    for time in range(1, 4):
        result = sample(detector, time, evidence=proofs(time, kind=None), snapshot=snap)
    assert result["state"] == "persistent_immobility"
    assert result["cause_status"] == "occupation_only"
    assert result["obstruction_verdict"] == "candidate"
    assert result["evidence"]["qualified_dependencies"] == []


def test_unknown_cause_does_not_erase_observed_immobility():
    detector = ObstructionDetector({"J": 2}, observation_interval_s=1)
    for time in range(1, 4):
        result = sample(detector, time, evidence=proofs(time, kind=None))
    assert result["state"] == "persistent_immobility"
    assert result["obstruction_verdict"] == "indeterminate"
    assert result["physical_resolution"] == "unknown"


def test_follower_at_a_normal_queue_does_not_alone_confirm_operational_obstruction():
    detector = ObstructionDetector({"J": 2})
    result = sample(detector, .5, evidence=proofs(20, kind="leader"))
    assert result["obstruction_verdict"] == "indeterminate"
    assert result["evidence"]["qualified_dependencies"] == []


def test_one_qualified_constraint_does_not_turn_a_nearby_red_wait_into_a_collective_blockage():
    detector = ObstructionDetector({"J": 90})
    evidence = {"A": proofs(1)["A"], "B": proofs(1, permission="prohibited", kind=None)["B"]}
    result = sample(detector, .5, evidence=evidence)
    assert result["state"] == "observed_constraint"
    assert result["identified_members"] == ["A"]
    assert result["signal_wait_members"] == ["B"]
    assert result["actor_scope"] == "individual_identified_vehicle"
    assert result["current_halted_members"] == ["A", "B"]


def test_partial_progress_does_not_release_the_remaining_members():
    detector = ObstructionDetector({"J": 2}, observation_interval_s=1)
    sample(detector, 1)
    result = sample(detector, 2, reading=rows(a=5, speed=2), evidence=proofs(1))
    assert result["state"] == "observed_constraint"
    assert result["advanced_members"] == ["A"]
    assert result["physical_resolution"] == "unknown" and result["remaining_members"] == ["A", "B"]
    assert result["member_observations"]["A"]["first_progress_s"] == 2
    assert result["member_observations"]["A"]["first_progress_m"] == 5


def test_twenty_stopped_and_twelve_past_progresses_are_distinct_at_95():
    detector = ObstructionDetector({"J": 90})
    ids = [f"car{i:02}" for i in range(35)]
    reading = {item: {"lane": "a", "speed": 0, "distance": 0, "length": 5} for item in ids}
    sample(detector, 94.5, reading=reading, evidence={})
    for i, item in enumerate(ids):
        if i < 12:
            reading[item].update(distance=6, speed=3, lane="out" if i < 6 else "a", lane_position=6)
        elif i >= 32:
            reading[item].update(distance=2, speed=1)
    result = sample(detector, 95, reading=reading, evidence={})
    assert result["state"] == "stopped"
    assert len(result["members"]) == 35 and len(result["remaining_members"]) == 29
    assert len(result["advanced_members"]) == 12 and len(result["current_halted_members"]) == 20
    assert set(result["advanced_members"]).isdisjoint(result["current_halted_members"])
    assert all(result["member_observations"][item]["first_progress_s"] == 95 for item in ids[:12])


def test_old_progress_does_not_mask_a_current_obstacle_at_900():
    detector = ObstructionDetector({}, observation_interval_s=0.5)
    sample(detector, .5, evidence=proofs(0, kind=None))
    for tick in range(2, 1801):
        reading = rows(a=5, b=1.7, lane="out", speed=0)
        reading["A"]["lane_position"] = 5
        evidence = {"B": proofs(tick * .5)["B"]}
        result = sample(detector, tick * .5, reading=reading, evidence=evidence)
    assert result["state"] == "observed_constraint" and result["cause_status"] == "qualified"
    assert result["advanced_members"] == ["A"] and result["current_halted_members"] == ["B"]
    assert result["advancing_members"] == [] and result["cycle_s"] is None
    assert result["evidence"]["vehicles"]["B"]["low_progress_age_s"] == 900
    assert result["member_observations"]["A"]["first_progress_s"] == 1
    assert result["member_observations"]["B"]["step_progress_m"] == 0


def test_normal_departure_is_progress_not_recovery():
    detector = ObstructionDetector({"J": 90}, observation_interval_s=1)
    sample(detector, 1, reading={"A": rows()["A"]}, evidence={})
    result = sample(detector, 2, reading={"A": rows(a=5, speed=2)["A"]}, evidence={})
    assert result["state"] == "advancing" and result["advancing_members"] == ["A"]
    assert result["historical_progress_observed"] and result["obstruction_verdict"] == "indeterminate"
    assert result["physical_resolution"] == "unknown"


def test_unknown_cycle_keeps_measured_immobility_without_an_invented_threshold():
    detector = ObstructionDetector({})
    result = sample(detector, 900, evidence=proofs(765, kind=None))
    assert result["state"] == "stopped" and not result["persistent"]
    assert result["evidence"]["vehicles"]["A"]["low_progress_age_s"] == 765
    assert "service_cycle_unknown" in result["uncertainties"]


def test_return_resets_the_progress_date_and_individual_baseline():
    detector = ObstructionDetector({"J": 90}, observation_interval_s=1)
    sample(detector, 1, evidence={})
    moved = rows(a=5, lane="out", speed=1)
    moved["A"]["lane_position"] = 5
    saved = sample(detector, 2, reading=moved, evidence={})
    result = sample(detector, 3, reading=rows(a=100), evidence={})
    detail = result["member_observations"]["A"]
    assert detail["joined_s"] == 1 and detail["measurement_since_s"] == 3
    assert detail["first_progress_s"] is None and detail["baseline_distance_m"] == 100
    result = sample(detector, 4, reading=rows(a=105), evidence={})
    assert result["member_observations"]["A"]["first_progress_s"] == 4
    assert saved["member_observations"]["A"]["first_progress_s"] == 2


def test_a_gap_does_not_invent_progress_between_adjacent_steps():
    detector = ObstructionDetector({"J": 90}, observation_interval_s=1)
    sample(detector, 1, evidence={})
    result = sample(detector, 20, reading=rows(a=5, speed=1), evidence={})
    assert result["advancing_members"] == []
    assert result["member_observations"]["A"]["step_progress_m"] is None
    assert not result["continuous_measurement"]


def test_invalid_measurements_remain_unknown_and_serializable():
    detector = ObstructionDetector({"J": 90})
    reading = rows(a=float("nan"))
    reading["A"]["length"] = float("inf")
    result = sample(detector, .5, reading=reading, evidence={})
    assert not result["continuous_measurement"]
    assert result["member_observations"]["A"]["baseline_distance_m"] is None
    assert result["member_observations"]["A"]["length_m"] is None
    json.dumps(detector.snapshot(), allow_nan=False)


def test_resolved_event_and_later_arrivals_keep_their_real_times():
    detector = ObstructionDetector({"J": 2}, observation_interval_s=1)
    sample(detector, 1)
    moved = {item: {"lane": "out", "lane_position": 5, "speed": 1, "distance": 5, "length": 5} for item in ("A", "B")}
    sample(detector, 2, reading=moved, evidence={})
    sample(detector, 3, reading=moved, evidence={})
    events = detector.observe(4, moved, {"junction_observations": []}, {}, LANES)
    resolved = next(e for e in events if e["event"] == "assessment_changed" and e["state"] == "resolved")
    assert resolved["end_s"] == 4
    saved = detector.summary()
    assert len(detector.record_arrivals({"A": 4.5, "B": 5})) == 2
    assert detector.record_arrivals({"A": 4.5, "B": 5}) == []
    finished = detector.summary()["episodes"][0]
    assert finished["arrivals_s"] == {"A": 4.5, "B": 5} and finished["last_seen_s"] == 4
    assert saved["episodes"][0]["arrivals_s"] == {}


def test_resolution_needs_departure_from_the_passage_and_a_clear_cycle():
    detector = ObstructionDetector({"J": 2}, observation_interval_s=1)
    sample(detector, 1)
    released = {item: {"lane": "out", "lane_position": 5, "speed": 1, "distance": 5, "length": 5} for item in ("A", "B")}
    result = sample(detector, 2, reading=released, evidence={})
    assert result["state"] == "resolution_pending" and result["physical_resolution"] == "unknown"
    sample(detector, 3, reading=released, evidence={})
    result = sample(detector, 4, reading=released, evidence={})
    assert result["state"] == "resolved" and result["physical_resolution"] == "verified"
    assert not detector.active


def test_a_new_stop_during_confirmation_prevents_resolution():
    detector = ObstructionDetector({"J": 2}, observation_interval_s=1)
    sample(detector, 1)
    released = {item: {"lane": "out", "lane_position": 5, "speed": 1, "distance": 5, "length": 5} for item in ("A", "B")}
    sample(detector, 2, reading=released, evidence={})
    interrupted = {**released, "C": {"lane": "a", "speed": 0, "distance": 0, "length": 5}}
    result = sample(detector, 3, reading=interrupted, evidence={})
    assert result["state"] != "resolved" and "C" in result["members"]


def test_arrival_between_observations_is_dated_once_and_not_a_group_resolution():
    detector = ObstructionDetector({"J": 90}, observation_interval_s=1)
    sample(detector, 22)
    arrivals = {"A": 22.5, "B": 22.5}
    events = detector.record_arrivals(arrivals)
    assert len(events) == 2 and all(event["time_s"] == 22.5 for event in events)
    assert all(event["physical_resolution"] == "unknown" for event in events)
    assert detector.record_arrivals(arrivals) == []
    assert detector.last_time == 22
    ended = detector.finish(22.5, "completed")
    assert all(e["physical_resolution"] == "unknown" and not e["right_censored"] for e in ended)


def test_arrival_mid_run_is_not_a_disappearance():
    detector = ObstructionDetector({"J": 2}, observation_interval_s=1)
    sample(detector, 1)
    detector.record_arrivals({"A": 1.5})
    result = sample(detector, 2, reading={"B": rows()["B"]})
    assert result["state"] != "interrupted" and result["arrivals_s"] == {"A": 1.5}
    assert result["remaining_members"] == ["B"]


def test_full_verified_mission_drain_is_a_resolution_proof_not_a_new_observation():
    detector = ObstructionDetector({"J": 90}, observation_interval_s=1)
    sample(detector, 22)
    detector.record_arrivals({"A": 22.5, "B": 22.5})
    event = detector.finish(22.5, "completed", complete_drain=True)[0]
    assert event["physical_resolution"] == "verified" and event["end_s"] == 22.5
    assert event["state"] == "resolved_by_complete_drain" and event["last_seen_s"] == 22


def test_one_member_arrival_cannot_become_verified_drain_of_the_group():
    detector = ObstructionDetector({"J": 90})
    sample(detector, .5)
    detector.record_arrivals({"A": 1})
    event = detector.finish(1, "completed", complete_drain=True)[0]
    assert event["physical_resolution"] == "unknown"


def test_new_member_does_not_inherit_the_old_members_age():
    detector = ObstructionDetector({"J": 2}, observation_interval_s=1)
    sample(detector, 1, reading={"A": rows()["A"]}, evidence={"A": proofs(5)["A"]})
    evidence = {"A": proofs(6)["A"], "B": proofs(0)["B"]}
    result = sample(detector, 2, evidence=evidence)
    assert result["members"] == ["A", "B"] and result["persistent_members"] == ["A"]


def test_red_and_a_foe_occupation_still_do_not_confirm_a_conflict():
    detector = ObstructionDetector({"J": 2})
    snap = {"junction_observations": [{"vehicle_id": "A", "foes": [
        {"vehicle_id": "B", "state": "occupied", "active_constraint": False}]}]}
    result = sample(detector, .5, evidence=proofs(10, permission="prohibited"), snapshot=snap)
    assert result["state"] == "normal_wait" and result["obstruction_verdict"] == "not_confirmed"
    assert result["evidence"]["native_occupations"] == []


def test_measured_progress_resets_the_physical_age_exposed_to_detection():
    from cav_recovery.blockage_observation import BlockageObservation
    from test_blockage_observation import vehicle, snapshot, LANES as physical_lanes, MOVEMENTS
    tracker = BlockageObservation()
    tracker.observe(1, {"A": vehicle()}, snapshot(1), physical_lanes, MOVEMENTS)
    tracker.observe(100, {"A": vehicle(15)}, snapshot(100), physical_lanes, MOVEMENTS)
    assert tracker.current_evidence["A"]["low_progress_age_s"] == 0


@pytest.mark.parametrize("interval,cycle,speed", [(0, 2, .1), (.5, -1, .1), (.5, 2, float("nan"))])
def test_invalid_settings_are_rejected(interval, cycle, speed):
    with pytest.raises(ValueError):
        ObstructionDetector({"J": cycle}, observation_interval_s=interval, halting_speed=speed)


def test_missing_vehicle_without_arrival_is_an_interruption():
    detector = ObstructionDetector({"J": 2}, observation_interval_s=1)
    sample(detector, 1)
    result = sample(detector, 2, reading={"B": rows()["B"]})
    assert result["state"] == "interrupted" and result["physical_resolution"] == "unknown"


def test_horizon_is_censored_not_permanent_or_resolved():
    detector = ObstructionDetector({"J": 2})
    sample(detector, .5)
    event = detector.finish(1, "horizon_reached")[0]
    assert event["right_censored"] and event["end_s"] is None
    assert event["last_seen_s"] == .5 and event["observation_end_s"] == 1
    assert event["physical_resolution"] == "unknown"


def test_unknown_service_cycle_does_not_invent_a_time_threshold():
    detector = ObstructionDetector({}, observation_interval_s=1)
    result = sample(detector, 1, evidence=proofs(1000, kind=None))
    assert not result["persistent"] and "service_cycle_unknown" in result["uncertainties"]


def test_rear_remaining_in_the_passage_prevents_resolution():
    detector = ObstructionDetector({"J": 2}, observation_interval_s=1)
    sample(detector, 1)
    moved = {item: {"lane": "out", "lane_position": 2, "speed": 0, "distance": 5, "length": 5} for item in ("A", "B")}
    footprint = {item: ([("out", -3), ("a", 99)], []) for item in moved}
    for time in range(2, 6):
        result = sample(detector, time, reading=moved, evidence={}, footprints=footprint)
    assert result["physical_resolution"] == "unknown" and result["remaining_members"] == ["A", "B"]


def test_ambiguous_footprint_never_becomes_a_free_passage():
    detector = ObstructionDetector({"J": 2}, observation_interval_s=1)
    sample(detector, 1)
    moved = {item: {"lane": "out", "lane_position": 2, "speed": 1, "distance": 5, "length": 5} for item in ("A", "B")}
    footprint = {item: ([("out", -3)], ["a"]) for item in moved}
    for time in range(2, 6):
        result = sample(detector, time, reading=moved, evidence={}, footprints=footprint)
    assert result["physical_resolution"] == "unknown"


def test_returning_to_a_passage_does_not_leave_the_vehicle_marked_released():
    detector = ObstructionDetector({"J": 90}, observation_interval_s=1)
    sample(detector, 1)
    moved = {item: {"lane": "out", "lane_position": 5, "speed": 1, "distance": 5, "length": 5} for item in ("A", "B")}
    sample(detector, 2, reading=moved, evidence={})
    result = sample(detector, 3, reading=rows(a=100, b=100), evidence=proofs(0))
    assert result["remaining_members"] == ["A", "B"] and result["physical_resolution"] == "unknown"


def test_losing_body_clearance_proof_interrupts_the_confirmation_window():
    detector = ObstructionDetector({"J": 2}, observation_interval_s=1)
    sample(detector, 1)
    moved = {item: {"lane": "out", "lane_position": 5, "speed": 1, "distance": 5, "length": 5} for item in ("A", "B")}
    sample(detector, 2, reading=moved, evidence={})
    unknown = {item: ([], ["a"]) for item in moved}
    result = sample(detector, 3, reading=moved, evidence={}, footprints=unknown)
    assert result["state"] != "resolution_pending" and result["physical_resolution"] == "unknown"


def test_a_gap_or_unknown_progress_prevents_persistence_confirmation():
    detector = ObstructionDetector({"J": 2}, observation_interval_s=1)
    sample(detector, 1)
    result = sample(detector, 20, reading=rows(a=None), evidence=proofs(20))
    assert not result["persistent"] and not result["continuous_measurement"]


def test_cause_age_does_not_reuse_immobilization_age():
    detector = ObstructionDetector({"J": 2}, observation_interval_s=1)
    sample(detector, 1, evidence=proofs(20, kind=None))
    sample(detector, 2, evidence=proofs(21, kind=None))
    result = sample(detector, 3, evidence=proofs(22, kind="connection_obstacle"))
    assert result["persistent"] and result["qualified_cause_age_s"] == 0
    result = sample(detector, 4, evidence=proofs(23, kind="waits_for"))
    assert result["qualified_cause_age_s"] == 0


def test_inputs_and_serialization_are_deterministic():
    reading, evidence = rows(), proofs(1)
    before = copy.deepcopy((reading, evidence))
    a, b = ObstructionDetector({"J": 2}), ObstructionDetector({"J": 2})
    sample(a, .5, reading=reading, evidence=evidence)
    sample(b, .5, reading=dict(reversed(list(reading.items()))), evidence=evidence)
    assert json.dumps(a.snapshot(), sort_keys=True, allow_nan=False) == json.dumps(b.snapshot(), sort_keys=True, allow_nan=False)
    assert (reading, evidence) == before


def test_saved_snapshot_never_acquires_members_or_arrivals_from_a_later_step():
    detector = ObstructionDetector({"J": 90}, observation_interval_s=1)
    sample(detector, 1, reading={"A": rows()["A"]})
    saved = detector.snapshot()
    before = copy.deepcopy(saved)
    sample(detector, 2)
    detector.record_arrivals({"A": 2.5})
    assert saved == before
    assert saved["episodes"][0]["members"] == ["A"]
    assert saved["episodes"][0]["arrivals_s"] == {}


def test_cycles_come_from_the_generated_static_network(tmp_path):
    path = tmp_path / "net.xml"
    path.write_text('<net><tlLogic id="T" type="static"><phase duration="3"/><phase duration="7"/></tlLogic></net>')
    moves = {("a", "out"): [{"tls": "T", "from_lane": "a"}]}
    assert service_cycles(path, LANES, moves) == {"J": 10}
    path.write_text('<net><tlLogic id="T" type="actuated"><phase duration="3"/></tlLogic></net>')
    assert service_cycles(path, LANES, moves) == {"J": None}


def test_cli_enables_obstruction_detection_on_the_existing_pipeline(tmp_path):
    runner = Mock(return_value={"status": "completed", "counts": {}})
    assert cli.main(["run", "crossing", "--detect-obstructions", "--output-dir", str(tmp_path)], runner=runner) == 0
    assert runner.call_args.kwargs["detect_obstructions"] and not runner.call_args.kwargs["gui"]


def test_real_arrival_at_22_5_and_detector_do_not_change_sumo(tmp_path, monkeypatch):
    if not shutil.which("sumo"):
        pytest.skip("SUMO absent ; non-interférence native non validée.")
    from cav_recovery.simulation.road_network import read_config
    from cav_recovery.simulation.traffic_demand import Mission
    fixture = Path(__file__).parent / "fixtures/sumo_smoke"
    config = read_config()
    config["crdg"]["calculation_interval_s"] = 1
    monkeypatch.setattr(traffic_run, "read_config", lambda _: config)
    def prepare(config, directory, *args, **kwargs):
        for name in ("network.net.xml", "traffic.rou.xml", "simulation.sumocfg"):
            shutil.copyfile(fixture / name, directory / name)
        traffic_run.write_json(directory / "scenario.json", config)
        return {"duration_s": .5, "rates_veh_per_hour_per_entry": {},
                "routes": {"mission": ["approach", "destination"]}}, [
                    Mission("smoke_car", "LOW", "entry", "exit", "mission", ("approach", "destination"), "destination", 0)]
    monkeypatch.setattr(traffic_run, "prepare_traffic", prepare)
    off, on = tmp_path / "off", tmp_path / "on"
    a = traffic_run.run_traffic(off, crdg=True)
    b = traffic_run.run_traffic(on, detect_obstructions=True)
    assert a["counts"] == b["counts"] and b["counts"]["simulation_time_s"] == 22.5
    assert a["status"] == b["status"] == "completed"
    assert b["collision_ids"] == [] and b["counts"]["teleport_starts"] == 0
    for file in ("network.net.xml", "traffic.rou.xml", "vehicles.csv", "timeline.csv", "observations.jsonl", "crdg.jsonl"):
        assert (off / file).read_bytes() == (on / file).read_bytes()
    summary = json.loads((on / "obstruction_summary.json").read_text())
    assert summary["last_physical_observation_s"] == 22 and summary["gridlock"] == "not_evaluated"
    assert b["connection_closed"] and b["process_returncode"] == 0
