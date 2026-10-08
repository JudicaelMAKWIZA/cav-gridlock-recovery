"""Vérifie les observations temporelles sans diagnostic positif ni commande SUMO."""

import copy
import json
from pathlib import Path
import shutil

import pytest

from cav_recovery.blockage_observation import BlockageObservation
from cav_recovery.simulation import traffic_run


def vehicle(distance=10, speed=0, lane="in_0"):
    return {"distance": distance, "speed": speed, "length": 5, "lane": lane,
            "road_id": "in", "lane_position": 99, "route": ("in", "out"), "route_index": 0}


LANES = {"in_0": {"internal": False, "edge": "in", "length_m": 100},
         "out_0": {"internal": False, "edge": "out", "length_m": 100}}
MOVEMENTS = {("in_0", "out"): [{"from_lane": "in_0", "lane": "out_0", "tls": "light",
                                "link_index": 0, "internal_lanes": []}]}


def snapshot(time, reason="signal", signal="r", leader=None, first=0):
    edges = []
    if leader:
        edges = [{"source": "vehicle:A", "target": "vehicle:" + leader, "edge_type": "leader",
                  "dependency_id": f"leader|vehicle:A|vehicle:{leader}", "evidence": "native_follow_speed",
                  "dependency_first_seen_s": first, "dependency_last_seen_s": time, "dependency_age_s": time - first}]
    return {"time_s": time, "nodes": [], "edges": edges, "waiting_states": {"A": reason},
            "tls_states": {"light": signal}, "receiving_observations": [],
            "cycle_candidates": [], "closed_cycle_candidates": []}


def observe(tracker, time, rows=None, **kwargs):
    return tracker.observe(time, rows if rows is not None else {"A": vehicle()},
                           snapshot(time, **kwargs), LANES, MOVEMENTS)


def test_temporary_stop_motion_progress_and_later_normal_arrival():
    tracker = BlockageObservation()
    events = observe(tracker, 1)
    identity = events[0]["episode_id"]
    events = observe(tracker, 2, {"A": vehicle(11, 2)}, reason="moving", signal="G")
    assert tracker.active["A"]["episode_id"] == identity
    assert any(r["event"] == "motion_observed" for r in events)
    events = observe(tracker, 3, {"A": vehicle(15, 2)}, reason="moving", signal="G")
    assert events[-1]["end_reason"] == "vehicle_length_progress"
    assert events[-1]["progress_m"] == 5
    assert events[-1]["observed_age_s"] == 2
    assert events[-1]["end_interval_s"] == [2, 3]
    events = tracker.observe(4, {}, snapshot(4), LANES, MOVEMENTS, arrivals=["A"])
    assert events[0]["event"] == "arrival_observed" and events[0]["episode_id"] == identity
    assert tracker.summary()["gridlock"] == "not_evaluated"


def test_changing_leader_does_not_reset_physical_age_and_new_dependency_has_own_age():
    tracker = BlockageObservation()
    observe(tracker, 1, {"A": vehicle(), "B": vehicle(100, 1)}, leader="B", first=1)
    events = observe(tracker, 21, {"A": vehicle(), "C": vehicle(100, 1)}, leader="C", first=21)
    row = next(e for e in events if e["event"] == "cause_changed")
    assert row["observed_age_s"] == 20
    assert row["evidence"]["dependencies"][0]["dependency_age_s"] == 0
    assert tracker.active["A"]["first_seen_s"] == 1
    assert row["previous_causes"] != row["evidence"]["causes"]


def test_red_to_green_unknown_passage_is_not_automatically_free():
    tracker = BlockageObservation()
    row = observe(tracker, 1)[0]
    assert row["evidence"]["signal_permission"] == "prohibited"
    assert row["evidence"]["passage_state"] == "constrained"
    current = snapshot(2, "unknown", "G")
    current["receiving_observations"] = [{"vehicle_id": "A", "lane_states": {"out_0": "free"}}]
    events = tracker.observe(2, {"A": vehicle()}, current, LANES, MOVEMENTS)
    evidence = next(e["evidence"] for e in events if e["event"] == "passage_changed")
    assert evidence["signal_permission"] == "permitted"
    assert evidence["receiving_lane_states"] == {"out_0": "free"}
    assert evidence["passage_state"] == "unknown"


def test_slow_creeping_is_measured_instead_of_called_no_progress():
    tracker = BlockageObservation()
    observe(tracker, 1)
    observe(tracker, 11, {"A": vehicle(10.9, .09)})
    assert tracker.active["A"]["progress_m"] == pytest.approx(.9)
    assert tracker.active["A"]["without_vehicle_length_progress_s"] == 10
    events = observe(tracker, 101, {"A": vehicle(15, .05)})
    assert any(e["event"] == "progress_observed" for e in events)
    assert not tracker.active


def test_missing_distance_does_not_invent_zero_and_breaks_continuous_measurement():
    tracker = BlockageObservation()
    observe(tracker, 1)
    observe(tracker, 2, {"A": vehicle(None)})
    assert tracker.active["A"]["progress_m"] is None
    assert tracker.active["A"]["without_vehicle_length_progress_s"] is None
    observe(tracker, 3, {"A": vehicle(15)})
    assert "A" in tracker.active
    assert tracker.active["A"]["without_vehicle_length_progress_s"] == 0
    assert tracker.active["A"]["observed_age_s"] == 2


def test_related_vehicle_progress_is_kept_when_it_stops_being_the_cause():
    tracker = BlockageObservation()
    observe(tracker, 1, {"A": vehicle(), "B": vehicle(100, 2)}, leader="B")
    events = observe(tracker, 2, {"A": vehicle(), "B": vehicle(107, 2)}, signal="G", reason="unknown")
    evidence = next(e["evidence"] for e in events if e["event"] == "related_progress_observed")
    assert evidence["related_vehicles"][0]["progress_since_first_related_observation_m"] == 7
    assert evidence["related_vehicles"][0]["currently_related"] is False


@pytest.mark.parametrize("status,state,reason", [("horizon_reached", "right_censored", "horizon_reached"),
                                               ("failed", "interrupted", "run_ended_without_resolution")])
def test_open_episode_is_censored_or_interrupted_not_permanent(status, state, reason):
    tracker = BlockageObservation()
    observe(tracker, 1)
    observe(tracker, 1200)
    row = tracker.finish(1200.5, status)[0]
    assert row["state"] == state and row["end_reason"] == reason
    assert row["last_seen_s"] == 1200 and row["observed_age_s"] == 1199
    assert row["end_s"] is None
    assert row["right_censored"] is (state == "right_censored")
    assert tracker.finish(1201, status) == []
    assert "confirmed_gridlock" not in json.dumps(tracker.summary())


def test_disappearance_without_arrival_interrupts_the_evidence():
    tracker = BlockageObservation()
    observe(tracker, 1)
    row = observe(tracker, 2, {})[0]
    assert row["event"] == "observation_interrupted"
    assert row["end_s"] is None and row["end_reason"] == "missing_observation"


def test_disappearing_dependency_is_not_a_physical_resolution():
    tracker = BlockageObservation()
    observe(tracker, 1, leader="B")
    events = tracker.observe(586, {"A": vehicle()}, snapshot(586, "unknown", "G"), LANES, MOVEMENTS,
                             dependency_events=[{"event": "disappeared", "source": "vehicle:A", "age_s": 585}])
    assert "A" in tracker.active
    assert tracker.active["A"]["observed_age_s"] == 585
    assert any(e["event"] == "dependency_no_longer_represented" for e in events)
    assert all(e["event"] != "situation_ended" for e in events)


def test_groups_use_current_candidates_only_and_unknown_end_is_not_resolution():
    tracker = BlockageObservation()
    current = snapshot(1)
    current["cycle_candidates"] = [{"nodes": ["vehicle:B", "vehicle:A"], "vehicle_count": 2, "resource_count": 0}]
    rows = {"A": vehicle(), "B": vehicle(20)}
    events = tracker.observe(1, rows, current, LANES, MOVEMENTS)
    assert len(tracker.groups) == 1
    started = next(e for e in events if e["scope"] == "group")
    current["time_s"] = 2
    tracker.observe(2, rows, current, LANES, MOVEMENTS)
    assert next(iter(tracker.groups.values()))["episode_id"] == started["episode_id"]
    events = observe(tracker, 3, rows)
    ended = next(e for e in events if e["scope"] == "group")
    assert ended["physical_resolution"] == "unknown"
    assert not tracker.groups


def test_inputs_are_not_mutated_and_events_are_deterministic_json():
    rows, current = {"A": vehicle()}, snapshot(1)
    original = copy.deepcopy((rows, current))
    a = BlockageObservation().observe(1, rows, current, LANES, MOVEMENTS)
    b = BlockageObservation().observe(1, rows, current, LANES, MOVEMENTS)
    assert json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)
    assert (rows, current) == original


def test_priority_signal_change_is_kept_even_when_permission_stays_permitted():
    tracker = BlockageObservation()
    observe(tracker, 1, signal="g", reason="unknown")
    events = observe(tracker, 2, signal="G", reason="unknown")
    assert any(r["event"] == "passage_changed" and r["evidence"]["service"][0]["signal_state"] == "G" for r in events)


def test_uncontrolled_legal_movement_is_not_material_access_proof():
    tracker = BlockageObservation()
    movements = copy.deepcopy(MOVEMENTS)
    movements[("in_0", "out")][0]["tls"] = None
    events = tracker.observe(1, {"A": vehicle()}, snapshot(1, "unknown"), LANES, movements)
    assert events[0]["evidence"]["signal_permission"] == "permitted"
    assert events[0]["evidence"]["passage_state"] == "unknown"


def test_entry_red_does_not_stop_a_vehicle_already_on_an_internal_lane():
    tracker = BlockageObservation()
    lanes = {**LANES, ":ego_0": {"internal": True, "edge": ":ego", "length_m": 20}}
    movement = {**MOVEMENTS[("in_0", "out")][0], "internal_lanes": [":ego_0"]}
    movements = {**MOVEMENTS, (":ego_0", "out"): [movement]}
    row = {**vehicle(lane=":ego_0"), "road_id": ":ego", "lane_position": 2}
    events = tracker.observe(1, {"A": row}, snapshot(1, "unknown", "r"), lanes, movements)
    evidence = events[0]["evidence"]
    assert evidence["signal_permission"] == "not_applicable"
    assert evidence["service"][0]["entry_signal_state"] == "r"
    assert evidence["passage_state"] == "unknown"


def test_closed_group_is_not_duplicated_and_kind_change_keeps_the_same_episode():
    tracker = BlockageObservation()
    current = snapshot(1)
    candidate = {"nodes": ["vehicle:A", "vehicle:B"], "vehicle_count": 2, "resource_count": 0}
    current["cycle_candidates"] = current["closed_cycle_candidates"] = [candidate]
    rows = {"A": vehicle(), "B": vehicle(20)}
    tracker.observe(1, rows, current, LANES, MOVEMENTS)
    assert len(tracker.groups) == 1
    group = next(iter(tracker.groups.values()))
    current["closed_cycle_candidates"] = []
    events = tracker.observe(2, rows, current, LANES, MOVEMENTS)
    assert next(iter(tracker.groups.values()))["episode_id"] == group["episode_id"]
    assert any(e["event"] == "candidate_kind_changed" for e in events)
    row = tracker.finish(2, "horizon_reached")[-1]
    assert row["right_censored"] and row["end_s"] is None


def test_dependency_disappearance_is_kept_on_the_same_step_as_observed_progress():
    tracker = BlockageObservation()
    observe(tracker, 1, leader="B")
    events = tracker.observe(2, {"A": vehicle(15, 2)}, snapshot(2, "moving", "G"), LANES, MOVEMENTS,
                             dependency_events=[{"event": "disappeared", "source": "vehicle:A", "age_s": 1}])
    assert any(e["event"] == "dependency_no_longer_represented" for e in events)
    assert any(e["event"] == "situation_ended" for e in events)


def test_missing_speed_interrupts_without_fabricating_a_physical_end():
    tracker = BlockageObservation()
    observe(tracker, 1)
    events = observe(tracker, 2, {"A": vehicle(speed=None)})
    assert events[0]["state"] == "interrupted" and events[0]["end_s"] is None


def test_real_observer_error_keeps_controlled_sumo_closure(tmp_path, monkeypatch):
    if not shutil.which("sumo") or not shutil.which("netconvert"):
        pytest.skip("SUMO absent ; fermeture sur erreur non validée.")
    def fail(*args, **kwargs):
        raise RuntimeError("Observation physique interrompue")
    monkeypatch.setattr(BlockageObservation, "observe", fail)
    result = traffic_run.run_traffic(tmp_path / "failure", duration_s=.5, blockage_evidence=True)
    assert result["status"] == "failed" and result["reason"] == "Observation physique interrompue"
    assert result["connection_closed"] and result["process_stopped"]
    assert result["process_returncode"] == 0 and not result["cleanup_errors"]
    assert json.loads((tmp_path / "failure/blockage_summary.json").read_text())["run_status"] == "failed"


def test_cli_enables_evidence_without_changing_existing_crdg_option(monkeypatch, tmp_path):
    import runpy
    import sys
    from unittest.mock import Mock
    runner = Mock(return_value={"status": "completed", "counts": {}})
    monkeypatch.setattr(traffic_run, "run_traffic", runner)
    monkeypatch.setattr(sys, "argv", ["run_traffic.py", "--blockage-evidence", "--output-dir", str(tmp_path)])
    cli = runpy.run_path(str(Path(__file__).parents[1] / "scripts/run_traffic.py"))
    assert cli["main"]() == 0
    assert runner.call_args.kwargs["blockage_evidence"] is True
    assert runner.call_args.kwargs["crdg"] is False


def test_real_small_run_evidence_has_identical_physics_and_arrivals(tmp_path):
    if not shutil.which("sumo") or not shutil.which("netconvert"):
        pytest.skip("SUMO absent ; non-interférence réelle non validée.")
    off, on = tmp_path / "off", tmp_path / "on"
    a = traffic_run.run_traffic(off, duration_s=60, crdg=True)
    b = traffic_run.run_traffic(on, duration_s=60, blockage_evidence=True)
    assert a["status"] == b["status"] == "completed"
    assert a["counts"] == b["counts"]
    assert a["collision_ids"] == b["collision_ids"] == []
    for name in ("network.net.xml", "traffic.rou.xml", "vehicles.csv", "timeline.csv", "lanes.csv",
                 "observations.jsonl", "crdg.jsonl", "crdg_events.jsonl"):
        assert (off / name).read_bytes() == (on / name).read_bytes()
    assert not (off / "blockage_events.jsonl").exists()
    events = [json.loads(line) for line in (on / "blockage_events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "situation_started" for e in events)
    assert any(e["event"] == "arrival_observed" for e in events)
    assert not any(e["event"] == "episode_censored" for e in events)
    summary = json.loads((on / "blockage_summary.json").read_text())
    assert summary["gridlock"] == "not_evaluated"
    assert b["connection_closed"] and b["process_stopped"] and not b["cleanup_errors"]
