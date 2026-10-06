"""Preuves synthétiques d'occupation aval, de service empêché et de reprise externe."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cav_recovery.simulation import traffic_incident as incident
from cav_recovery.simulation import traffic_run as run
from cav_recovery.simulation import road_network as network
from cav_recovery.simulation.traffic_demand import Mission, TrafficInputError
from test_traffic import Connection, fake_run, mission, normal_frames


def placement():
    return incident.IncidentPlacement("incident", "down", "down_0", 15, 60, 5, 2.5, 300, ("in",), 0)


def frame():
    return {
        "incident": {"edge_id": "down", "lane_id": "down_0", "position_m": 15, "speed_mps": 0,
                     "distance_m": 100, "progress_m": 0, "stop_state": 1, "leader_id": None, "next_tls": []},
        "a": {"edge_id": "in", "lane_id": "in_0", "position_m": 9, "speed_mps": 0,
              "distance_m": 80, "progress_m": 0, "stop_state": 0, "leader_id": "b",
              "next_tls": [{"id": network.CENTER_NODE, "state": "G"}]},
        "b": {"edge_id": "down", "lane_id": "down_0", "position_m": 7.5, "speed_mps": 0,
              "distance_m": 90, "progress_m": 0, "stop_state": 0, "leader_id": "incident", "next_tls": []},
    }


def active_experiment():
    experiment = incident.LocalBlockage(placement(), 600)
    experiment.state = "active"
    experiment.t_phys = 0
    return experiment


def green_phase(experiment, change=None, start=0):
    for i in range(1, 79):
        values = frame()
        if change:
            change(values, i)
        experiment.update(start + i * .5, values, 0, start, start + 39)


def test_incident_alone_does_not_form_blockage():
    experiment = active_experiment()
    green_phase(experiment, lambda rows, i: (rows.pop("a"), rows.pop("b")))
    assert experiment.t_phys == 0 and experiment.t_form is None


def test_not_active_does_not_form_blockage():
    experiment = incident.LocalBlockage(placement(), 600)
    green_phase(experiment)
    assert experiment.t_phys is None and experiment.t_form is None


def test_red_is_not_a_service_opportunity():
    experiment = active_experiment()
    for i in range(1, 13):
        rows = frame()
        rows["a"]["next_tls"][0]["state"] = "r"
        experiment.update(39 + i * .5, rows, 1, 39, 45)
    assert experiment.t_form is None and experiment.opportunities == []


@pytest.mark.parametrize("change", [
    lambda rows, i: rows.pop("b"),
    lambda rows, i: rows["a"].update(edge_id="far_upstream"),
    lambda rows, i: rows["a"].update(leader_id=None),
    lambda rows, i: rows["a"]["next_tls"].append({"id": "auxiliary", "state": "r"}),
    lambda rows, i: rows["a"]["next_tls"].append({"id": "yield", "state": "g"}),
    lambda rows, i: rows["a"].update(stop_state=1),
    lambda rows, i: rows["incident"].update(lane_id="another_lane"),
    lambda rows, i: rows["incident"].update(speed_mps=1),
])
def test_each_physical_condition_is_required(change):
    experiment = active_experiment()
    green_phase(experiment, change)
    assert experiment.t_form is None


def test_cohort_cannot_be_replaced_during_the_phase():
    experiment = active_experiment()
    def change(rows, i):
        if i > 20:
            rows["c"] = rows.pop("a")
    green_phase(experiment, change)
    assert experiment.t_form is None


def test_progress_invalidates_even_a_halting_vehicle():
    experiment = active_experiment()
    def change(rows, i):
        if i == 20:
            rows["a"].update(speed_mps=.01, progress_m=.005, distance_m=80.005)
    green_phase(experiment, change)
    assert experiment.t_form is None


def test_partial_green_is_not_a_complete_phase():
    experiment = active_experiment()
    for i in range(2, 79):
        experiment.update(i * .5, frame(), 0, 0, 39)
    assert experiment.t_form is None


def test_missing_sample_cannot_be_hidden_by_sample_count():
    experiment = active_experiment()
    experiment.update(.5, frame(), 0, 0, 39)
    with pytest.raises(RuntimeError, match="manquante"):
        experiment.update(1.5, frame(), 0, 0, 39)


def test_full_green_confirms_once_with_measured_progress():
    experiment = active_experiment()
    for i in range(1, 78):
        experiment.update(i * .5, frame(), 0, 0, 39)
        assert experiment.t_form is None
    experiment.update(39, frame(), 0, 0, 39)
    assert experiment.t_form == 39
    assert experiment.evidence["samples"] == 78
    assert experiment.evidence["affected_ids"] == ["a", "b"]
    assert experiment.evidence["vehicles"]["a"]["total_progress_m"] == 0
    assert experiment.evidence["vehicles"]["a"]["end_chain"] == ["a", "b", "incident"]
    green_phase(experiment, start=90)
    assert experiment.t_form == 39 and len(experiment.opportunities) == 2


def test_ulp_tolerance_is_not_a_physical_motion_threshold():
    assert incident.position_tolerance(100) < 1e-10
    assert incident.HALTING_SPEED_MPS == .1


def test_leader_cycle_or_absence_is_not_evidence():
    rows = frame()
    rows["b"]["leader_id"] = "a"
    assert incident.leader_chain("a", rows, "incident") == []
    rows.pop("b")
    assert incident.leader_chain("a", rows, "incident") == []


def test_moving_intermediate_leader_breaks_the_physical_cause():
    experiment = active_experiment()
    def change(rows, i):
        rows["moving"] = {**rows["b"], "speed_mps": 2, "progress_m": 1}
        rows["a"]["leader_id"] = rows["b"]["leader_id"] = "moving"
    green_phase(experiment, change)
    assert experiment.t_form is None and experiment.affected_ids == set()


def stop_connection():
    values = frame()["incident"]
    calls = []
    vehicle = SimpleNamespace(
        getSpeed=lambda item: values["speed_mps"], getLanePosition=lambda item: values["position_m"],
        getDistance=lambda item: values["distance_m"], getLeader=lambda *args: None,
        getRoadID=lambda item: values["edge_id"], getLaneID=lambda item: values["lane_id"],
        getStopState=lambda item: values["stop_state"], getNextTLS=lambda item: [],
        getLength=lambda item: 5, getMinGap=lambda item: 2.5,
        setStop=lambda *args, **kwargs: calls.append(("stop", args, kwargs)),
        resume=lambda item: calls.append(("resume", item)),
    )
    connection = SimpleNamespace(vehicle=vehicle, lane=SimpleNamespace(getLength=lambda item: 60),
                                 trafficlight=SimpleNamespace(getPhase=lambda item: 0,
                                                             getNextSwitch=lambda item: 39))
    return connection, values, calls


def test_activation_is_observed_not_the_requested_time():
    connection, values, calls = stop_connection()
    values.update(stop_state=0, speed_mps=2, position_m=11, distance_m=96)
    experiment = incident.LocalBlockage(placement(), 600)
    experiment.observe(connection, .5, set())
    assert experiment.state == "not_requested" and calls == []
    experiment.observe(connection, 1, {"incident"})
    assert experiment.state == "requested" and experiment.t_phys is None
    assert calls[0][2]["duration"] > 600
    values.update(stop_state=1, speed_mps=0, position_m=15, distance_m=100)
    experiment.observe(connection, 1.5, {"incident"})
    assert experiment.t_phys == 1.5 and experiment.state == "active"
    assert len(calls) == 1


@pytest.mark.parametrize("field,value", [("edge_id", "wrong"), ("lane_id", "wrong_0"), ("position_m", 16)])
def test_wrong_physical_stop_is_refused(field, value):
    connection, values, _ = stop_connection()
    experiment = incident.LocalBlockage(placement(), 600)
    experiment.state = "requested"
    values[field] = value
    with pytest.raises(RuntimeError, match="position"):
        experiment.observe(connection, .5, {"incident"})


def test_feasibility_release_is_separate_and_not_automatic():
    connection, values, calls = stop_connection()
    experiment = incident.LocalBlockage(placement(), 600, True)
    experiment.state = "active"
    experiment.previous = {"incident": dict(values)}
    experiment.observe(connection, .5, {"incident"})
    assert not calls and experiment.release_time_s is None
    experiment = active_experiment()
    experiment.feasibility_release = True
    green_phase(experiment)
    experiment.previous = {"incident": dict(values)}
    experiment.observe(connection, 39, {"incident"})
    assert calls == [("resume", "incident")]
    assert experiment.release_time_s == 39 and experiment.resume_time_s is None
    values.update(stop_state=0, speed_mps=1.3, position_m=15.65, distance_m=100.65)
    experiment.observe(connection, 39.5, {"incident"})
    assert experiment.resume_time_s == 39.5
    assert experiment.events[-2]["kind"] == "feasibility_release"
    assert experiment.state == "released" and len(calls) == 1


def synthetic_placement_file(tmp_path):
    import xml.etree.ElementTree as ET
    root = ET.Element("net")
    for ident, destination, length in [("in", network.CENTER_NODE, 30),
                                       ("middle", "next", 10), ("23183369#5", "exit", 60)]:
        edge = ET.SubElement(root, "edge", id=ident, to=destination)
        ET.SubElement(edge, "lane", id=ident + "_0", length=str(length))
    ET.SubElement(root, "connection", **{"from": "in", "to": "middle", "tl": network.CENTER_NODE,
                                       "linkIndex": "0"})
    path = tmp_path / "network.xml"
    ET.ElementTree(root).write(path)
    missions = [Mission(v, "LOW", "entry", "exit", "r", ("in", "middle", "23183369#5"),
                        "23183369#5", t) for v, t in [("later", 5), ("first", 0)]]
    return path, missions


def test_placement_uses_mission_order_and_vehicle_geometry(tmp_path):
    path, missions = synthetic_placement_file(tmp_path)
    result = incident.incident_placement(path, list(reversed(missions)), {"length": "5", "minGap": "2.5"})
    assert result.vehicle_id == "first" and result.position_m == 15
    assert result.incoming_edges == ("in",) and result.link_index == 0


@pytest.mark.parametrize("length", ["nan", "0", "20"])
def test_unsafe_geometry_is_refused(tmp_path, length):
    path, missions = synthetic_placement_file(tmp_path)
    with pytest.raises(TrafficInputError):
        incident.incident_placement(path, missions, {"length": length, "minGap": "2.5"})


def test_nominal_runner_does_not_collect_or_request_an_incident(monkeypatch, tmp_path):
    prepare = Mock(side_effect=AssertionError("incident non demandé"))
    monkeypatch.setattr(run, "incident_placement", prepare)
    result, connection, _, _ = fake_run(monkeypatch, tmp_path, normal_frames())
    assert result["status"] == "passed" and connection.closed
    assert "incident" not in result
    assert not (tmp_path / "result/incident_observations.jsonl").exists()


def test_incident_source_is_in_code_digest(monkeypatch):
    original = Path.read_bytes
    before = run.code_provenance()["sha256"]
    monkeypatch.setattr(Path, "read_bytes", lambda p: original(p) + (b"\n" if p.name == "traffic_incident.py" else b""))
    assert run.code_provenance()["sha256"] != before


def test_cli_preserves_nominal_arguments_and_exposes_release(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("incident_cli", Path(__file__).parents[1] / "scripts/run_traffic.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    result = {"status": "passed", "reason": None, "counts": run.TrafficLedger([mission()]).snapshot(),
              "connection_closed": True, "process_stopped": True}
    execute = Mock(return_value=result)
    monkeypatch.setattr(cli, "run_traffic", execute)
    monkeypatch.setattr("sys.argv", ["run", "--scenario-dir", "synthetic", "--regime", "LOW",
                                     "--output-dir", str(tmp_path), "--local-blockage", "--feasibility-release"])
    assert cli.main() == 0
    assert execute.call_args.kwargs["local_blockage"] and execute.call_args.kwargs["feasibility_release"]


def incident_run(monkeypatch, tmp_path, frames, observation_error=None, feasibility=False, cleanup_error=False):
    connection = Connection(frames)
    monkeypatch.setattr(run, "read_scenario", lambda path: {
        "vehicle_type": {"length": "5", "minGap": "2.5"},
        "regimes": {"LOW": {"missions": [mission().__dict__], "plan": {"injection_s": .5}}}})
    monkeypatch.setattr(run, "check_environment", lambda *args: {})
    monkeypatch.setattr(run, "file_hash", lambda path: "synthetic")
    monkeypatch.setattr(run, "code_provenance", lambda: {"sha256": "synthetic"})
    monkeypatch.setattr(run.shutil, "which", lambda name: name)
    process = Mock()
    process.poll.return_value = 0
    process.wait.side_effect = [OSError("attente"), None] if cleanup_error else None
    monkeypatch.setattr(run.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(run.importlib, "import_module", lambda name: SimpleNamespace(connect=lambda **kwargs: connection))
    monkeypatch.setattr(run, "verify_trips", lambda path, ledger: {i: {"arrival_s": 1} for i in ledger.arrivals})
    place = replace(placement(), vehicle_id="car")
    monkeypatch.setattr(run, "incident_placement", lambda *args: place)
    experiment = incident.LocalBlockage(place, 1.5)
    experiment.observe = Mock(return_value={}, side_effect=observation_error)
    if feasibility:
        experiment.feasibility_release = True
        experiment.t_form = .5
        experiment.resume_time_s = .5
        experiment.state = "released"
    monkeypatch.setattr(run, "LocalBlockage", lambda *args: experiment)
    result = run.run_traffic(tmp_path, "LOW", tmp_path / "result", local_blockage=True,
                            feasibility_release=feasibility, drain_horizon_s=1)
    return result, connection, process


@pytest.mark.parametrize("bad_frame,message", [
    ({"teleport_starts": ["car"]}, "Téléportation"),
    ({}, "Disparition"),
    ({"active": {"car": {"route": ("start", "wrong")}}}, "destination"),
])
def test_incident_does_not_relax_mission_integrity(monkeypatch, tmp_path, bad_frame, message):
    result, connection, process = incident_run(monkeypatch, tmp_path, [normal_frames()[0], bad_frame])
    assert result["status"] == "failed" and message in result["reason"]
    assert not result["incident"]["validated"]
    assert connection.closed and process.wait.called
    assert result["connection_closed"] and result["process_stopped"]
    saved = json.loads((tmp_path / "result/summary.json").read_text())
    assert saved["failure_observation"] is not None


def test_measurement_error_still_closes_and_does_not_invent_integrity(monkeypatch, tmp_path):
    result, connection, _ = incident_run(monkeypatch, tmp_path, normal_frames(), RuntimeError("mesure"))
    assert result["status"] == "failed" and result["reason"] == "mesure"
    assert result["incident"]["integrity"]["collision_count"] is None
    assert connection.closed and result["process_stopped"]


def test_horizon_without_formation_is_a_negative_result(monkeypatch, tmp_path):
    frames = [normal_frames()[0], {"active": {"car": {"road": "start", "index": 0}}},
              {"active": {"car": {"road": "start", "index": 0}}}]
    result, connection, _ = incident_run(monkeypatch, tmp_path, frames)
    assert result["status"] == "failed"
    assert "Horizon" in result["reason"]
    assert result["incident"]["t_form"] is None
    assert connection.closed


def test_release_without_incident_is_refused_before_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "read_scenario", lambda path: {})
    launch = Mock()
    monkeypatch.setattr(run.subprocess, "Popen", launch)
    with pytest.raises(TrafficInputError, match="libération"):
        run.run_traffic(tmp_path, "LOW", tmp_path / "result", feasibility_release=True)
    launch.assert_not_called()


def test_cleanup_failure_invalidates_even_a_feasibility_result(monkeypatch, tmp_path):
    result, connection, process = incident_run(monkeypatch, tmp_path, normal_frames(),
                                              feasibility=True, cleanup_error=True)
    assert result["counts"]["arrived"] == 1
    assert result["status"] == "failed" and not result["incident"]["validated"]
    assert result["incident"]["t_form"] == .5
    assert result["forced_process_stop"] and process.terminate.called
    assert connection.closed


def test_uncommanded_release_is_not_a_success():
    connection, values, _ = stop_connection()
    experiment = active_experiment()
    values.update(stop_state=0, speed_mps=1)
    with pytest.raises(RuntimeError, match="sans commande"):
        experiment.observe(connection, .5, {"incident"})
    assert experiment.state == "unexpected_release" and experiment.events[-1]["kind"] == "unexpected_release"
