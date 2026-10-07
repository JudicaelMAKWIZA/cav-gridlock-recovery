"""Tests des causes physiques et des groupes de dépendances, sans action SUMO."""

import copy
import json
from pathlib import Path
import shutil

import networkx as nx
import pytest

from cav_recovery import crdg
from cav_recovery.simulation import traffic_run


def vehicle(edge, position, speed=0):
    return {"route": [edge], "road_id": edge, "route_index": 0, "lane": edge + "_0",
            "lane_position": position, "speed": speed, "length": 5}


@pytest.fixture
def scene():
    lanes = {edge + "_0": {"edge": edge, "length_m": 100 if edge == "a" else 20,
                           "street_name": "Rue " + edge, "internal": False} for edge in ("a", "b")}
    a = vehicle("a", 99)
    a["route"] = ["a", "b"]
    return {"readings": {"A": a}, "leaders": {}, "lanes": lanes,
            "movements": {("a_0", "b"): [{"lane": "b_0", "tls": "light", "link_index": 0}]},
            "tls_states": {"light": "G"}, "first_halted": {"A": 0}, "time_s": 10}


def build(scene, spaces=None):
    return crdg.build_graph(**scene,
                           spaces=spaces if spaces is not None else crdg.receiving_spaces(scene["readings"], scene["lanes"]),
                           min_wait_s=5, halting_speed=0.1, min_gap_m=2.5, step_s=0.5)


def test_waiting_resets_on_motion_and_removes_departed_vehicles():
    first = {}
    rows = {"A": vehicle("a", 99)}
    crdg.update_waiting(first, rows, 1, 0.1)
    crdg.update_waiting(first, rows, 6, 0.1)
    assert first == {"A": 1}
    rows["A"]["speed"] = 0.1
    crdg.update_waiting(first, rows, 6.5, 0.1)
    assert first == {}
    rows["A"]["speed"] = 0
    crdg.update_waiting(first, rows, 7, 0.1)
    assert first == {"A": 7}
    crdg.update_waiting(first, {}, 8, 0.1)
    assert first == {}


@pytest.mark.parametrize("speed,first_halted", [(1, 0), (0, 6)])
def test_moving_or_recently_halted_vehicle_is_not_added(scene, speed, first_halted):
    scene["readings"]["A"]["speed"] = speed
    scene["readings"]["A"]["lane_position"] = 91.499
    scene["first_halted"]["A"] = first_halted
    scene["readings"]["B"] = vehicle("a", 99)
    scene["leaders"]["A"] = ("B", 0.001)
    assert build(scene).number_of_nodes() == 0


def test_only_a_close_observed_halted_leader_creates_an_edge(scene):
    scene["readings"]["A"]["lane_position"] = 91.499
    scene["readings"]["B"] = vehicle("a", 99)
    scene["leaders"]["A"] = ("B", 0.001)
    graph = build(scene)
    assert list(graph.edges) == [("vehicle:A", "vehicle:B")]
    assert graph.edges["vehicle:A", "vehicle:B"]["edge_type"] == "leader"
    assert graph.nodes["vehicle:A"]["waiting_s"] == 10
    assert graph.nodes["vehicle:B"]["waiting_s"] == 0
    for leader in (("B", 5), ("absent", 0), None):
        scene["leaders"]["A"] = leader
        assert build(scene).number_of_edges() == 0
    scene["leaders"]["A"] = ("B", 0)
    scene["readings"]["B"]["speed"] = 1
    assert build(scene).number_of_edges() == 0


@pytest.mark.parametrize("state", ["r", "y", "u"])
@pytest.mark.parametrize("occupied", [False, True])
def test_signal_queue_ends_at_signal_without_a_fake_resource(scene, state, occupied):
    scene["tls_states"]["light"] = state
    scene["readings"]["F"] = vehicle("a", 91.499)
    scene["leaders"]["F"] = ("A", 0.001)
    scene["first_halted"]["F"] = 0
    if occupied:
        scene["readings"]["B"] = vehicle("b", 6)
    graph = build(scene)
    assert list(graph.edges) == [("vehicle:F", "vehicle:A")]
    assert graph.nodes["vehicle:A"]["waiting_reason"] == "signal"
    assert crdg.snapshot(graph)["resource_count"] == 0
    assert crdg.cycle_candidates(graph) == []


def test_green_with_free_receiving_space_has_no_dependency(scene):
    assert build(scene).number_of_nodes() == 0


def test_receiving_resource_uses_actual_rear_position_and_known_occupant(scene):
    scene["readings"]["B"] = vehicle("b", 6)
    spaces = crdg.receiving_spaces(scene["readings"], scene["lanes"])
    assert spaces["b_0"] == {"free_space_m": 1, "occupied": True, "occupant_id": "B"}
    graph = build(scene, spaces)
    resource = "resource:receiving:a_0:b"
    assert graph.edges["vehicle:A", resource]["edge_type"] == "waits_for"
    assert graph.edges[resource, "vehicle:B"]["edge_type"] == "occupied_by"
    assert graph.nodes[resource]["required_space_m"] == 7.5
    assert graph.nodes[resource]["street_name"] == "Rue b"
    assert graph.nodes[resource]["max_free_space_m"] == 1
    assert graph.nodes[resource]["candidate_lanes"] == ["b_0"]
    scene["readings"]["B"]["length"] = 6
    assert crdg.receiving_spaces(scene["readings"], scene["lanes"])["b_0"]["free_space_m"] == 0


def test_one_free_served_alternative_prevents_a_receiving_dependency(scene):
    scene["readings"]["B"] = vehicle("b", 6)
    scene["lanes"]["b_1"] = {**scene["lanes"]["b_0"]}
    scene["movements"][("a_0", "b")].append({"lane": "b_1", "tls": "light", "link_index": 1})
    scene["tls_states"]["light"] = "GG"
    assert build(scene).number_of_nodes() == 0


@pytest.mark.parametrize("states", ["Gr", "gr", "Gy", "Gu"])
def test_free_lane_without_current_service_does_not_hide_blocked_receiving_space(scene, states):
    scene["readings"]["B"] = vehicle("b", 6)
    scene["lanes"]["b_1"] = {**scene["lanes"]["b_0"]}
    scene["movements"][("a_0", "b")].append({"lane": "b_1", "tls": "light", "link_index": 1})
    scene["tls_states"]["light"] = states
    graph = build(scene)
    resource = "resource:receiving:a_0:b"
    assert graph.edges["vehicle:A", resource]["edge_type"] == "waits_for"
    assert graph.nodes[resource]["candidate_lanes"] == ["b_0"]
    assert graph.nodes[resource]["blocked_lanes"] == ["b_0"]
    assert set(graph.nodes[resource]["lane_spaces"]) == {"b_0"}
    assert graph.nodes[resource]["max_free_space_m"] == 1
    assert [connection["lane"] for connection in graph.nodes[resource]["service"]] == ["b_0", "b_1"]
    assert set(graph.successors(resource)) == {"vehicle:B"}


@pytest.mark.parametrize("occupied", [False, True])
def test_uncontrolled_connection_uses_current_receiving_space(scene, occupied):
    scene["movements"][("a_0", "b")][0].update(tls=None, link_index=-1)
    if occupied:
        scene["readings"]["B"] = vehicle("b", 6)
        graph = build(scene)
        resource = "resource:receiving:a_0:b"
        assert graph.edges["vehicle:A", resource]["edge_type"] == "waits_for"
        assert graph.nodes[resource]["service"][0]["state"] is None
    else:
        assert build(scene).number_of_nodes() == 0


def test_all_blocked_lanes_reference_their_real_occupants(scene):
    scene["readings"]["B"] = vehicle("b", 6)
    scene["readings"]["C"] = {**vehicle("b", 5), "lane": "b_1"}
    scene["lanes"]["b_1"] = {**scene["lanes"]["b_0"]}
    scene["movements"][("a_0", "b")].append({"lane": "b_1", "tls": "light", "link_index": 0})
    graph = build(scene)
    resource = "resource:receiving:a_0:b"
    assert graph.nodes[resource]["blocked_lanes"] == ["b_0", "b_1"]
    assert set(graph.successors(resource)) == {"vehicle:B", "vehicle:C"}


def test_unidentified_occupant_does_not_create_a_fake_vehicle(scene):
    graph = build(scene, {"b_0": {"occupied": True, "free_space_m": 0, "occupant_id": None}})
    assert crdg.snapshot(graph)["resource_count"] == 1
    assert crdg.snapshot(graph)["vehicle_count"] == 1
    assert all(row["edge_type"] != "occupied_by" for _, _, row in graph.edges(data=True))


@pytest.mark.parametrize("position,route", [(50, ["a", "b"]), (99, ["a"])])
def test_far_from_exit_or_at_destination_does_not_request_receiving_space(scene, position, route):
    scene["readings"]["A"].update(lane_position=position, route=route)
    scene["readings"]["B"] = vehicle("b", 5)
    assert build(scene).number_of_nodes() == 0


def test_internal_conflicts_are_not_invented(scene):
    scene["readings"]["A"]["road_id"] = ":internal"
    scene["lanes"]["a_0"]["internal"] = True
    scene["readings"]["B"] = vehicle("b", 5)
    assert build(scene).number_of_nodes() == 0


def test_empty_short_connector_is_not_treated_as_a_capacity_obstruction(scene):
    scene["lanes"]["b_0"]["length_m"] = 0.2
    assert build(scene).number_of_nodes() == 0


def test_graph_is_rebuilt_without_stale_edges_and_json_is_deterministic(scene):
    scene["readings"]["B"] = vehicle("b", 5)
    untouched = copy.deepcopy(scene)
    first = crdg.snapshot(build(scene))
    scene["readings"] = dict(reversed(list(scene["readings"].items())))
    assert crdg.snapshot(build(scene)) == first
    assert scene == untouched
    assert json.loads(json.dumps(first, allow_nan=False)) == first
    scene["readings"]["B"]["lane_position"] = 15
    assert build(scene).number_of_nodes() == 0


def test_controlled_heterogeneous_cycle_is_only_a_candidate(scene):
    scene["lanes"]["a_0"]["length_m"] = scene["lanes"]["b_0"]["length_m"] = 7.5
    scene["readings"]["A"]["lane_position"] = 5
    scene["readings"]["B"] = {**vehicle("b", 5), "route": ["b", "a"]}
    scene["first_halted"]["B"] = 0
    scene["movements"][("b_0", "a")] = [{"lane": "a_0", "tls": None, "link_index": -1}]
    row = crdg.snapshot(build(scene))
    assert len(row["cycle_candidates"]) == 1
    assert row["cycle_candidates"][0]["vehicle_count"] == 2
    assert row["cycle_candidates"][0]["resource_count"] == 2
    assert "gridlock" not in row


def test_singletons_self_loops_and_simple_chain_are_not_cycle_candidates():
    graph = nx.DiGraph(time_s=5)
    graph.add_nodes_from((item, {"node_type": "vehicle"}) for item in ("A", "B"))
    graph.add_edges_from([("A", "A"), ("A", "B")])
    assert crdg.cycle_candidates(graph) == []


def test_summary_and_peak_keep_nulls_and_first_chronological_tie(scene):
    summary = crdg.empty_summary()
    empty = crdg.snapshot(build(scene))
    assert summary["first_dependency_s"] is summary["first_cycle_candidate_s"] is None
    peak = crdg.record_snapshot(summary, empty, None)
    scene["readings"]["B"] = vehicle("b", 5)
    row = crdg.snapshot(build(scene))
    peak = crdg.record_snapshot(summary, row, peak)
    later = {**row, "time_s": 15}
    assert crdg.record_snapshot(summary, later, peak) is row
    assert summary["sample_count"] == 3 and summary["snapshots_with_dependencies"] == 2
    assert summary["first_dependency_s"] == 10
    cycle = {**row, "time_s": 20, "cycle_candidates": [{"nodes": [], "vehicle_count": 2, "resource_count": 1}]}
    assert crdg.record_snapshot(summary, cycle, peak) is cycle
    assert summary["first_cycle_candidate_s"] == 20


@pytest.mark.parametrize("interval", [2.5, 5])
def test_real_small_run_has_identical_missions_and_physics_with_and_without_crdg(tmp_path, interval):
    if not shutil.which("sumo") or not shutil.which("netconvert"):
        pytest.skip("SUMO absent ; non-interférence réelle non validée.")
    off, on = tmp_path / "off", tmp_path / "on"
    from cav_recovery.simulation.road_network import read_config
    config = read_config()
    config["crdg"]["sample_interval_s"] = interval
    path = tmp_path / "config.json"
    traffic_run.write_json(path, config)
    a = traffic_run.run_traffic(off, duration_s=60, config_path=path)
    b = traffic_run.run_traffic(on, duration_s=60, config_path=path, crdg=True)
    assert a["status"] == b["status"] == "completed"
    assert a["counts"] == b["counts"]
    assert a["collision_ids"] == b["collision_ids"] == []
    assert a["network_sha256"] == b["network_sha256"]
    assert (off / "vehicles.csv").read_bytes() == (on / "vehicles.csv").read_bytes()
    assert (off / "traffic.rou.xml").read_bytes() == (on / "traffic.rou.xml").read_bytes()
    assert not (off / "crdg.jsonl").exists()
    summary = json.loads((on / "crdg_summary.json").read_text())
    lines = [json.loads(line) for line in (on / "crdg.jsonl").read_text().splitlines()]
    assert len(lines) == summary["sample_count"] > 0
    assert summary["sample_count"] == int(b["counts"]["simulation_time_s"] // interval)
    lanes, _ = crdg.read_network(off / "network.net.xml")
    assert lanes["427630648#3_0"]["street_name"] == "Avenue Colonel Mondjiba"
    assert summary["gridlock"] == "not_evaluated"
    assert a["connection_closed"] and b["connection_closed"]
    assert a["process_stopped"] and b["process_stopped"]


def test_static_connections_are_passenger_legal():
    path = Path(__file__).parent / "fixtures/sumo_smoke/network.net.xml"
    lanes, movements = crdg.read_network(path)
    assert lanes["approach_0"]["length_m"] > 0
    assert any(row["lane"] == "destination_0" for row in movements[("approach_0", "destination")])


@pytest.mark.parametrize("field,value", [("sample_interval_s", 0), ("sample_interval_s", 0.25),
                                        ("sample_interval_s", float("nan")), ("min_wait_s", -1)])
def test_invalid_graph_parameters_are_rejected_before_preparation(tmp_path, field, value):
    from cav_recovery.simulation.road_network import read_config
    config = read_config()
    config["crdg"][field] = value
    path = tmp_path / "config.json"
    traffic_run.write_json(path, config)
    with pytest.raises(ValueError, match="C-RDG"):
        traffic_run.run_traffic(tmp_path / "result", config_path=path, crdg=True)
    assert not (tmp_path / "result").exists()


def test_observer_failure_still_closes_sumo(tmp_path, monkeypatch):
    if not shutil.which("sumo") or not shutil.which("netconvert"):
        pytest.skip("SUMO absent ; fermeture réelle non validée.")
    def fail(*args, **kwargs):
        raise RuntimeError("Erreur de graphe simulée")
    monkeypatch.setattr(crdg, "build_graph", fail)
    result = traffic_run.run_traffic(tmp_path / "failure", duration_s=60, crdg=True)
    assert result["status"] == "failed" and result["reason"] == "Erreur de graphe simulée"
    assert result["connection_closed"] and result["process_stopped"]
    assert result["process_returncode"] == 0 and not result["cleanup_errors"]


def test_legacy_config_without_graph_settings_still_reaches_preparation(tmp_path, monkeypatch):
    from cav_recovery.simulation.road_network import read_config
    config = read_config()
    del config["crdg"]
    path = tmp_path / "legacy.json"
    traffic_run.write_json(path, config)
    def stop(*args, **kwargs):
        raise RuntimeError("Préparation atteinte")
    monkeypatch.setattr(traffic_run, "prepare_traffic", stop)
    with pytest.raises(RuntimeError, match="Préparation atteinte"):
        traffic_run.run_traffic(tmp_path / "result", config_path=path)
