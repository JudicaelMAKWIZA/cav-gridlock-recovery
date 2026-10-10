"""Vérifie les conflits natifs et les alternatives, sans commande de conduite."""

import copy
import json
from pathlib import Path
import shutil
import socket
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import networkx as nx
import pytest

from cav_recovery import crdg
from cav_recovery.simulation import traffic_run
from cav_recovery.simulation.sumo_process import close_sumo


@pytest.fixture
def crossing():
    lanes = {item: {"edge": edge, "length_m": 100 if item == "in_0" else 20,
                    "street_name": "Rue " + edge, "junction": "J", "internal": item.startswith(":")}
             for item, edge in (("in_0", "in"), ("out_0", "out"), (":ego_0", ":ego"),
                                ("foe_in_0", "foe_in"), ("foe_out_0", "foe_out"), (":foe_0", ":foe"))}
    ego = {"from_lane": "in_0", "lane": "out_0", "via_lane": ":ego_0", "internal_lanes": [":ego_0"],
           "tls": "light", "link_index": 0}
    foe = {"from_lane": "foe_in_0", "lane": "foe_out_0", "via_lane": ":foe_0", "internal_lanes": [":foe_0"],
           "tls": "light", "link_index": 1}
    movements = {("in_0", "out"): [ego], (":ego_0", "out"): [ego],
                 ("foe_in_0", "foe_out"): [foe], (":foe_0", "foe_out"): [foe]}
    readings = {
        "A": {"road_id": "in", "route": ["in", "out"], "route_index": 0,
              "lane": "in_0", "lane_position": 99, "speed": 0, "length": 5},
        "B": {"road_id": ":foe", "route": ["foe_in", "foe_out"], "route_index": 0,
              "lane": ":foe_0", "lane_position": 5, "speed": 1, "length": 5},
    }
    observation = {"foes": [("B", 2, -1, 5, 2, ":ego_0", ":foe_0", True, False)],
                   "links": [("out_0", False, False, True, ":ego_0", "g", "s", 10)],
                   "priority_foes": {("in_0", "out_0"): ["foe_in_0"]},
                   "internal_foes": {":ego_0": [":foe_0"]}, "stop_line_speed": 0,
                   "halting_speed": .1, "connection_blocker": None}
    return {"readings": readings, "lanes": lanes, "movements": movements, "leaders": {},
            "first_halted": {"A": 0}, "tls_states": {"light": "gG"}, "time_s": 10,
            "junctions": {"A": observation}}


def graph(scene):
    data = {key: value for key, value in scene.items() if key != "leaders"}
    return crdg.build_graph(**data, spaces=crdg.receiving_spaces(scene["readings"], scene["lanes"]),
                            halting_speed=.1, min_gap_m=2.5)


def junctions(scene):
    return [data for _, data in graph(scene).nodes(data=True) if data.get("resource_type") == "junction_conflict"]


def test_permissive_green_has_native_priority_and_an_occupied_conflict(crossing):
    before = copy.deepcopy(crossing)
    current = graph(crossing)
    resource = "resource:junction::ego_0::foe_0"
    assert current.edges["vehicle:A", resource]["edge_type"] == "waits_for"
    edge = current.edges[resource, "vehicle:B"]
    assert edge["edge_type"] == "blocked_by"
    request = current.edges["vehicle:A", resource]
    assert request["ego_response"] is True and request["foe_response"] is False
    assert request["priority_evidence"] == "native_priority_at_stopline"
    assert current.nodes["vehicle:A"]["waiting_reason"] == "junction_conflict"
    assert current.nodes[resource]["release_mode"] == "all_blockers_clear"
    assert crossing == before
    assert json.loads(json.dumps(crdg.snapshot(current), allow_nan=False)) == crdg.snapshot(current)


@pytest.mark.parametrize("state", [None, "g"])
def test_uncontrolled_and_permissive_connections_require_current_priority_evidence(crossing, state):
    if state is None:
        crossing["movements"][("in_0", "out")][0]["tls"] = None
        crossing["junctions"]["A"]["links"][0] = ("out_0", False, False, True, ":ego_0", "m", "s", 10)
    assert len(junctions(crossing)) == 1
    crossing["junctions"]["A"]["priority_foes"] = {}
    assert junctions(crossing) == []


def test_geometric_foe_with_ego_priority_does_not_create_an_approaching_dependency(crossing):
    crossing["tls_states"]["light"] = "Gr"
    obs = crossing["junctions"]["A"]
    obs["links"] = [("out_0", True, True, True, ":ego_0", "G", "s", 10)]
    obs["foes"] = [("B", 2, 10, 5, 13, ":ego_0", ":foe_0", False, True)]
    assert junctions(crossing) == []


def test_priority_green_requires_more_than_a_physically_occupied_conflict(crossing):
    crossing["tls_states"]["light"] = "Gr"
    obs = crossing["junctions"]["A"]
    obs["links"] = [("out_0", True, True, True, ":ego_0", "G", "s", 10)]
    obs["foes"] = [("B", 2, -1, 5, 2, ":ego_0", ":foe_0", False, True)]
    assert junctions(crossing) == []
    obs["connection_blocker"] = "B"
    current = graph(crossing)
    resource = "resource:junction::ego_0::foe_0"
    assert current.edges["vehicle:A", resource]["priority_evidence"] == "native_connection_obstacle"
    obs["internal_foes"] = {}
    assert junctions(crossing) == []


def test_internal_priority_link_state_is_not_a_signal_color(crossing):
    crossing["readings"]["A"].update(lane=":ego_0", road_id=":ego", lane_position=1)
    crossing["junctions"]["A"]["links"] = [("out_0", True, True, True, "", "M", "s", 10)]
    crossing["junctions"]["A"]["connection_blocker"] = "B"
    current = graph(crossing)
    request = current.edges["vehicle:A", "resource:junction::ego_0::foe_0"]
    assert request["link_state"] == "M"
    assert request["signal_state"] is None


def test_shared_conflict_keeps_each_ego_observation(crossing):
    crossing["readings"]["A"]["lane_position"] = 100
    crossing["junctions"]["A"]["foes"] = [("B", 0, -1, 3, 2, ":ego_0", ":foe_0", True, False)]
    crossing["readings"]["C"] = {**crossing["readings"]["A"], "lane_position": 92.5}
    crossing["first_halted"]["C"] = 0
    crossing["junctions"]["C"] = copy.deepcopy(crossing["junctions"]["A"])
    crossing["junctions"]["C"]["foes"] = [("B", 7.5, -1, 10.5, 2, ":ego_0", ":foe_0", True, False)]
    current = graph(crossing)
    resource = "resource:junction::ego_0::foe_0"
    assert current.edges["vehicle:A", resource]["ego_distance_m"] == 0
    assert current.edges["vehicle:C", resource]["ego_distance_m"] == 7.5
    assert current.edges["vehicle:A", resource]["from_lane"] == "in_0"
    assert current.edges["vehicle:A", resource]["to_lane"] == "out_0"
    assert "from_lane" not in current.nodes[resource]
    assert current.edges[resource, "vehicle:B"]["foe_distance_m"] == -1
    assert len([d for _, d in current.nodes(data=True) if d.get("resource_type") == "junction_conflict"]) == 1


@pytest.mark.parametrize("state", ["r", "y", "u"])
def test_signal_with_foes_is_not_a_junction_dependency(crossing, state):
    crossing["tls_states"]["light"] = state + "G"
    crossing["junctions"]["A"]["links"][0] = ("out_0", False, False, True, ":ego_0", state, "s", 10)
    crossing["readings"]["F"] = {**crossing["readings"]["A"], "lane_position": 91}
    crossing["first_halted"]["F"] = 0
    crossing["leaders"]["F"] = ("A", 0)
    current = graph(crossing)
    assert current.graph["waiting_states"]["A"] == "signal"
    assert not any(d.get("resource_type") == "junction_conflict" for _, d in current.nodes(data=True))


@pytest.mark.parametrize("problem", ["absent", "different_lane", "far", "cleared", "no_response",
                                     "open_link", "no_foe", "unknown_movement", "not_at_stopline", "moving"])
def test_insufficient_junction_evidence_does_not_invent_a_blocker(crossing, problem):
    obs = crossing["junctions"]["A"]
    if problem == "absent":
        del crossing["readings"]["B"]
    elif problem == "different_lane":
        crossing["readings"]["B"]["lane"] = "foe_in_0"
    elif problem == "far":
        obs["foes"] = [("B", 20, -1, 23, 2, ":ego_0", ":foe_0", True, False)]
        obs["stop_line_speed"] = 1
    elif problem == "cleared":
        obs["foes"] = [("B", 2, -10, 5, -7, ":ego_0", ":foe_0", True, False)]
    elif problem == "no_response":
        obs["foes"] = [("B", 2, -1, 5, 2, ":ego_0", ":foe_0", False, True)]
    elif problem == "open_link":
        obs["links"] = [("out_0", False, True, True, ":ego_0", "g", "s", 10)]
    elif problem == "no_foe":
        obs["links"] = [("out_0", False, False, False, ":ego_0", "g", "s", 10)]
    elif problem == "unknown_movement":
        crossing["readings"]["A"]["route_index"] = -1
    elif problem == "not_at_stopline":
        obs["stop_line_speed"] = .2
    else:
        crossing["readings"]["A"]["speed"] = .1
    assert junctions(crossing) == []


def test_independent_receiving_and_junction_constraints_can_coexist(crossing):
    crossing["readings"]["F"] = {**crossing["readings"]["A"], "lane_position": 100}
    crossing["leaders"]["A"] = ("F", 0)
    crossing["readings"]["F"].update(lane="out_0", road_id="out", lane_position=5, route_index=1)
    assert junctions(crossing)
    assert any(d.get("resource_type") == "receiving_space" for _, d in graph(crossing).nodes(data=True))


def test_internal_movement_is_verified_without_inventing_next_edge(crossing):
    crossing["readings"]["A"].update(lane=":ego_0", road_id=":ego", lane_position=1)
    assert crdg.next_edge(crossing["readings"]["A"]) is None
    assert len(junctions(crossing)) == 1
    crossing["readings"]["A"]["route"] = ["wrong", "out"]
    assert junctions(crossing) == []


def test_disappearing_conflict_leaves_no_stale_arc(crossing):
    assert len(junctions(crossing)) == 1
    crossing["junctions"]["A"]["foes"] = []
    assert junctions(crossing) == []


def alternatives_graph(all_enclosed=False):
    graph = nx.DiGraph(time_s=10)
    for item in ("A", "B", "C"):
        graph.add_node("vehicle:" + item, node_type="vehicle")
    graph.add_node("X", node_type="resource", resource_type="receiving_space", release_mode="one_candidate_free",
                   candidate_lanes=["one", "two"], blocked_lanes=["one", "two"], required_space_m=7.5,
                   lane_spaces={"one": {"occupied": True, "free_space_m": 0, "occupant_id": "B"},
                                "two": {"occupied": True, "free_space_m": 0, "occupant_id": "C"}})
    graph.add_edge("vehicle:A", "X", edge_type="waits_for")
    for vehicle, lane in (("B", "one"), ("C", "two")):
        graph.add_edge("X", "vehicle:" + vehicle, edge_type="occupied_by", lanes=[lane])
    graph.add_edge("vehicle:B", "vehicle:A", edge_type="leader")
    if all_enclosed:
        graph.add_edge("vehicle:C", "vehicle:A", edge_type="leader")
    return graph


def test_one_external_alternative_prevents_a_closed_cycle():
    current = crdg.snapshot(alternatives_graph())
    assert len(current["cycle_candidates"]) == 1
    assert current["closed_cycle_candidates"] == []


def test_all_alternatives_in_same_component_produce_a_closed_candidate():
    current = crdg.snapshot(alternatives_graph(True))
    assert current["closed_cycle_candidates"] == current["cycle_candidates"]
    assert current["closed_cycle_candidates"][0]["vehicle_count"] == 3
    assert "gridlock" not in current


@pytest.mark.parametrize("escape", ["unknown", "free", "missing_arc", "wrong_lane"])
def test_unknown_or_inconsistent_alternative_is_not_a_closed_candidate(escape):
    graph = alternatives_graph(True)
    resource = graph.nodes["X"]
    if escape == "unknown":
        resource["lane_spaces"]["two"]["occupant_id"] = None
    elif escape == "free":
        resource["lane_spaces"]["two"]["free_space_m"] = 7.5
    elif escape == "missing_arc":
        graph.remove_edge("X", "vehicle:C")
    else:
        graph.edges["X", "vehicle:C"]["lanes"] = ["wrong"]
    assert crdg.cycle_candidates(graph)
    assert crdg.closed_cycle_candidates(graph) == []


def test_single_receiving_lane_still_has_a_closed_candidate():
    graph = alternatives_graph()
    resource = graph.nodes["X"]
    resource["candidate_lanes"] = resource["blocked_lanes"] = ["one"]
    del resource["lane_spaces"]["two"]
    graph.remove_node("vehicle:C")
    assert crdg.closed_cycle_candidates(graph) == crdg.cycle_candidates(graph)


def test_closed_candidate_precedes_a_larger_structural_cycle_and_ties_keep_first():
    summary = crdg.empty_summary()
    structural = crdg.snapshot(alternatives_graph())
    structural["cycle_candidates"][0]["vehicle_count"] = 100
    closed = crdg.snapshot(alternatives_graph(True))
    closed["time_s"] = 15
    peak = crdg.record_snapshot(summary, structural, None)
    assert crdg.record_snapshot(summary, closed, peak) is closed
    assert crdg.record_snapshot(summary, {**closed, "time_s": 20}, closed) is closed
    assert summary["snapshots_with_closed_cycle_candidates"] == 2
    assert summary["first_closed_cycle_candidate_s"] == 15
    assert summary["max_closed_cycle_vehicle_count"] == 3


def test_junction_summary_reports_first_instant_and_maximum(crossing):
    summary = crdg.empty_summary()
    row = crdg.snapshot(graph(crossing))
    crdg.record_snapshot(summary, row, None)
    assert summary["snapshots_with_junction_dependencies"] == 1
    assert summary["first_junction_dependency_s"] == 10
    assert summary["max_junction_resources"] == 1


def test_foe_queries_use_stopped_vehicles_and_reuse_static_cache(crossing):
    connection = SimpleNamespace(
        lane=SimpleNamespace(getLinks=Mock(return_value=crossing["junctions"]["A"]["links"]),
                             getFoes=Mock(return_value=("foe_in_0",)),
                             getInternalFoes=Mock(return_value=(":foe_0",))),
        vehicle=SimpleNamespace(getJunctionFoes=Mock(return_value=crossing["junctions"]["A"]["foes"]),
                                getStopSpeed=Mock(return_value=0)))
    cache = {}
    def sample():
        return traffic_run.sample_junctions(connection, crossing["readings"],
                                           crossing["lanes"], crossing["movements"], cache, halting_speed=.1)
    assert set(sample()) == {"A"}
    assert set(sample()) == {"A"}
    assert connection.lane.getFoes.call_count == connection.lane.getInternalFoes.call_count == 1
    assert connection.vehicle.getJunctionFoes.call_count == 2
    crossing["readings"]["A"]["lane_position"] = 50
    assert sample()
    assert connection.vehicle.getJunctionFoes.call_count == 3
    assert connection.vehicle.getJunctionFoes.call_args.args[1] == 70


@pytest.fixture(scope="module", params=["priority", "green", "red", "internal"])
def native_crossing(request, tmp_path_factory):
    if not shutil.which("sumo") or not shutil.which("netconvert"):
        pytest.skip("SUMO absent ; sémantique native non vérifiée.")
    import traci

    kind = request.param
    directory = tmp_path_factory.mktemp("junction_" + kind)
    fixture = Path(__file__).parent / "fixtures/junction_conflicts"
    files = (("left_turn.nod.xml", "left_turn.edg.xml", "left_turn.con.xml") if kind == "internal"
             else ("nodes.nod.xml", "edges.edg.xml", "connections.con.xml"))
    command = ["netconvert", "--node-files", str(fixture / files[0]), "--edge-files", str(fixture / files[1]),
               "--connection-files", str(fixture / files[2]), "--no-turnarounds", "true",
               "--output-file", str(directory / "network.net.xml")]
    if kind in ("green", "red"):
        command.extend(["--tls.set", "crossing"])
    subprocess.run(command, capture_output=True, text=True, check=True)
    root = ET.parse(directory / "network.net.xml").getroot()
    if kind in ("green", "red"):
        tls = root.find("tlLogic")
        tls.clear()
        tls.attrib.update(id="crossing", programID="0", type="static", offset="0")
        controlled = [c for c in root.findall("connection") if c.get("tl")]
        state = ["r"] * len(controlled)
        for c in controlled:
            state[int(c.get("linkIndex"))] = (("g" if kind == "green" else "r")
                                             if c.get("from") == "west_in" else "G")
        ET.SubElement(tls, "phase", duration="65", state="".join(state))
        ET.SubElement(tls, "phase", duration="2", state="r" * len(state))
        ET.SubElement(tls, "phase", duration="20", state="".join("G" if s != "G" else "r" for s in state))
        traffic_run.write_xml(directory / "network.net.xml", root)
    routes = ET.Element("routes")
    ET.SubElement(routes, "vType", traffic_run.VEHICLE_TYPE)
    ET.SubElement(routes, "route", id="ego", edges="south_in west_out" if kind == "internal" else "west_in east_out")
    ET.SubElement(routes, "route", id="foe", edges="north_in south_out" if kind == "internal" else "south_in north_out")
    ET.SubElement(routes, "vehicle", id="A", type="passenger_CAV", route="ego", depart="0", departSpeed="0")
    for i in range(22):
        ET.SubElement(routes, "vehicle", id=f"B{i:02}", type="passenger_CAV", route="foe", depart=str(i * 1.5), departSpeed="0")
    traffic_run.write_xml(directory / "routes.rou.xml", routes)
    lanes, movements = crdg.read_network(directory / "network.net.xml")
    first, cache, frames, raw, trails = {}, {}, [], [], {}
    connection = process = None
    result = {"connection_closed": False, "process_stopped": False, "process_returncode": None,
              "forced_process_stop": False, "cleanup_errors": []}
    try:
        with socket.socket() as port_socket:
            port_socket.bind(("127.0.0.1", 0))
            port = port_socket.getsockname()[1]
        with (directory / "sumo.log").open("w") as log:
            process = subprocess.Popen(["sumo", "-n", str(directory / "network.net.xml"),
                                        "-r", str(directory / "routes.rou.xml"), "--step-length", "0.5",
                                        "--time-to-teleport", "-1", "--max-depart-delay", "-1",
                                        "--collision.check-junctions", "true", "--seed", "1",
                                        "--no-step-log", "true", "--remote-port", str(port)], stdout=log, stderr=log)
            connection = traci.connect(port=port, proc=process, numRetries=60, waitBetweenRetries=.1)
            assert "1.27.1" in connection.getVersion()[1]
            for _ in range(200):
                connection.simulationStep()
                readings = traffic_run.subscribed_readings(connection, include_length=True)
                time = connection.simulation.getTime()
                assert not connection.simulation.getCollidingVehiclesIDList()
                assert not connection.simulation.getStartingTeleportIDList()
                assert not connection.simulation.getEndingTeleportIDList()
                crdg.update_waiting(first, readings, time, .1)
                leaders = {item: connection.vehicle.getLeader(item) for item in readings}
                model_samples = [(gap, speed, connection.vehicle.getFollowSpeed("A", 0, gap, speed, 4.5))
                                 for gap, speed in ((0, 0), (.05, 0), (.1, 0), (.2, 0), (.1, 1), (.1, 3))] if "A" in readings else []
                following = traffic_run.sample_following(connection, readings, leaders, lanes, movements, cache,
                                                         halting_speed=.1, leader_decel=4.5)
                footprints = crdg.update_footprints(trails, readings, lanes, movements)
                # Deux témoins suffisent pour confronter les réponses aux priorités connues.
                for item in ("A", "B01"):
                    if item in readings:
                        for foe in connection.vehicle.getJunctionFoes(item, 7.5):
                            raw.append({"id": item, "time_s": time, "foe": foe,
                                        "model_samples": model_samples,
                                        "reading": readings[item].copy(), "foe_reading": readings[foe[0]].copy(),
                                        "links": connection.lane.getLinks(readings[item]["lane"], extended=True),
                                        "internal_foes": connection.lane.getInternalFoes(foe[5])})
                observations = traffic_run.sample_junctions(
                    connection, readings, lanes, movements, cache, halting_speed=.1, following=following)
                states = {item: connection.trafficlight.getRedYellowGreenState(item)
                          for item in connection.trafficlight.getIDList()}
                current = crdg.build_graph(readings, crdg.receiving_spaces(readings, lanes, footprints),
                                           lanes, movements, states, first, time,
                                           halting_speed=.1, min_gap_m=2.5, junctions=observations, following=following)
                frames.append(crdg.snapshot(current))
    finally:
        close_sumo(connection, process, result)
    assert result["connection_closed"] and result["process_stopped"]
    assert result["process_returncode"] == 0 and not result["forced_process_stop"] and not result["cleanup_errors"]
    return kind, root, raw, frames


def test_native_responses_match_known_request_priority(native_crossing):
    kind, root, raw, _ = native_crossing
    assert raw
    junction = root.find("junction[@id='crossing']")
    internal = junction.get("intLanes").split()
    responses = {int(row.get("index")): row.get("response")[::-1] for row in junction.findall("request")}
    for record in raw:
        foe = record["foe"]
        ego_index, foe_index = internal.index(foe[5]), internal.index(foe[6])
        # True correspond au bit response : ce mouvement doit répondre à l'autre.
        assert foe[7] == (responses[ego_index][foe_index] == "1")
        assert foe[8] == (responses[foe_index][ego_index] == "1")
        if record["id"] == "A":
            assert foe[7] is True and foe[8] is False
        else:
            assert foe[7] is False and foe[8] is True
        assert foe[6] in record["internal_foes"]
    occupied = [r for r in raw if r["id"] == "A" and r["foe_reading"]["lane"] == r["foe"][6]
                and r["foe"][2] <= 0 < r["foe"][4] + r["foe_reading"]["length"]]
    assert occupied
    if kind != "red":
        assert any(r["reading"]["speed"] < .1 and r["links"][0][1:4] == (False, False, True) for r in occupied)


def test_native_krauss_follow_speed_separates_limiting_gaps_and_mobile_leader(native_crossing):
    _, _, raw, _ = native_crossing
    samples = next(r["model_samples"] for r in raw if r["model_samples"])
    speeds = {(gap, leader_speed): speed for gap, leader_speed, speed in samples}
    assert speeds[(0, 0)] == 0
    assert 0 < speeds[(.05, 0)] < speeds[(.1, 0)] < .1
    assert speeds[(.2, 0)] >= .1
    # Un leader lent peut s'arrêter en un pas : sa mobilité ne suffit pas à libérer le suivi.
    assert speeds[(.1, 1)] == pytest.approx(speeds[(.1, 0)])
    assert speeds[(.1, 3)] >= .1


def test_native_distances_follow_entry_exit_and_the_observed_vehicle(native_crossing):
    _, _, raw, _ = native_crossing
    relevant = [r for r in raw if r["id"] == "A" and r["foe"][0] == "B01"
                and r["foe_reading"]["lane"] == r["foe"][6]]
    assert len(relevant) >= 2
    first = relevant[0]
    entry = first["foe"][2] + first["foe_reading"]["lane_position"]
    exit_position = first["foe"][4] + first["foe_reading"]["lane_position"]
    assert exit_position > entry
    for record in relevant:
        f = record["foe"]
        assert f[3] > f[1] and f[4] > f[2]
        assert f[2] + record["foe_reading"]["lane_position"] == pytest.approx(entry, abs=1e-7)
        assert f[4] + record["foe_reading"]["lane_position"] == pytest.approx(exit_position, abs=1e-7)


def test_real_junction_dependencies_follow_priority_signal_and_internal_state(native_crossing):
    kind, _, _, frames = native_crossing
    before_switch = [frame for frame in frames if frame["time_s"] < 65]
    resources = [node for frame in before_switch for node in frame["nodes"]
                 if node.get("resource_type") == "junction_conflict"]
    if kind == "red":
        assert resources == []
    else:
        assert resources
        assert any(edge["edge_type"] == "blocked_by" for frame in before_switch for edge in frame["edges"])
        assert all(not frame["closed_cycle_candidates"] for frame in frames)
    if kind == "internal":
        assert any(node.get("vehicle_id") == "A" and node["internal"] and node["next_edge"] is None
                   and node["waiting_reason"] == "junction_conflict"
                   for frame in frames for node in frame["nodes"])
