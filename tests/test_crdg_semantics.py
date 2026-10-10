"""Contraintes immédiates, empreintes de véhicules et règles de libération."""

import json
from pathlib import Path
import shutil
import socket
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import pytest

from cav_recovery import crdg
from cav_recovery.simulation import traffic_run
from cav_recovery.simulation.sumo_process import close_sumo
from test_crdg import scene, vehicle, build
from test_junction_conflicts import crossing, graph, alternatives_graph
from test_traffic import actual_network


@pytest.mark.parametrize("gap,follow_speed,limited", [(0, 0, True), (.05, .049, True),
                                                     (.1, .099, True), (.2, .199, False)])
def test_native_follow_speed_not_a_five_centimetre_cutoff(scene, gap, follow_speed, limited):
    scene["readings"]["A"]["lane_position"] = 92.5 - gap
    scene["readings"]["B"] = vehicle("a", 100)
    scene["leaders"]["A"] = ("B", gap)
    scene["follow_speed"] = follow_speed
    assert build(scene).has_edge("vehicle:A", "vehicle:B") is limited


def test_unqualified_native_leader_is_not_a_strong_dependency(scene):
    scene["readings"]["B"] = vehicle("b", 15)
    scene["leaders"]["A"] = ("B", 0)
    data = {key: value for key, value in scene.items() if key != "leaders"}
    g = crdg.build_graph(**data, spaces=crdg.receiving_spaces(scene["readings"], scene["lanes"]),
                         halting_speed=.1, min_gap_m=2.5,
                         following={"A": crdg.Following("B", 0, 0, "unknown")})
    assert not g.has_edge("vehicle:A", "vehicle:B")


def test_moving_vehicle_is_not_labelled_as_blocked_even_with_a_leader(scene):
    scene["readings"]["A"]["speed"] = .1
    scene["readings"]["B"] = vehicle("a", 100)
    scene["leaders"]["A"] = ("B", 0)
    scene["follow_speed"] = 0
    graph = build(scene)
    assert not graph.has_node("vehicle:A")
    assert "A" not in graph.graph["waiting_states"]


def test_moving_vehicle_does_not_trigger_native_blockage_queries(scene):
    scene["readings"]["A"]["speed"] = 1
    scene["readings"]["B"] = vehicle("a", 100, speed=1)
    connection = SimpleNamespace(vehicle=Mock(), lane=Mock())
    assert traffic_run.sample_following(connection, scene["readings"], {"A": ("B", 0)}, scene["lanes"],
                                       scene["movements"], {}, halting_speed=.1, leader_decel=4.5) == {}
    assert traffic_run.sample_junctions(connection, scene["readings"], scene["lanes"], scene["movements"],
                                       {}, halting_speed=.1) == {}
    assert not connection.vehicle.getFollowSpeed.called and not connection.vehicle.getJunctionFoes.called


def test_connection_obstacle_is_not_named_longitudinal_leader(crossing):
    following = {"A": crdg.Following("B", 0, 0, "connection_obstacle")}
    data = {key: value for key, value in crossing.items() if key != "leaders"}
    g = crdg.build_graph(**data, spaces=crdg.receiving_spaces(crossing["readings"], crossing["lanes"]),
                         halting_speed=.1, min_gap_m=2.5, following=following)
    assert g.edges["vehicle:A", "vehicle:B"]["edge_type"] == "connection_obstacle"
    assert g.edges["vehicle:A", "vehicle:B"]["evidence"] == "native_connection_obstacle"
    assert not any(e["edge_type"] == "leader" for _, _, e in g.edges(data=True))


@pytest.mark.parametrize("state", ["r", "y", "u"])
def test_red_connection_obstacle_remains_a_signal_wait(crossing, state):
    crossing["tls_states"]["light"] = state + "G"
    data = {key: value for key, value in crossing.items() if key != "leaders"}
    g = crdg.build_graph(**data, spaces=crdg.receiving_spaces(crossing["readings"], crossing["lanes"]),
                         halting_speed=.1, min_gap_m=2.5,
                         following={"A": crdg.Following("B", 0, 0, "connection_obstacle")})
    assert not g.has_edge("vehicle:A", "vehicle:B")
    assert g.graph["waiting_states"]["A"] == "signal"
    assert g.graph["following_observations"][0]["active_constraint"] is False


def test_dependency_age_is_not_halted_age_and_disappearance_resets_it(scene):
    scene["first_halted"]["A"] = -20
    scene["readings"]["B"] = vehicle("b", 5)
    history = {}
    a = build(scene)
    events = crdg.track_dependencies(a, history)
    request = next(e for _, _, e in a.edges(data=True) if e["edge_type"] == "waits_for")
    assert request["dependency_age_s"] == 0
    assert a.nodes["vehicle:A"]["halted_age_s"] == 30
    assert any(e["event"] == "appeared" for e in events)
    scene["time_s"] = 10.5
    b = build(scene)
    crdg.track_dependencies(b, history)
    request = next(e for _, _, e in b.edges(data=True) if e["edge_type"] == "waits_for")
    assert request["dependency_age_s"] == .5
    assert request["observations_count"] == 2
    scene["tls_states"]["light"] = "r"
    c = build(scene)
    assert any(e["event"] == "disappeared" for e in crdg.track_dependencies(c, history))
    scene["time_s"] = 11
    scene["tls_states"]["light"] = "G"
    d = build(scene)
    crdg.track_dependencies(d, history)
    request = next(e for _, _, e in d.edges(data=True) if e["edge_type"] == "waits_for")
    assert request["dependency_first_seen_s"] == 11
    assert request["dependency_age_s"] == 0
    assert json.loads(json.dumps(crdg.snapshot(d), allow_nan=False)) == crdg.snapshot(d)


def test_signal_to_receiving_keeps_a_new_cause_age(scene):
    scene["readings"]["B"] = vehicle("b", 5)
    scene["tls_states"]["light"] = "r"
    history = {}
    crdg.track_dependencies(build(scene), history)
    scene["tls_states"]["light"] = "G"
    scene["time_s"] = 10.5
    g = build(scene)
    events = crdg.track_dependencies(g, history)
    assert any(e["event"] == "cause_changed" and e["previous"] == ["signal"] for e in events)
    assert all(e["dependency_age_s"] == 0 for _, _, e in g.edges(data=True))


def test_known_resume_between_graphs_breaks_the_dependency_episode(scene):
    scene["readings"]["B"] = vehicle("b", 5)
    history = {}
    crdg.track_dependencies(build(scene), history)
    scene["time_s"] = 11
    scene["first_halted"]["A"] = 10.5
    after = build(scene)
    events = crdg.track_dependencies(after, history)
    request = next(e for _, _, e in after.edges(data=True) if e["edge_type"] == "waits_for")
    assert request["dependency_age_s"] == 0
    assert any(e.get("reason") == "halt_period_changed" for e in events)


def test_leader_to_junction_does_not_inherit_the_old_age(crossing):
    history = {}
    before = graph({**crossing, "junctions": {}})
    before.add_node("vehicle:A", node_type="vehicle")
    before.add_node("vehicle:B", node_type="vehicle")
    before.add_edge("vehicle:A", "vehicle:B", edge_type="leader")
    crdg.track_dependencies(before, history)
    crossing["time_s"] = 11
    after = graph(crossing)
    events = crdg.track_dependencies(after, history)
    assert any(e["event"] == "cause_changed" for e in events)
    assert all(e["dependency_age_s"] == 0 for _, _, e in after.edges(data=True))


def test_body_on_two_verified_segments_marks_the_previous_as_occupied():
    lanes = {"a": {"edge": "a", "length_m": 20, "successors": ["b"]},
             "b": {"edge": "b", "length_m": .2, "successors": ["c"]},
             "c": {"edge": "c", "length_m": 20, "successors": []}}
    row = {"lane": "a", "lane_position": 19, "length": 5, "route": ["a", "b", "c"]}
    trails = {}
    crdg.update_footprints(trails, {"B": row}, lanes, {})
    row = {**row, "lane": "c", "lane_position": 1}
    bodies = crdg.update_footprints(trails, {"B": row}, lanes, {})
    spaces = crdg.receiving_spaces({"B": row}, lanes, bodies)
    assert spaces["b"]["occupied"] and spaces["b"]["occupant_id"] == "B"
    assert spaces["b"]["free_space_m"] == 0
    assert spaces["a"]["free_space_m"] == pytest.approx(16.2)
    assert spaces["a"]["knowledge"] == "known"


def test_ambiguous_body_path_is_unknown_not_free():
    lanes = {"a": {"edge": "a", "length_m": 20, "successors": ["b", "c"]},
             "b": {"edge": "b", "length_m": 1, "successors": ["d"]},
             "c": {"edge": "b", "length_m": 1, "successors": ["d"]},
             "d": {"edge": "d", "length_m": 20, "successors": []}}
    row = {"lane": "a", "lane_position": 19, "length": 5, "route": ["a", "b", "d"]}
    trails = {}
    crdg.update_footprints(trails, {"B": row}, lanes, {})
    row = {**row, "lane": "d", "lane_position": 1}
    spaces = crdg.receiving_spaces({"B": row}, lanes, crdg.update_footprints(trails, {"B": row}, lanes, {}))
    assert spaces["b"]["knowledge"] == spaces["c"]["knowledge"] == "unknown"
    assert spaces["b"]["free_space_m"] is None


def test_unobserved_lateral_change_makes_the_previous_internal_path_unknown():
    lanes = {"a0": {"edge": "a", "length_m": 20, "successors": ["v0"], "lateral": ["a1"]},
             "a1": {"edge": "a", "length_m": 20, "successors": ["v1"]},
             "v0": {"edge": ":j", "length_m": 2, "successors": ["b"]},
             "v1": {"edge": ":j", "length_m": 2, "successors": ["b"]},
             "b": {"edge": "b", "length_m": 20, "successors": []}}
    movements = {("a0", "b"): [{"internal_lanes": ["v0"]}],
                 ("a1", "b"): [{"internal_lanes": ["v1"]}]}
    row = {"lane": "a0", "lane_position": 19, "length": 5, "route": ["a", "b"]}
    trails = {}
    crdg.update_footprints(trails, {"B": row}, lanes, movements)
    row = {**row, "lane": "b", "lane_position": 1}
    body = crdg.update_footprints(trails, {"B": row}, lanes, movements)
    spaces = crdg.receiving_spaces({"B": row}, lanes, body)
    assert spaces["v0"]["knowledge"] == spaces["v1"]["knowledge"] == "unknown"
    assert spaces["b"]["knowledge"] == "known" and spaces["b"]["occupied"]


def test_lateral_step_is_never_added_as_a_longitudinal_body_segment():
    lanes = {"a0": {"edge": "a", "length_m": 20, "successors": [], "lateral": ["a1"]},
             "a1": {"edge": "a", "length_m": 20, "successors": []}}
    row = {"lane": "a0", "lane_position": 1, "length": 5, "route": ["a"]}
    trails = {}
    crdg.update_footprints(trails, {"B": row}, lanes, {})
    row = {**row, "lane": "a1", "lane_position": 2}
    body = crdg.update_footprints(trails, {"B": row}, lanes, {})
    assert body["B"][0] == [("a1", -3)]
    assert "a0" in body["B"][1]


def test_unknown_alternative_does_not_create_a_receiving_constraint(scene):
    scene["readings"]["B"] = vehicle("b", 5)
    g = build(scene, {"b_0": {"knowledge": "unknown", "free_space_m": None,
                              "occupied": False, "occupant_id": None}})
    assert g.number_of_edges() == 0
    assert g.graph["receiving_observations"][0]["lane_states"] == {"b_0": "unknown"}


def test_candidate_accepts_a_legal_later_lane_change_but_not_a_dead_end(scene):
    row = {**scene["readings"]["A"], "route": ["a", "b", "c"]}
    scene["lanes"]["b_1"] = {**scene["lanes"]["b_0"], "lateral": ["b_0"]}
    scene["lanes"]["c_0"] = {"edge": "c"}
    connections = [{"lane": "b_0"}, {"lane": "b_1"}]
    movements = {("b_0", "c"): [{"lane": "c_0"}]}
    assert crdg.admissible_lanes(row, connections, scene["lanes"], movements) == ["b_0", "b_1"]
    scene["lanes"]["b_1"]["lateral"] = []
    assert crdg.admissible_lanes(row, connections, scene["lanes"], movements) == ["b_0"]


def test_all_blockers_clear_needs_only_one_enclosed_blocker_to_remain_blocked():
    g = alternatives_graph()
    g.nodes["X"].update(resource_type="junction_conflict", release_mode="all_blockers_clear")
    for target in ("vehicle:B", "vehicle:C"):
        g.edges["X", target]["edge_type"] = "blocked_by"
    closed = crdg.closed_cycle_candidates(g)
    assert len(closed) == 1
    assert set(closed[0]["nodes"]) == {"vehicle:A", "X", "vehicle:B"}


def test_small_closed_group_survives_removal_of_an_escape_from_large_scc():
    g = alternatives_graph(True)
    g.add_node("Y", node_type="resource", resource_type="junction_conflict", release_mode="all_blockers_clear")
    g.add_edge("vehicle:A", "Y", edge_type="waits_for")
    g.add_edge("Y", "vehicle:B", edge_type="blocked_by")
    g.nodes["X"]["lane_spaces"]["two"]["knowledge"] = "unknown"
    structural = crdg.cycle_candidates(g)
    assert len(structural) == 1 and structural[0]["vehicle_count"] == 3
    assert crdg.closed_cycle_candidates(g)[0]["nodes"] == ["Y", "vehicle:A", "vehicle:B"]


def test_native_following_collector_uses_exact_model_inputs(scene):
    scene["readings"]["A"]["lane_position"] = 92.4
    scene["readings"]["B"] = vehicle("a", 100)
    scene["leaders"]["A"] = ("B", .1)
    connection = SimpleNamespace(vehicle=SimpleNamespace(getFollowSpeed=Mock(return_value=.099)))
    result = traffic_run.sample_following(connection, scene["readings"], scene["leaders"], scene["lanes"],
                                         scene["movements"], {}, halting_speed=.1, leader_decel=4.5)
    connection.vehicle.getFollowSpeed.assert_called_once_with("A", 0, .1, 0, 4.5, "B")
    assert result["A"].relation == "longitudinal_following"


def test_real_kintambo_link_leader_is_a_connection_obstacle(actual_network, monkeypatch):
    """Les deux missions normales convergent ; aucun arrêt n'est imposé."""
    import traci
    directory, config, inspection, routes = actual_network
    monkeypatch.setattr(traffic_run, "build_network", lambda *args, **kwargs: (inspection, routes))
    traffic_run.prepare_traffic(config, directory, "HIGH", 1, 600, None)
    lanes, movements = crdg.read_network(directory / "network.net.xml")
    connection = process = None
    cleanup = {"connection_closed": False, "process_stopped": False, "process_returncode": None,
               "forced_process_stop": False, "cleanup_errors": []}
    witnessed = False
    try:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        with (directory / "native_obstacle.log").open("w") as log:
            process = subprocess.Popen(["sumo", "-c", str(directory / "simulation.sumocfg"),
                                        "--remote-port", str(port), "--no-step-log", "true"], stdout=log, stderr=log)
            connection = traci.connect(port=port, proc=process, numRetries=100, waitBetweenRetries=.1)
            for _ in range(940):
                connection.simulationStep()
                readings = traffic_run.subscribed_readings(connection, include_length=True)
                assert not connection.simulation.getCollidingVehiclesIDList()
                assert not connection.simulation.getStartingTeleportIDList()
                item = "mondjiba_000032"
                if item not in readings or readings[item]["speed"] >= .1:
                    continue
                leaders = {item: connection.vehicle.getLeader(item)}
                following = traffic_run.sample_following(connection, readings, leaders, lanes, movements, {},
                                                          halting_speed=.1, leader_decel=4.5)
                proof = following.get(item)
                if proof and proof.vehicle_id == "nguma_000096" and proof.follow_speed < .1:
                    assert proof.relation == "connection_obstacle"
                    assert readings[item]["lane"] != readings[proof.vehicle_id]["lane"]
                    assert any(link[1] for link in connection.lane.getLinks(readings[item]["lane"], extended=True))
                    witnessed = True
                    break
    finally:
        close_sumo(connection, process, cleanup)
    assert witnessed
    assert cleanup["connection_closed"] and cleanup["process_stopped"]
    assert cleanup["process_returncode"] == 0 and not cleanup["cleanup_errors"]


def test_native_priority_green_occupied_foe_does_not_imply_waiting(tmp_path):
    """Le mouvement prioritaire approche pendant que l'autre finit de traverser."""
    if not shutil.which("sumo") or not shutil.which("netconvert"):
        pytest.skip("SUMO absent ; priorité native non vérifiée.")
    import traci
    fixture = Path(__file__).parent / "fixtures/junction_conflicts"
    network = tmp_path / "network.net.xml"
    subprocess.run(["netconvert", "--node-files", str(fixture / "nodes.nod.xml"),
                    "--edge-files", str(fixture / "edges.edg.xml"),
                    "--connection-files", str(fixture / "connections.con.xml"),
                    "--tls.set", "crossing", "--no-turnarounds", "true", "-o", str(network)],
                   capture_output=True, text=True, check=True)
    root = ET.parse(network).getroot()
    tls = root.find("tlLogic")
    for phase in list(tls):
        tls.remove(phase)
    connections = [c for c in root.findall("connection") if c.get("tl")]
    state = ["r"] * len(connections)
    for c in connections:
        state[int(c.get("linkIndex"))] = "G" if c.get("from") == "south_in" else "g"
    ET.SubElement(tls, "phase", duration="20", state="".join(state))
    ET.SubElement(tls, "phase", duration="2", state="r" * len(state))
    ET.SubElement(tls, "phase", duration="20", state="".join("g" if s == "G" else "G" for s in state))
    traffic_run.write_xml(network, root)
    routes = ET.Element("routes")
    ET.SubElement(routes, "vType", traffic_run.VEHICLE_TYPE)
    ET.SubElement(routes, "route", id="ego_route", edges="south_in north_out")
    ET.SubElement(routes, "route", id="foe_route", edges="west_in east_out")
    ET.SubElement(routes, "vehicle", id="foe", type="passenger_CAV", route="foe_route", depart="0")
    ET.SubElement(routes, "vehicle", id="ego", type="passenger_CAV", route="ego_route", depart="6")
    traffic_run.write_xml(tmp_path / "routes.rou.xml", routes)
    lanes, movements = crdg.read_network(network)
    first, cache = {}, {}
    connection = process = None
    cleanup = {"connection_closed": False, "process_stopped": False, "process_returncode": None,
               "forced_process_stop": False, "cleanup_errors": []}
    witnessed = False
    try:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        with (tmp_path / "sumo.log").open("w") as log:
            process = subprocess.Popen(["sumo", "-n", str(network), "-r", str(tmp_path / "routes.rou.xml"),
                                        "--step-length", ".5", "--time-to-teleport", "-1",
                                        "--collision.check-junctions", "true", "--remote-port", str(port),
                                        "--no-step-log", "true"], stdout=log, stderr=log)
            connection = traci.connect(port=port, proc=process, numRetries=100, waitBetweenRetries=.1)
            for _ in range(40):
                connection.simulationStep()
                readings = traffic_run.subscribed_readings(connection, include_length=True)
                now = connection.simulation.getTime()
                crdg.update_waiting(first, readings, now, .1)
                assert not connection.simulation.getCollidingVehiclesIDList()
                assert not connection.simulation.getStartingTeleportIDList()
                if "ego" not in readings or readings["ego"]["road_id"] != "south_in":
                    continue
                assert any(link[5] == "G" for link in connection.lane.getLinks(readings["ego"]["lane"], extended=True))
                movement = crdg.junction_movements(readings["ego"], lanes, movements)
                scope = (lanes[readings["ego"]["lane"]]["length_m"] - readings["ego"]["lane_position"]
                         + max(sum(lanes[lane]["length_m"] for lane in c["internal_lanes"]) for c in movement))
                for raw in connection.vehicle.getJunctionFoes("ego", scope):
                    foe = crdg.JunctionFoe(*raw)
                    other = readings.get(foe.vehicle_id)
                    if (other and other["lane"] == foe.foe_lane and foe.foe_distance <= 0 < foe.foe_exit + other["length"]):
                        assert readings["ego"]["speed"] >= .1
                        observed = traffic_run.sample_junctions(connection, readings, lanes, movements, cache,
                                                                 halting_speed=.1)
                        current = crdg.build_graph(readings, crdg.receiving_spaces(readings, lanes), lanes, movements,
                                                   {"crossing": connection.trafficlight.getRedYellowGreenState("crossing")},
                                                   first, now, halting_speed=.1, min_gap_m=2.5, junctions=observed)
                        assert "vehicle:ego" not in current or current.out_degree("vehicle:ego") == 0
                        witnessed = True
    finally:
        close_sumo(connection, process, cleanup)
    assert witnessed
    assert cleanup["connection_closed"] and cleanup["process_stopped"] and not cleanup["cleanup_errors"]
