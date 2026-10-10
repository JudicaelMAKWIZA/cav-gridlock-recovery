"""Protège les alias, les groupes complets et les seules écritures graphiques."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import networkx as nx
import pytest

from cav_recovery.simulation.crdg_live import (CrdgPanelLink, visual_ids, dependency_groups,
                                               SELECTED, CONSTRAINED, PARTICIPANT, TEXT_KEY,
                                               view_boundary, local_sector, atomic_json)
from cav_recovery.simulation.traffic_demand import Mission
from cav_recovery.simulation.traffic_run import run_traffic


def missions():
    return [Mission(item, "case", "entry", "exit", "route", ("a", "b"), "b", i)
            for i, item in enumerate(("north_000001", "south_000001", "north_000002"))]


def current(time=5, *, edge=True):
    nodes = [{"id": "vehicle:north_000001", "node_type": "vehicle", "vehicle_id": "north_000001"},
             {"id": "vehicle:south_000001", "node_type": "vehicle", "vehicle_id": "south_000001"}]
    edges = [{"source": nodes[0]["id"], "target": nodes[1]["id"], "edge_type": "leader", "dependency_id": "leader-A-B"}] if edge else []
    return {"time_s": time, "nodes": nodes if edge else [], "edges": edges,
            "vehicle_count": 2 if edge else 0, "resource_count": 0, "edge_count": len(edges),
            "cycle_candidates": [], "closed_cycle_candidates": [], "waiting_states": {}}


def native():
    vehicle = Mock()
    vehicle.getColor.return_value = (255, 255, 0, 255)
    vehicle.getParameter.return_value = ""
    return SimpleNamespace(vehicle=vehicle)


def graph_of(snapshot):
    graph = nx.DiGraph()
    graph.add_nodes_from(row["id"] for row in snapshot["nodes"])
    graph.add_edges_from((row["source"], row["target"]) for row in snapshot["edges"])
    return graph


def test_aliases_are_global_deterministic_and_do_not_rename_missions():
    values = missions()
    expected = {"north_000001": "V001", "south_000001": "V002", "north_000002": "V003"}
    assert visual_ids(values) == visual_ids(list(reversed(values))) == expected
    assert [m.vehicle_id for m in values] == list(expected)


def test_all_components_accessible_not_just_the_selected_vehicle():
    snapshot = current()
    other = {"id": "vehicle:north_000002", "node_type": "vehicle", "vehicle_id": "north_000002"}
    resource = {"id": "resource:X", "node_type": "resource", "street_name": "Avenue"}
    snapshot["nodes"].extend((other, resource))
    snapshot["edges"].append({"source": other["id"], "target": resource["id"], "edge_type": "waits_for"})
    groups = dependency_groups(graph_of(snapshot), snapshot)
    assert len(groups) == 2
    assert {item for group in groups for item in group["nodes"]} == {row["id"] for row in snapshot["nodes"]}


def test_colours_and_labels_are_restored_when_constraints_disappear(tmp_path):
    link, connection = CrdgPanelLink(tmp_path, missions()), native()
    readings = {m.vehicle_id: {} for m in missions()}
    snapshot = current()
    link.observe(connection, graph_of(snapshot), snapshot, readings)
    assert link.colours["north_000001"] == CONSTRAINED
    assert link.colours["south_000001"] == PARTICIPANT
    empty = current(5.5, edge=False)
    link.observe(connection, graph_of(empty), empty, readings)
    assert link.colours == {} and link.original == {}
    assert connection.vehicle.setColor.call_args.args[1] == (255, 255, 0, 255)
    assert set(link.labels.values()) == {""}
    link.restore(connection, readings)
    assert connection.vehicle.setParameter.call_args.args[1:] == (TEXT_KEY, "")
    assert {call[0] for call in connection.vehicle.method_calls} <= {"getColor", "getParameter", "setColor", "setParameter"}


def test_panel_frames_every_five_simulated_seconds_and_expiry_between(tmp_path):
    link, connection = CrdgPanelLink(tmp_path, missions()), native()
    readings = {m.vehicle_id: {} for m in missions()}
    for stamp in (0.5, 5, 5.5, 10):
        snapshot = current(stamp, edge=stamp != 5.5)
        link.observe(connection, graph_of(snapshot), snapshot, readings)
        if stamp == 5.5:
            clock = json.loads((link.directory / "clock.json").read_text())
            assert clock["inactive_edges"] == ["leader-A-B"]
            assert json.loads((link.directory / "frame.json").read_text())["snapshot"]["time_s"] == 5
    frames = list(map(json.loads, (link.output_directory / "frames.jsonl").read_text().splitlines()))
    assert [frame["snapshot"]["time_s"] for frame in frames] == [5, 10]


def test_selection_does_not_change_native_ids_or_other_roles(tmp_path):
    link, connection = CrdgPanelLink(tmp_path, missions(), focus="south_000001"), native()
    snapshot = current()
    link.observe(connection, graph_of(snapshot), snapshot, {m.vehicle_id: {} for m in missions()})
    assert link.colours["south_000001"] == SELECTED
    assert link.colours["north_000001"] == CONSTRAINED


def test_live_mode_requires_gui_without_creating_output(tmp_path):
    with pytest.raises(ValueError, match="--gui"):
        run_traffic(tmp_path / "absent", crdg_live=True)
    assert not (tmp_path / "absent").exists()


def test_explicit_close_restores_only_present_vehicles_without_advancing(tmp_path):
    link, connection = CrdgPanelLink(tmp_path, missions()), native()
    snapshot = current()
    readings = {m.vehicle_id: {} for m in missions()}
    link.observe(connection, graph_of(snapshot), snapshot, readings)
    process = Mock(returncode=0)
    process.poll.return_value = 0
    link.process = process
    connection.vehicle.reset_mock()
    atomic_json(link.directory / "control.json", {"close_requested": True})
    link.observe(connection, graph_of(snapshot), snapshot, {"north_000001": {}})
    assert link.stats["closed_by_user"] and not link.control()
    assert connection.vehicle.setColor.call_args_list[0].args == ("north_000001", (255, 255, 0, 255))
    assert connection.vehicle.setColor.call_count == 1
    assert not link.original and not link.colours


def test_finish_closes_panel_even_if_colour_restore_fails(tmp_path):
    link, connection = CrdgPanelLink(tmp_path, missions()), native()
    snapshot = current()
    readings = {m.vehicle_id: {} for m in missions()}
    link.observe(connection, graph_of(snapshot), snapshot, readings)
    link.process = Mock()
    link.process.poll.side_effect = [None, 0]
    connection.vehicle.setColor.side_effect = RuntimeError("affichage indisponible")
    with pytest.raises(RuntimeError, match="affichage indisponible"):
        link.finish(connection, readings, "failed", 5)
    link.process.wait.assert_called_once_with(timeout=5)
    assert json.loads((link.output_directory / "clock.json").read_text())["status"] == "closing"
    assert not link.directory.exists()


@pytest.mark.parametrize("failure", ["terminate", "wait", "poll", "archive"])
def test_finish_releases_files_and_native_view_even_after_shutdown_error(tmp_path, monkeypatch, failure):
    import subprocess
    link, connection = CrdgPanelLink(tmp_path, missions()), native()
    link.log = (link.output_directory / "panel.log").open("w")
    link.native_view = Mock()
    link.process = Mock()
    link.process.poll.side_effect = [None, 0]
    link.process.wait.side_effect = [subprocess.TimeoutExpired("panel", 5), 0]
    if failure == "terminate":
        link.process.terminate.side_effect = OSError("terminate failed")
        link.process.wait.side_effect = [subprocess.TimeoutExpired("panel", 5), subprocess.TimeoutExpired("panel", 5), 0]
    elif failure == "wait":
        link.process.wait.side_effect = [OSError("wait failed"), 0]
    elif failure == "poll":
        link.process.poll.side_effect = [OSError("poll failed"), 0]
    else:
        monkeypatch.setattr("shutil.copyfile", Mock(side_effect=OSError("archive failed")))
    with pytest.raises(OSError, match="failed"):
        link.finish(connection, {}, "failed", 5)
    assert link.log.closed and not link.directory.exists()
    link.native_view.close.assert_called_once()
    assert json.loads((link.output_directory / "summary.json").read_text())["errors"]
    if failure == "terminate":
        link.process.kill.assert_called_once()


def test_software_error_keeps_native_camera_if_sumo_is_still_alive(tmp_path, monkeypatch):
    link, connection, readings = camera_link(tmp_path)
    link.sumo_process = Mock()
    link.sumo_process.poll.return_value = None
    link.camera_request = {"sequence": 1, "kind": "fit", "vehicles": ["north_000001"]}
    monkeypatch.setattr(link, "control", Mock(return_value=True))
    monkeypatch.setattr("cav_recovery.simulation.crdg_live._enter_pressed", lambda: True)
    monkeypatch.setattr("cav_recovery.simulation.crdg_live.sys.stdin", SimpleNamespace(isatty=lambda: True))
    link.hold_view(connection, readings, "failed", 9, reason="Erreur logicielle")
    assert connection.gui.setBoundary.called and not hasattr(connection, "simulationStep")


def test_native_view_uses_only_graphical_parameters(tmp_path):
    import xml.etree.ElementTree as ET
    from cav_recovery.simulation.crdg_live import configure_live_view
    path = tmp_path / "view.xml"
    path.write_text('<viewsettings><scheme name="kintambo"><vehicles vehicleQuality="2"/></scheme></viewsettings>')
    configure_live_view(path)
    values = ET.parse(path).find("scheme/vehicles").attrib
    assert values["vehicleTextParam"] == TEXT_KEY
    assert values["vehicleText_show"] == "true" and values["vehicleQuality"] == "2"
    assert values["vehicleExaggeration"] == "1" and values["vehicleMinSize"] == "4"
    assert all(key.startswith("vehicleText") or key in ("vehicleQuality", "vehicleExaggeration", "vehicleMinSize") for key in values)


def test_cli_live_flag_keeps_one_command_for_sumo_and_panel(monkeypatch):
    import runpy
    import sys
    from pathlib import Path
    from cav_recovery.simulation import traffic_run
    runner = Mock(return_value={"status": "completed", "counts": {}})
    monkeypatch.setattr(traffic_run, "run_traffic", runner)
    monkeypatch.setattr(sys, "argv", ["run_traffic.py", "--kintambo-case", "crossing", "--gui", "--crdg-live"])
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/run_traffic.py"))
    assert module["main"]() == 0
    assert runner.call_args.kwargs["crdg_live"] and runner.call_args.kwargs["gui"]
    assert runner.call_args.kwargs["kintambo_case"] == "crossing"


def test_small_graph_layout_is_deterministic_without_numpy(monkeypatch):
    import builtins
    from cav_recovery.simulation.crdg_panel import graph_positions
    original = builtins.__import__
    def no_numpy(name, *args, **kwargs):
        if name == "numpy" or name.startswith("numpy."):
            raise ImportError("pas de dépendance graphique implicite")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_numpy)
    a = nx.DiGraph([(str(i), str(i + 1)) for i in range(10)])
    b = nx.DiGraph(list(reversed(list(a.edges))))
    assert graph_positions(a) == graph_positions(b)
    assert graph_positions(nx.DiGraph()) == {}
    assert graph_positions(nx.DiGraph([("A", "A")])) == {"A": (0.0, 0.0)}
    assert set(graph_positions(a)) == set(a)


def test_chain_is_named_as_following_not_deadlock():
    snapshot = current()
    group = dependency_groups(graph_of(snapshot), snapshot)[0]
    assert group["label"] == "File de 2 voitures"
    assert group["categories"] == ["following"]
    assert "cycle" not in group["categories"]


def test_group_categories_come_from_resources_and_structural_candidates():
    snapshot = current()
    resource = {"id": "resource:X", "node_type": "resource", "resource_type": "receiving_space"}
    snapshot["nodes"].append(resource)
    snapshot["edges"].extend(({"source": "vehicle:south_000001", "target": "resource:X", "edge_type": "waits_for"},
                              {"source": "resource:X", "target": "vehicle:north_000001", "edge_type": "occupied_by"}))
    group = dependency_groups(graph_of(snapshot), snapshot)[0]
    assert group["categories"] == ["following", "receiving"]
    snapshot["cycle_candidates"] = [{"nodes": sorted(graph_of(snapshot))}]
    group = dependency_groups(graph_of(snapshot), snapshot)[0]
    assert "cycle" in group["categories"] and group["label"].startswith("Cycle candidat")


def test_native_connection_obstacle_is_a_passage_group_not_a_following_file():
    snapshot = current()
    snapshot["edges"][0]["edge_type"] = "connection_obstacle"
    group = dependency_groups(graph_of(snapshot), snapshot)[0]
    assert group["categories"] == ["junction"]
    assert group["label"] == "Contrainte de passage"


def test_default_labels_only_for_selection_and_change_without_a_colour_change(tmp_path):
    link, connection = CrdgPanelLink(tmp_path, missions()), native()
    link.label_mode = "selected"
    atomic_json(link.directory / "control.json", {"labels": "selected"})
    readings = {m.vehicle_id: {} for m in missions()}
    snapshot = current()
    link.observe(connection, graph_of(snapshot), snapshot, readings)
    assert all(value == "" for value in link.labels.values())
    atomic_json(link.directory / "control.json", {"focus": None, "highlighted": ["north_000001", "south_000001"], "paused": False, "labels": "selected"})
    link.control()
    link.apply_controls(connection, readings, 5)
    assert link.labels == {"north_000001": "V001", "south_000001": "V002", "north_000002": ""}
    assert set(link.colours.values()) == {SELECTED}
    atomic_json(link.directory / "control.json", {"focus": "north_000001", "highlighted": [], "paused": False, "labels": "selected"})
    link.control()
    connection.vehicle.reset_mock()
    link.apply_controls(connection, readings, 5)
    assert link.labels["north_000001"] == "V001" and link.labels["south_000001"] == ""
    assert ("south_000001", TEXT_KEY, "") in [call.args for call in connection.vehicle.setParameter.call_args_list]


def camera_link(tmp_path):
    link, connection = CrdgPanelLink(tmp_path, missions(), sector_boundary=(0, 0, 180, 160), network_boundary=(0, 0, 600, 500)), native()
    connection.gui = Mock()
    readings = {"north_000001": {"position": (100, 200)}, "south_000001": {"position": (160, 220)}}
    link.current = current()
    return link, connection, readings


def test_camera_uses_current_positions_and_only_once_per_request(tmp_path):
    link, connection, readings = camera_link(tmp_path)
    link.camera_request = {"sequence": 1, "kind": "fit", "vehicles": ["north_000001", "south_000001"]}
    link.apply_controls(connection, readings, 8.5)
    expected = view_boundary([row["position"] for row in readings.values()])
    connection.gui.setBoundary.assert_called_once_with("View #0", *expected)
    assert link.camera_status["time_s"] == 8.5
    readings["north_000001"]["position"] = (1000, 2000)
    link.apply_controls(connection, readings, 9)
    assert connection.gui.setBoundary.call_count == 1
    assert not connection.vehicle.getPosition.called


def test_absent_vehicle_does_not_use_an_old_position(tmp_path):
    link, connection, readings = camera_link(tmp_path)
    link.camera_request = {"sequence": 1, "kind": "fit", "vehicles": ["north_000002"]}
    link.apply_controls(connection, readings, 9)
    assert not connection.gui.setBoundary.called
    assert link.camera_status["vehicles"] == [] and link.camera_status["missing"] == ["north_000002"]
    assert link.camera_status["boundary_m"] is None


def test_follow_and_manual_camera_use_only_gui_controls(tmp_path):
    link, connection, readings = camera_link(tmp_path)
    link.camera_request = {"sequence": 1, "kind": "follow", "vehicles": ["north_000001"]}
    link.apply_controls(connection, readings, 9)
    assert connection.gui.trackVehicle.call_args.args == ("View #0", "north_000001")
    link.camera_request = {"sequence": 2, "kind": "manual", "vehicles": []}
    connection.gui.reset_mock()
    link.apply_controls(connection, readings, 9)
    connection.gui.trackVehicle.assert_called_once_with("View #0", "")
    assert not connection.gui.setBoundary.called


def test_selection_and_camera_work_during_pause_without_physical_step(tmp_path, monkeypatch):
    link, connection, readings = camera_link(tmp_path)
    atomic_json(link.directory / "control.json", {"focus": "north_000001", "highlighted": [], "paused": True,
                                                 "camera": {"sequence": 1, "kind": "fit", "vehicles": ["north_000001"]}})
    def resume(seconds):
        assert seconds == .1
        assert link.camera_status["time_s"] == 9
        command = json.loads((link.directory / "control.json").read_text())
        command["paused"] = False
        atomic_json(link.directory / "control.json", command)
    monkeypatch.setattr("cav_recovery.simulation.crdg_live.time.sleep", resume)
    link.before_step(connection, readings, 9)
    assert link.colours["north_000001"] == SELECTED
    assert not hasattr(connection, "simulationStep")


def test_local_sector_is_graphical_and_smaller_than_network(tmp_path):
    path = tmp_path / "network.xml"
    path.write_text('<net><junction id="3675784999" shape="100,100 120,120"/>'
                    '<junction id="magasin_nguma" shape="130,110 135,120"/>'
                    '<junction id="magasin_oua" shape="135,90 140,100"/></net>')
    before = path.read_bytes()
    bounds = local_sector(path, (0, 0, 600, 500))
    assert bounds[2] - bounds[0] < 600 and bounds[3] - bounds[1] < 500
    assert path.read_bytes() == before
    assert view_boundary([]) is None


def test_filters_never_delete_source_groups():
    from cav_recovery.simulation.crdg_panel import InfoPanel, FILTERS
    panel = InfoPanel.__new__(InfoPanel)
    panel.frame = {"groups": [{"id": "G1", "categories": ["following"]}, {"id": "G2", "categories": ["receiving"]}]}
    panel.filter = Mock()
    for name, category in FILTERS.items():
        panel.filter.get.return_value = name
        expected = 2 if category == "all" else 1 if category in ("following", "receiving") else 0
        assert len(panel.filtered_groups()) == expected
    assert len(panel.frame["groups"]) == 2


def test_follow_stops_when_vehicle_is_no_longer_present(tmp_path):
    link, connection, readings = camera_link(tmp_path)
    link.camera_request = {"sequence": 1, "kind": "follow", "vehicles": ["north_000001"]}
    link.apply_controls(connection, readings, 9)
    connection.gui.reset_mock()
    link.apply_controls(connection, {}, 9.5)
    connection.gui.trackVehicle.assert_called_once_with("View #0", "")
    assert link.tracked_vehicle is None and "absente" in link.camera_status["message"]


def test_display_labels_do_not_turn_unknown_into_a_known_cause():
    from cav_recovery.simulation.crdg_panel import InfoPanel
    panel = InfoPanel.__new__(InfoPanel)
    panel.frame = {"snapshot": {"nodes": [{"id": "resource:X", "resource_type": "receiving_space"}]}}
    edge = {"edge_type": "waits_for", "target": "resource:X"}
    assert panel.relation_label(edge) == "Espace insuffisant"
    panel.frame["snapshot"]["nodes"] = []
    assert panel.relation_label(edge) == "Cause inconnue"


def test_final_view_is_consultable_without_an_extra_physical_step(tmp_path, monkeypatch):
    link, connection, readings = camera_link(tmp_path)
    link.camera_request = {"sequence": 1, "kind": "fit", "vehicles": ["north_000001"]}
    monkeypatch.setattr(link, "control", Mock(return_value=True))
    monkeypatch.setattr("cav_recovery.simulation.crdg_live._enter_pressed", lambda: True)
    monkeypatch.setattr("cav_recovery.simulation.crdg_live.sys.stdin", SimpleNamespace(isatty=lambda: True))
    link.hold_view(connection, readings, "horizon_reached", 9)
    assert link.camera_status["time_s"] == 9 and connection.gui.setBoundary.called
    assert json.loads((link.directory / "clock.json").read_text())["status"] == "horizon_reached"
    assert not hasattr(connection, "simulationStep")


def test_final_view_exits_on_panel_close_without_stdin_reader(tmp_path, monkeypatch):
    link, connection, readings = camera_link(tmp_path)
    monkeypatch.setattr(link, "control", Mock(return_value=False))
    entered = Mock()
    monkeypatch.setattr("cav_recovery.simulation.crdg_live._enter_pressed", entered)
    link.hold_view(connection, readings, "completed", 9)
    entered.assert_not_called()
    assert not connection.gui.setBoundary.called


def test_simulation_waits_for_panel_opening_not_for_a_fake_observation(tmp_path, monkeypatch):
    link = CrdgPanelLink(tmp_path, missions())
    def opened(*args, **kwargs):
        atomic_json(link.directory / "ready.json", {"ready": True})
        return Mock()
    monkeypatch.setattr("cav_recovery.simulation.crdg_live.subprocess.Popen", opened)
    link.start()
    assert link.stats["startup_s"] >= 0
    assert not (link.directory / "frame.json").exists()
    assert json.loads((link.directory / "clock.json").read_text())["time_s"] == 0
    link.log.close()


def test_failed_panel_opening_has_an_explicit_message(tmp_path):
    link = CrdgPanelLink(tmp_path, missions())
    link.process = Mock()
    link.process.poll.return_value = 1
    with pytest.raises(RuntimeError, match="pas pu s'ouvrir"):
        link.wait_ready()


def test_whole_network_view_keeps_distant_branches_without_changing_xml(tmp_path):
    from cav_recovery.simulation.crdg_live import whole_network
    path = tmp_path / "network.xml"
    path.write_text('<net><edge><lane shape="-100,-200 900,700"/></edge></net>')
    before = path.read_bytes()
    left, bottom, right, top = whole_network(path, (0, 0, 100, 100))
    assert left < -100 and bottom < -200 and right > 900 and top > 700
    assert path.read_bytes() == before
