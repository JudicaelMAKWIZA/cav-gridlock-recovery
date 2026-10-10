"""Tests du trafic Poisson, du réseau et du suivi des véhicules."""

from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import pytest

from cav_recovery.simulation import road_network as network
from cav_recovery.simulation import traffic_run as run
from cav_recovery.simulation.traffic_demand import poisson_missions
from cav_recovery.simulation.sumo_process import close_sumo
from test_vehicle_tracking import Connection, normal_frames, mission


ROUTES = {"entry__a": ["start", "a"], "entry__b": ["start", "b"]}
WEIGHTS = {"entry": {"a": 1, "b": 3}}
RATES = {"entry": 360}


def test_poisson_seed_and_order_are_reproducible():
    first = poisson_missions(ROUTES, WEIGHTS, RATES, 100, 1, "LOW")
    random.seed(345)
    assert first == poisson_missions(dict(reversed(list(ROUTES.items()))),
                                    {"entry": {"b": 3, "a": 1}}, RATES, 100, 1, "LOW")
    assert first != poisson_missions(ROUTES, WEIGHTS, RATES, 100, 2, "LOW")
    expected = random.Random(1).expovariate(0.1)
    assert first[0].sampled_s == expected
    assert first[0].scheduled_s == math.ceil(expected * 1000) / 1000
    assert all(0 <= m.scheduled_s - m.sampled_s < 0.001 for m in first)
    assert len({m.vehicle_id for m in first}) == len(first)
    assert all(0 <= m.scheduled_s < 100 and m.destination == m.route[-1] for m in first)
    assert any(m.scheduled_s % 0.5 for m in first)
    assert first == sorted(first, key=lambda m: (m.scheduled_s, m.vehicle_id))


def test_poisson_rate_and_weights_over_large_sample():
    missions = poisson_missions(ROUTES, WEIGHTS, RATES, 36000, 2, "LOW")
    # On accepte une marge autour des comptes et proportions attendus.
    assert abs(len(missions) - 3600) < 4 * math.sqrt(3600)
    counts = Counter(m.exit_gate for m in missions)
    assert abs(counts["b"] / len(missions) - 0.75) < 0.03


@pytest.mark.parametrize("rate,duration,seed", [
    (-1, 100, 1), (float("nan"), 100, 1), (360, 0, 1), (360, 100, -1),
])
def test_bad_poisson_parameters_rejected(rate, duration, seed):
    with pytest.raises(ValueError):
        poisson_missions(ROUTES, WEIGHTS, {"entry": rate}, duration, seed, "LOW")


def test_missing_route_not_silently_replaced():
    with pytest.raises(ValueError, match="route"):
        poisson_missions({}, WEIGHTS, RATES, 100, 1, "LOW")


def test_source_identity_and_synthetic_adaptations(tmp_path):
    data = gzip.decompress((network.SCENARIO_DIR / "roads.osm.gz").read_bytes())
    config = network.read_config()
    assert hashlib.sha256(data).hexdigest() == config["source"]["roads_sha256_uncompressed"]
    target = tmp_path / "adapted.osm"
    network.adapt_roads(config, target)
    original = ET.fromstring(data)
    adapted = ET.parse(target).getroot()
    assert [n.attrib for n in original.findall("node")] == [n.attrib for n in adapted.findall("node")]
    before = {w.get("id"): w for w in original.findall("way")}
    for way in adapted.findall("way"):
        old = before[way.get("id")]
        assert [n.attrib for n in old.findall("nd")] == [n.attrib for n in way.findall("nd")]
        tags = {t.get("k"): t.get("v") for t in way.findall("tag")}
        old_tags = {t.get("k"): t.get("v") for t in old.findall("tag")}
        assert tags.get("oneway") == old_tags.get("oneway")
        if tags["highway"] == "primary":
            assert tags["lanes"] == ("4" if tags.get("oneway") == "yes" else "8")


@pytest.fixture(scope="module")
def actual_network(tmp_path_factory):
    if not shutil.which("netconvert"):
        pytest.skip("netconvert indisponible ; intégration non validée.")
    directory = tmp_path_factory.mktemp("kintambo")
    config = network.read_config()
    inspection, routes = network.build_network(config, directory)
    return directory, config, inspection, routes


def test_real_network_and_routes(actual_network):
    import sumolib
    directory, config, inspection, routes = actual_network
    net = sumolib.net.readNet(str(directory / "network.net.xml"), withPrograms=True)
    assert inspection["branching_junctions"] > 4
    assert len(routes) == sum(len(destinations) for destinations in config["demand"]["destination_weights"].values())
    for route_id, ids in routes.items():
        origin, destination = route_id.split("__")
        assert ids[0] == config["demand"]["entries"][origin]["edge"]
        assert ids[-1] == config["demand"]["entries"][destination]["exit"]
        assert len(ids) == len(set(ids))
        for first, second in zip(ids, ids[1:]):
            connections = net.getEdge(first).getOutgoing().get(net.getEdge(second), [])
            assert any(c.getFromLane().allows("passenger") and c.getToLane().allows("passenger")
                       and c.getDirection() != "t" for c in connections)
    root = ET.parse(directory / "network.net.xml").getroot()
    assert len(root.findall("tlLogic")) > 1
    for tls in root.findall("tlLogic"):
        assert tls.get("type") == "static"
        phases = tls.findall("phase")
        assert sum(float(p.get("duration")) for p in phases) == 90
        assert any(p.get("state").count("G") > 1 for p in phases)
    assert all(c.get("keepClear", "1") not in ("0", "false") for c in root.findall("connection"))


def test_preparation_xml_and_deterministic_missions(actual_network, monkeypatch, tmp_path):
    _, config, inspection, routes = actual_network
    monkeypatch.setattr(run, "build_network", lambda *args, **kwargs: (inspection, routes))
    prepared, first = run.prepare_traffic(config, tmp_path, "LOW", 1, 60, None)
    _, second = run.prepare_traffic(config, tmp_path, "LOW", 1, 60, None)
    assert first == second
    root = ET.parse(tmp_path / "traffic.rou.xml").getroot()
    assert root.find("vType").get("guiShape") == "passenger/sedan"
    assert [v.get("id") for v in root.findall("vehicle")] == [m.vehicle_id for m in first]
    config_xml = ET.parse(tmp_path / "simulation.sumocfg").getroot()
    assert config_xml.find("processing/time-to-teleport").get("value") == "-1"
    assert config_xml.find("processing/collision.check-junctions").get("value") == "true"
    street_view = ET.parse(tmp_path / "view.xml").find("scheme/edges")
    assert street_view.get("streetName_show") == "false"
    assert street_view.get("streetName_constantSize") == "true"
    assert street_view.get("streetName_onlySelected") == "false"
    assert ET.parse(tmp_path / "view.xml").find("scheme").get("name") == "kintambo"
    assert ET.parse(tmp_path / "view.xml").find("scheme/vehicles").get("vehicleQuality") == "2"
    assert prepared["seed"] == 1


def test_nonempty_results_preserved(tmp_path):
    path = tmp_path / "keep.txt"
    path.write_text("important")
    with pytest.raises(ValueError, match="absent ou vide"):
        run.run_traffic(tmp_path)
    assert path.read_text() == "important"


@pytest.mark.parametrize("name,included", [
    ("crdg.py", True),
    ("blockage_observation.py", True),
    ("kintambo_scenarios.py", True), ("sumo_snapshot.py", True),
    ("sumo_view.py", True),
    ("crdg_live.py", True), ("crdg_panel.py", True),
    ("road_network.py", True), ("traffic_demand.py", True), ("traffic_run.py", True),
    ("sumo_process.py", True), ("vehicle_tracking.py", True), ("sumo_smoke.py", False),
])
def test_code_digest_covers_actual_execution(monkeypatch, name, included):
    original = Path.read_bytes
    before = run.code_provenance()["sha256"]
    monkeypatch.setattr(Path, "read_bytes",
                        lambda path: original(path) + (b"\n" if path.name == name else b""))
    assert (before != run.code_provenance()["sha256"]) is included


@pytest.mark.parametrize("status,code", [("completed", 0), ("horizon_reached", 3), ("failed", 1)])
def test_cli_reports_result(monkeypatch, tmp_path, status, code):
    spec = importlib.util.spec_from_file_location("cli", Path(__file__).parents[1] / "scripts/run_traffic.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    invoke = Mock(return_value={"status": status, "counts": {}})
    monkeypatch.setattr(cli, "run_traffic", invoke)
    monkeypatch.setattr(sys, "argv", ["demo", "--demand", "HIGH", "--seed", "2", "--gui",
                                     "--output-dir", str(tmp_path / "result")])
    assert cli.main() == code
    assert invoke.call_args.kwargs["gui"] is True
    assert invoke.call_args.kwargs["seed"] == 2


def test_cli_street_names_are_opt_in(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("cli", Path(__file__).parents[1] / "scripts/run_traffic.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    invoke = Mock(return_value={"status": "completed", "counts": {}})
    monkeypatch.setattr(cli, "run_traffic", invoke)
    for flag, expected in (([], False), (["--street-names"], True)):
        monkeypatch.setattr(sys, "argv", ["traffic", "--gui", "--output-dir", str(tmp_path)] + flag)
        assert cli.main() == 0
        assert invoke.call_args.kwargs["street_names"] is expected


def test_cli_crdg_is_opt_in(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("cli", Path(__file__).parents[1] / "scripts/run_traffic.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    invoke = Mock(return_value={"status": "completed", "counts": {}})
    monkeypatch.setattr(cli, "run_traffic", invoke)
    for flag, expected in (([], False), (["--crdg"], True)):
        monkeypatch.setattr(sys, "argv", ["traffic", "--output-dir", str(tmp_path)] + flag)
        assert cli.main() == 0
        assert invoke.call_args.kwargs["crdg"] is expected


def test_street_names_only_change_view(actual_network, monkeypatch, tmp_path):
    _, config, inspection, routes = actual_network
    monkeypatch.setattr(run, "build_network", lambda *args, **kwargs: (inspection, routes))
    first, missions = run.prepare_traffic(config, tmp_path, "LOW", 1, 60, None)
    scientific_files = {name: (tmp_path / name).read_bytes() for name in ("traffic.rou.xml", "simulation.sumocfg")}
    second, labeled = run.prepare_traffic(config, tmp_path, "LOW", 1, 60, None, street_names=True)
    assert ET.parse(tmp_path / "view.xml").find("scheme/edges").get("streetName_show") == "true"
    assert missions == labeled and first["routes"] == second["routes"]
    assert all((tmp_path / name).read_bytes() == data for name, data in scientific_files.items())


def test_focus_keeps_local_links_ports_and_rejects_unknown_junctions(actual_network):
    import sumolib
    directory, config, inspection, routes = actual_network
    net = sumolib.net.readNet(str(directory / "network.net.xml"))
    selected = {node for nodes in config["network"]["active_junctions"].values() for node in nodes}
    assert {node.getID() for node in net.getNodes()} == selected
    assert set(inspection["active_edges"]) == {edge.getID() for edge in net.getEdges()}
    assert {edge.getName() for edge in net.getEdges()} >= {
        "Avenue Colonel Mondjiba", "Avenue Nguma", "Avenue Kasa-Vubu",
        "Avenue de l’OUA", "Avenue des Ecuries", "Avenue Transversale", "Avenue Yoseki", "Avenue du Parc"}
    for loop in (["956500885#0", "363410577", "-23386395#0", "680630199#6", "1053620734"],
                 ["427630648#1", "427630648#2", "-57358917#1", "4642209#0", "4642209#1"]):
        edges = [net.getEdge(item) for item in loop]
        assert all(any(c.getDirection() != "t" and c.getFromLane().allows("passenger")
                       and c.getToLane().allows("passenger") for c in first.getOutgoing().get(second, []))
                   for first, second in zip(edges, edges[1:] + edges[:1]))
    for entry, port in config["demand"]["entries"].items():
        assert net.getEdge(port["edge"]).getLength() >= 15
        assert net.getEdge(port["exit"]).getLength() >= 15
        assert len([key for key in routes if key.startswith(entry + "__")]) >= 2
    with pytest.raises(ValueError, match="absentes"):
        network.select_active_edges(net, {"absent": ["unknown"]})
    with pytest.raises(ValueError, match="aucune route"):
        network.select_active_edges(net, {})


def test_real_light_traffic_drains(tmp_path):
    if not shutil.which("sumo") or not shutil.which("netconvert"):
        pytest.skip("SUMO indisponible ; intégration non validée.")
    result = run.run_traffic(tmp_path / "light", duration_s=60, seed=1)
    assert result["status"] == "completed", result["reason"]
    counts = result["counts"]
    assert counts["scheduled"] == counts["departed"] == counts["arrived"] > 0
    assert counts["active"] == counts["pending"] == counts["teleport_starts"] == counts["teleport_ends"] == 0
    assert result["collision_ids"] == []
    assert result["connection_closed"] and result["process_stopped"]
    assert result["process_returncode"] == 0 and not result["forced_process_stop"]


def fake_run(monkeypatch, tmp_path, frames, gui=False, cleanup_error=False):
    import traci
    connection = Connection(frames)
    connection.lane = SimpleNamespace(getIDList=lambda: [])
    connection.gui = SimpleNamespace(setBoundary=Mock(), setSchema=Mock())
    config = network.read_config()
    prepared = {"configuration": config, "duration_s": 0.5, "rates_veh_per_hour_per_entry": {"entry": 90},
                "routes": {"straight": ["start", "end"]}, "network": {"view_boundary_m": [-1, -1, 1, 1]}}
    def prepare(config, directory, *args, **kwargs):
        for name in ("scenario.json", "network.net.xml", "traffic.rou.xml", "simulation.sumocfg", "view.xml"):
            (directory / name).write_text("synthetic")
        return prepared, [mission()]
    monkeypatch.setattr(run, "prepare_traffic", prepare)
    monkeypatch.setattr(run, "require_binary", lambda name: name)
    monkeypatch.setattr(run, "code_provenance", lambda: {"sha256": "synthetic"})
    monkeypatch.setattr(run, "subscribed_readings", lambda connection: None)
    monkeypatch.setattr(run, "verify_trips", lambda path, ledger: {i: {"arrival_s": 1} for i in ledger.arrivals})
    process = Mock()
    process.poll.return_value = 0
    process.wait.side_effect = [OSError("attente"), None] if cleanup_error else None
    launch = Mock(return_value=process)
    monkeypatch.setattr(run.subprocess, "Popen", launch)
    monkeypatch.setattr(traci, "connect", lambda **kwargs: connection)
    result = run.run_traffic(tmp_path / "result", gui=gui, gui_delay_ms=7, drain_horizon_s=1)
    return result, connection, process, launch.call_args.args[0]


@pytest.mark.parametrize("frames,message", [
    ([normal_frames()[0], {}], "Disparition"),
    ([{"departed": ["car"], "active": {"car": {"route": ("start", "wrong")}}}], "Route"),
    ([{"teleport_starts": ["car"]}], "Téléportation"),
    ([{"collisions": ["car"]}], "Collision"),
])
def test_runner_integrity_failure_keeps_diagnostics_and_closes(monkeypatch, tmp_path, frames, message):
    result, connection, process, _ = fake_run(monkeypatch, tmp_path, frames)
    assert result["status"] == "failed" and message in result["reason"]
    assert result["failure_observation"] is not None
    assert connection.closed and result["process_stopped"]
    assert process.wait.called
    saved = json.loads((tmp_path / "result/summary.json").read_text())
    assert saved["counts"] == result["counts"]


def test_runner_horizon_not_reported_as_gridlock(monkeypatch, tmp_path):
    result, connection, _, _ = fake_run(monkeypatch, tmp_path, [{}, {}, {}])
    assert result["status"] == "horizon_reached"
    assert result["gridlock"] == "not_evaluated"
    assert result["counts"]["delayed_not_inserted"] == 1
    assert result["remaining_ids"] == ["car"] and connection.closed


def test_runner_cleanup_errors_are_failures(monkeypatch, tmp_path):
    result, connection, process, _ = fake_run(monkeypatch, tmp_path, normal_frames(), cleanup_error=True)
    assert result["status"] == "failed"
    assert result["simulation_outcome"] == "completed"
    assert "Fermeture normale non confirmée" in result["reason"]
    assert result["cleanup_errors"] and result["forced_process_stop"]
    assert process.terminate.called and connection.closed


def test_gui_only_changes_display(monkeypatch, tmp_path):
    closer = Mock(wraps=close_sumo)
    monkeypatch.setattr(run, "close_sumo", closer)
    result, connection, process, command = fake_run(monkeypatch, tmp_path, normal_frames(), gui=True)
    closer.assert_called_once_with(connection, process, result)
    assert result["status"] == "completed"
    assert command[0] == "sumo-gui"
    assert command[command.index("--delay") + 1] == "7"
    assert command[command.index("--start") + 1] == "true"
    assert result["step_s"] == 0.5 and result["seed"] == 1
    connection.gui.setSchema.assert_called_once_with("View #0", "kintambo")
    connection.gui.setBoundary.assert_called_once_with("View #0", -1, -1, 1, 1)


def test_preparation_failure_retains_diagnostic(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "prepare_traffic", Mock(side_effect=RuntimeError("conversion")))
    with pytest.raises(RuntimeError, match="conversion"):
        run.run_traffic(tmp_path / "failure")
    saved = json.loads((tmp_path / "failure/preparation_error.json").read_text())
    assert saved == {"status": "failed", "reason": "conversion"}


def test_physical_samples_keep_units_and_do_not_diagnose_gridlock():
    connection = SimpleNamespace(
        vehicle=SimpleNamespace(getLeader=lambda item: None,
                                getNextTLS=lambda item: [("light", 0, 2, "r")]),
        lane=SimpleNamespace(getLastStepOccupancy=lambda lane: 0.65))
    ledger = SimpleNamespace(time_s=5, observed_active={"first", "second"})
    readings = {item: {"lane": "lane", "road_id": "road", "lane_position": position,
                       "speed": 0, "distance": position, "route_index": 0}
                for item, position in (("first", 7), ("second", 20))}
    previous = {}
    vehicles, lanes = run.sample_physics(connection, ledger, {"lane": 100}, previous, readings)
    assert all(v["progress_m"] is None for v in vehicles)
    assert all(v["leader_id"] is None for v in vehicles)
    assert lanes[0]["occupancy_ratio"] == 0.65
    assert lanes[0]["upstream_free_m"] == 2
    assert lanes[0]["queue_extent_m"] == 98
    assert lanes[0]["queue_reaches_upstream"] is True
    assert "gridlock" not in lanes[0]  # Une file au rouge ne prouve pas un gridlock.
    ledger.time_s = 10
    vehicles, _ = run.sample_physics(connection, ledger, {"lane": 100}, previous, readings)
    assert all(v["progress_m"] == 0 for v in vehicles)


def test_one_halted_vehicle_is_not_an_upstream_queue():
    connection = SimpleNamespace(vehicle=SimpleNamespace(getLeader=lambda item: None,
                                getNextTLS=lambda item: []),
                                 lane=SimpleNamespace(getLastStepOccupancy=lambda lane: 0.05))
    ledger = SimpleNamespace(time_s=5, observed_active={"car"})
    readings = {"car": {"lane": "lane", "road_id": "road", "lane_position": 5,
                        "speed": 0, "distance": 5, "route_index": 0}}
    _, lanes = run.sample_physics(connection, ledger, {"lane": 100}, {}, readings)
    assert lanes[0]["halting"] == 1 and lanes[0]["queue_reaches_upstream"] is False


def test_street_name_survives_source_adaptation_and_conversion(actual_network):
    directory, _, _, _ = actual_network
    source = ET.fromstring(gzip.decompress((network.SCENARIO_DIR / "roads.osm.gz").read_bytes()))
    expected = source.find("way[@id='427630648']/tag[@k='name']").get("v")
    assert expected == "Avenue Colonel Mondjiba"
    assert ET.parse(directory / "adapted.osm").find("way[@id='427630648']/tag[@k='name']").get("v") == expected
    edges = [e for e in ET.parse(directory / "network.net.xml").getroot().findall("edge")
             if e.get("id", "").startswith("427630648")]
    assert edges and all(e.get("name") == expected for e in edges)


@pytest.mark.parametrize("length,short,shared", [(0.2, True, True), (12.81, True, False), (15, False, False)])
def test_lane_preparation_preserves_movements_and_only_couples_links_without_buffer(length, short, shared):
    root = ET.fromstring(f"""<net>
      <edge id="approach" from="before" to="a"><lane index="0" length="100" /></edge>
      <edge id="connector" from="a" to="b">
        <lane index="0" length="{length}" /><lane index="1" length="{length}" />
      </edge>
      <edge id="exit" from="b" to="after"><lane index="0" length="100" /></edge>
      <connection from="approach" to="connector" fromLane="0" toLane="0" />
      <connection from="connector" to="exit" fromLane="0" toLane="0" />
      <connection from="connector" to="exit" fromLane="1" toLane="0" />
      <junction id="a" type="traffic_light" /><junction id="b" type="traffic_light" />
    </net>""")
    edges, connections, nodes, report = network._connection_patches(root, 7.5, {"a"})
    assert bool(edges.findall("edge")) is short
    assert bool(nodes.findall("node")) is shared
    assert report["added_lane_connections"] == ([('approach', 'connector', 0, 1)] if short else [])
    movements = {(c.get("from"), c.get("to")) for c in root.findall("connection")}
    assert all((c.get("from"), c.get("to")) in movements for c in connections)
    assert all(c.get("contPos") == "0" and c.get("changeLeft") == c.get("changeRight") == "emergency"
               for c in connections)
    assert report["shared_controller_junctions"] == ([["a", "b"]] if shared else [])


def test_short_connectors_and_intersections_forbid_passenger_lane_changes(actual_network):
    directory, _, inspection, _ = actual_network
    root = ET.parse(directory / "network.net.xml").getroot()
    short = set(inspection["connection_rules"]["short_connectors"])
    assert "-957109397" in short
    for edge in root.findall("edge"):
        if edge.get("id") in short or edge.get("function") == "internal":
            lanes = edge.findall("lane")
            for index, lane in enumerate(lanes):
                # SUMO peut omettre l'interdiction quand il n'y a pas de voie voisine.
                if index + 1 < len(lanes):
                    if lane.get("changeLeft") != "emergency":
                        # netconvert laisse changer de voie pour éviter une impasse.
                        assert not root.findall(
                            f"connection[@from='{edge.get('id')}'][@fromLane='{index}']")
                        assert "Ignoring changeLeft prohibition" in (
                            directory / "continuity.log").read_text()
                if index > 0:
                    assert lane.get("changeRight") == "emergency"
    pairs = {tuple(row[:2]) for row in inspection["connection_rules"]["added_lane_connections"]}
    original = ET.parse(directory / "unsignalized.net.xml").getroot()
    movements = {(c.get("from"), c.get("to")) for c in original.findall("connection")}
    assert pairs.issubset(movements)


def test_coupled_signal_programs_are_static_and_all_movements_have_service(actual_network):
    directory, _, inspection, _ = actual_network
    assert inspection["connection_rules"]["shared_controller_junctions"] == [
        ["3675784999", "magasin_west"], ["magasin_nguma", "magasin_oua"]]
    root = ET.parse(directory / "network.net.xml").getroot()
    for tls in root.findall("tlLogic"):
        states = [p.get("state") for p in tls.findall("phase")]
        for index in range(len(states[0])):
            assert any(state[index] in "Gg" for state in states)


def test_real_medium_collision_regression(tmp_path):
    if not shutil.which("sumo") or not shutil.which("netconvert"):
        pytest.skip("SUMO absent ; intégration non validée.")
    result = run.run_traffic(tmp_path / "medium", demand="MEDIUM", seed=1)
    # Les entrées sont plus proches, mais le débit total attendu ne change pas.
    assert result["counts"]["scheduled"] == 606
    assert result["counts"]["simulation_time_s"] > 189
    assert result["status"] in ("completed", "horizon_reached"), result["reason"]
    assert result["collision_ids"] == []
    assert result["counts"]["teleport_starts"] == result["counts"]["teleport_ends"] == 0
    assert result["counts"]["missing_without_arrival"] == 0
    assert result["connection_closed"] and result["process_stopped"]
