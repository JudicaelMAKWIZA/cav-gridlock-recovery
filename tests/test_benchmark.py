"""Demande Poisson, réseau publié, intégrité et lancement du benchmark."""

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

from cav_recovery.simulation import benchmark_network as network
from cav_recovery.simulation import benchmark_run as run
from cav_recovery.simulation.synthetic_demand import poisson_missions
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
    # Tolérances statistiques sur un tirage fixe, pas des effectifs imposés.
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
    assert hashlib.sha256(data).hexdigest() == network.SOURCE_SHA256
    config = network.read_config()
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
    assert inspection["branching_junctions"] > 20
    assert len(routes) == 56
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
    monkeypatch.setattr(run, "build_network", lambda *args: (inspection, routes))
    prepared, first = run.prepare_experiment(config, tmp_path, "LOW", 1, 60, None)
    _, second = run.prepare_experiment(config, tmp_path, "LOW", 1, 60, None)
    assert first == second
    root = ET.parse(tmp_path / "traffic.rou.xml").getroot()
    assert root.find("vType").get("guiShape") == "passenger/sedan"
    assert [v.get("id") for v in root.findall("vehicle")] == [m.vehicle_id for m in first]
    config_xml = ET.parse(tmp_path / "simulation.sumocfg").getroot()
    assert config_xml.find("processing/time-to-teleport").get("value") == "-1"
    assert config_xml.find("processing/collision.check-junctions").get("value") == "true"
    assert ET.parse(tmp_path / "view.xml").find("scheme/vehicles").get("vehicleQuality") == "2"
    assert prepared["seed"] == 1


def test_nonempty_results_preserved(tmp_path):
    path = tmp_path / "keep.txt"
    path.write_text("important")
    with pytest.raises(ValueError, match="absent ou vide"):
        run.run_experiment(tmp_path)
    assert path.read_text() == "important"


@pytest.mark.parametrize("name,included", [
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
    spec = importlib.util.spec_from_file_location("cli", Path(__file__).parents[1] / "scripts/run_experiment.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    invoke = Mock(return_value={"status": status, "counts": {}})
    monkeypatch.setattr(cli, "run_experiment", invoke)
    monkeypatch.setattr(sys, "argv", ["demo", "--demand", "HIGH", "--seed", "2", "--gui",
                                     "--output-dir", str(tmp_path / "result")])
    assert cli.main() == code
    assert invoke.call_args.kwargs["gui"] is True
    assert invoke.call_args.kwargs["seed"] == 2


def test_real_light_traffic_drains(tmp_path):
    if not shutil.which("sumo") or not shutil.which("netconvert"):
        pytest.skip("SUMO indisponible ; intégration non validée.")
    result = run.run_experiment(tmp_path / "light", duration_s=60, seed=1)
    assert result["status"] == "completed", result["reason"]
    counts = result["counts"]
    assert counts["scheduled"] == counts["departed"] == counts["arrived"] > 0
    assert counts["active"] == counts["pending"] == counts["teleport_starts"] == counts["teleport_ends"] == 0
    assert result["collision_ids"] == []
    assert result["connection_closed"] and result["process_stopped"]
    assert result["process_returncode"] == 0 and not result["forced_process_stop"]


def fake_run(monkeypatch, tmp_path, frames, gui=False, cleanup_error=False):
    connection = Connection(frames)
    connection.lane = SimpleNamespace(getIDList=lambda: [])
    connection.gui = SimpleNamespace(setOffset=Mock(), setZoom=Mock())
    config = network.read_config()
    prepared = {"duration_s": 0.5, "rates_veh_per_hour_per_entry": {"entry": 90},
                "routes": {"straight": ["start", "end"]}, "network": {"center_xy_m": [0, 0]}}
    def prepare(config, directory, *args):
        for name in ("scenario.json", "network.net.xml"):
            (directory / name).write_text("synthetic")
        return prepared, [mission()]
    monkeypatch.setattr(run, "prepare_experiment", prepare)
    monkeypatch.setattr(run, "require_binary", lambda name: name)
    monkeypatch.setattr(run, "code_provenance", lambda: {"sha256": "synthetic"})
    monkeypatch.setattr(run, "subscribed_readings", lambda connection: None)
    monkeypatch.setattr(run, "verify_trips", lambda path, ledger: {i: {"arrival_s": 1} for i in ledger.arrivals})
    process = Mock()
    process.poll.return_value = 0
    process.wait.side_effect = [OSError("attente"), None] if cleanup_error else None
    launch = Mock(return_value=process)
    monkeypatch.setattr(run.subprocess, "Popen", launch)
    monkeypatch.setattr(sys.modules["traci"], "connect", lambda **kwargs: connection)
    result = run.run_experiment(tmp_path / "result", gui=gui, gui_delay_ms=7, drain_horizon_s=1)
    return result, connection, process, launch.call_args.args[0]


@pytest.mark.parametrize("frames,message", [
    ([normal_frames()[0], {}], "Disparition"),
    ([{"departed": ["car"], "active": {"car": {"route": ("start", "wrong")}}}], "Route"),
    ([{"teleport_starts": ["car"]}], "Téléportation"),
    ([{"collisions": ["car"]}], "Collision"),
])
def test_runner_integrity_failure_keeps_diagnostics_and_closes(monkeypatch, tmp_path, frames, message):
    import traci
    result, connection, process, _ = fake_run(monkeypatch, tmp_path, frames)
    assert result["status"] == "failed" and message in result["reason"]
    assert result["failure_observation"] is not None
    assert connection.closed and result["process_stopped"]
    assert process.wait.called
    saved = json.loads((tmp_path / "result/summary.json").read_text())
    assert saved["counts"] == result["counts"]


def test_runner_horizon_not_reported_as_gridlock(monkeypatch, tmp_path):
    import traci
    result, connection, _, _ = fake_run(monkeypatch, tmp_path, [{}, {}, {}])
    assert result["status"] == "horizon_reached"
    assert result["gridlock"] == "not_evaluated"
    assert result["counts"]["delayed_not_inserted"] == 1
    assert result["remaining_ids"] == ["car"] and connection.closed


def test_runner_cleanup_errors_are_failures(monkeypatch, tmp_path):
    import traci
    result, connection, process, _ = fake_run(monkeypatch, tmp_path, normal_frames(), cleanup_error=True)
    assert result["status"] == "failed"
    assert result["cleanup_errors"] and result["forced_process_stop"]
    assert process.terminate.called and connection.closed


def test_gui_only_changes_display(monkeypatch, tmp_path):
    import traci
    closer = Mock(wraps=close_sumo)
    monkeypatch.setattr(run, "close_sumo", closer)
    result, connection, process, command = fake_run(monkeypatch, tmp_path, normal_frames(), gui=True)
    closer.assert_called_once_with(connection, process, result)
    assert result["status"] == "completed"
    assert command[0] == "sumo-gui"
    assert command[command.index("--delay") + 1] == "7"
    assert command[command.index("--start") + 1] == "true"
    assert result["step_s"] == 0.5 and result["seed"] == 1


def test_preparation_failure_retains_diagnostic(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "prepare_experiment", Mock(side_effect=RuntimeError("conversion")))
    with pytest.raises(RuntimeError, match="conversion"):
        run.run_experiment(tmp_path / "failure")
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
    assert "gridlock" not in lanes[0]  # Le rouge et une file ne suffisent pas.
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
