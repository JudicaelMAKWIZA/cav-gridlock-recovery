"""Vérifie les missions ciblées sur Kintambo sans présumer leur résultat."""

import pytest

from cav_recovery.simulation.kintambo_scenarios import case_missions, case_names, read_case
from cav_recovery.simulation.road_network import read_config
from cav_recovery.simulation.traffic_run import run_traffic
from test_traffic import actual_network


def test_cases_do_not_modify_canonical_configuration():
    before = read_config()
    assert case_names() == ("clearance", "crossing", "spillback", "junction_adverse", "loop_nominal", "loop_adverse")
    for name, count in (("clearance", 24), ("crossing", 96), ("spillback", 210)):
        case = read_case(name)
        assert sum(row["count"] for row in case["streams"]) == count
        assert "evidence" in case and "magasin_nguma" in case["sector"]
    assert before == read_config()


def test_loop_pair_has_the_same_missions_before_departure():
    nominal, adverse = read_case("loop_nominal"), read_case("loop_adverse")
    routes = nominal["routes"]
    a, b = case_missions(nominal, routes, 1), case_missions(adverse, routes, 1)
    assert len(a) == len(b) == 96
    assert [(m.vehicle_id, m.scheduled_s, m.route, m.destination, m.exit_gate) for m in a] == [
        (m.vehicle_id, m.scheduled_s, m.route, m.destination, m.exit_gate) for m in b]
    assert len({m.entry_gate for m in a}) == 4
    assert "network_variant" not in nominal
    assert adverse["network_variant"]["keep_clear"] is False


@pytest.mark.parametrize("names", [("loop_nominal", "loop_adverse"), ("crossing", "junction_adverse")])
def test_adverse_network_changes_only_declared_keep_clear_connections(tmp_path, names):
    import shutil
    import xml.etree.ElementTree as ET
    from cav_recovery.simulation.traffic_run import prepare_traffic
    if not shutil.which("netconvert"):
        pytest.skip("netconvert absent ; réseau réel non validé.")
    prepared = []
    for name, folder in zip(names, ("nominal", "adverse")):
        config = read_config()
        config["controlled_case"] = read_case(name)
        directory = tmp_path / folder
        directory.mkdir()
        result, _ = prepare_traffic(config, directory, "LOW", 1, None, None)
        prepared.append(result)
    nominal, adverse = prepared
    assert nominal["routes"] == adverse["routes"]
    assert (tmp_path / "nominal/traffic.rou.xml").read_bytes() == (tmp_path / "adverse/traffic.rou.xml").read_bytes()
    assert (tmp_path / "nominal/network.net.xml").read_bytes() == (tmp_path / "adverse/canonical.net.xml").read_bytes()
    original = ET.parse(tmp_path / "nominal/network.net.xml").getroot()
    altered = ET.parse(tmp_path / "adverse/network.net.xml").getroot()
    changes = adverse["network"]["experimental_variant"]["changes"]
    assert len(changes) > 0
    selected = {(c["from"], c["to"], c["fromLane"], c["toLane"]): c for c in changes}
    for connection in altered.findall("connection"):
        key = tuple(connection.get(name) for name in ("from", "to", "fromLane", "toLane"))
        if key in selected:
            assert connection.get("keepClear") == "false"
            previous = selected[key]["previous_keepClear"]
            if previous is None:
                del connection.attrib["keepClear"]
            else:
                connection.set("keepClear", previous)
    assert ET.tostring(original) == ET.tostring(altered)


def test_native_adverse_convergence_is_observed_then_clears(actual_network, monkeypatch):
    import json
    import shutil
    from cav_recovery.simulation import traffic_run
    directory, _, inspection, routes = actual_network
    def reuse_network(config, output, **kwargs):
        shutil.copyfile(directory / "network.net.xml", output / "network.net.xml")
        return dict(inspection), dict(routes)
    monkeypatch.setattr(traffic_run, "build_network", reuse_network)
    output = directory / "adverse_run"
    result = run_traffic(output, kintambo_case="junction_adverse", seed=1, crdg=True, drain_horizon_s=80.5)
    assert result["status"] == "horizon_reached" and result["counts"]["simulation_time_s"] == 105
    assert result["collision_ids"] == [] and result["counts"]["teleport_starts"] == 0
    assert result["counts"]["missing_without_arrival"] == 0
    assert result["connection_closed"] and result["process_returncode"] == 0
    frames = {row["time_s"]: row for row in map(json.loads, (output / "crdg.jsonl").read_text().splitlines())}
    pair = ("vehicle:kasa_vubu_000000", "vehicle:nguma_000002")
    edge = next(e for e in frames[85]["edges"] if (e["source"], e["target"]) == pair)
    assert edge["edge_type"] == "connection_obstacle" and edge["follow_speed_m_per_s"] < .1
    nodes = {n["id"]: n for n in frames[85]["nodes"]}
    assert nodes[pair[0]]["internal"] and nodes[pair[1]]["internal"]
    assert nodes[pair[0]]["lane"] != nodes[pair[1]]["lane"]
    assert not any((e["source"], e["target"]) == pair for e in frames[95]["edges"])
    observations = list(map(json.loads, (output / "observations.jsonl").read_text().splitlines()))
    physical = next(row for row in observations if row["time_s"] == 95)
    cars = {r["vehicle_id"]: r for r in physical["vehicles"]}
    assert all(cars[item.removeprefix("vehicle:")]["speed_m_per_s"] > .1 for item in pair)


def test_departures_reproducible_and_routes_and_destinations_fixed():
    case = read_case("spillback")
    routes = {stream["route"]: ["start" + str(index), "end" + str(index)] for index, stream in enumerate(case["streams"])}
    a, b = case_missions(case, routes, 1), case_missions(case, dict(reversed(list(routes.items()))), 1)
    assert a == b and len(a) == 210
    assert len({m.vehicle_id for m in a}) == 210
    assert all(m.destination == m.route[-1] and tuple(routes[m.route_id]) == m.route for m in a)
    c = case_missions(case, routes, 2)
    assert [m.scheduled_s for m in a] != [m.scheduled_s for m in c]
    assert {m.vehicle_id: m.destination for m in a} == {m.vehicle_id: m.destination for m in c}
    with pytest.raises(ValueError, match="absente"):
        case_missions(case, {}, 1)
    with pytest.raises(ValueError, match="seed"):
        case_missions(case, routes, -1)


@pytest.mark.parametrize("kwargs", [{"rate": 10}, {"duration_s": 5}, {"demand": "HIGH"}, {"config_path": "x"}, {"scenario": "mutual_yield"}])
def test_case_does_not_silently_accept_other_demand_or_network(tmp_path, kwargs):
    with pytest.raises(ValueError):
        run_traffic(tmp_path / "absent", kintambo_case="clearance", **kwargs)
    assert not (tmp_path / "absent").exists()


def test_cli_exposes_kintambo_case_and_native_state(monkeypatch, tmp_path):
    import runpy
    import sys
    from unittest.mock import Mock
    from pathlib import Path
    from cav_recovery.simulation import traffic_run
    runner = Mock(return_value={"status": "completed", "counts": {}})
    monkeypatch.setattr(traffic_run, "run_traffic", runner)
    monkeypatch.setattr(sys, "argv", ["run_traffic.py", "--kintambo-case", "crossing", "--crdg-scene-at", "81.5", "--crdg-focus", "nguma_000000"])
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/run_traffic.py"))
    assert module["main"]() == 0
    assert runner.call_args.kwargs["scenario"] == "kintambo"
    assert runner.call_args.kwargs["kintambo_case"] == "crossing"
    assert runner.call_args.kwargs["crdg_scene_times"] == (81.5,)


def test_native_scene_export_preserves_traffic_and_reloads_the_same_state(tmp_path):
    import json
    import math
    import shutil
    import traci
    from cav_recovery.simulation.road_network import require_binary
    if not shutil.which("sumo") or not shutil.which("netconvert"):
        pytest.skip("SUMO absent ; scènes réelles non validées.")
    off, on = tmp_path / "off", tmp_path / "on"
    a = run_traffic(off, kintambo_case="clearance", seed=1, crdg=True)
    b = run_traffic(on, kintambo_case="clearance", seed=1, crdg_scene_times=(20.5, 20.5))
    assert a["status"] == b["status"] == "completed"
    assert a["counts"] == b["counts"] and b["counts"]["arrived"] == 24
    assert a["collision_ids"] == b["collision_ids"] == []
    assert b["crdg_scenes"]["recorded_times_s"] == [20.5]
    for name in ("network.net.xml", "traffic.rou.xml", "vehicles.csv", "timeline.csv", "lanes.csv",
                 "observations.jsonl", "crdg.jsonl", "crdg_events.jsonl"):
        assert (off / name).read_bytes() == (on / name).read_bytes()
    folder = on / "crdg_scenes/20.5"
    data = json.loads((folder / "scene.json").read_text())
    traci.start([require_binary("sumo"), "-c", str(folder / "scene.sumocfg"), "--no-step-log", "true"])
    try:
        assert traci.simulation.getTime() == 20.5
        assert set(traci.vehicle.getIDList()) == set(data["readings"])
        for item, row in data["readings"].items():
            point = traci.vehicle.getPosition(item)
            assert math.dist(point, row["position"]) <= 16 * max(math.ulp(v) for v in (*point, *row["position"]))
            assert traci.vehicle.getRoute(item) == tuple(row["route"])
    finally:
        traci.close()
