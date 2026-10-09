"""Vérifie les scénarios contrôlés sans leur attribuer de diagnostic positif."""

import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import pytest

from cav_recovery.simulation.intersection_scenarios import read_scenario, scenario_names
from cav_recovery.simulation.traffic_run import prepare_intersection, run_traffic, VEHICLE_TYPE


def test_targets_and_evaluation_reserve_are_explicit():
    assert set(scenario_names()) == {"priority_wait", "priority_starvation", "mutual_yield",
                                    "junction_spillback", "priority_wait_holdout"}
    assert read_scenario("priority_wait_holdout")["case"]["role"] == "evaluation_reserved"
    assert "non présumé" in read_scenario("mutual_yield")["case"]["aim"]
    with pytest.raises(ValueError, match="inconnu"):
        read_scenario("absent")


@pytest.mark.parametrize("name", ["priority_wait", "priority_starvation", "mutual_yield", "junction_spillback"])
def test_prepared_network_missions_and_safety_are_reproducible(tmp_path, name):
    if not shutil.which("netconvert"):
        pytest.skip("netconvert absent ; préparation réelle non validée.")
    results = []
    for folder in ("a", "b"):
        output = tmp_path / folder
        output.mkdir()
        results.append(prepare_intersection(read_scenario(name), output, 1))
    a, b = results
    assert a[1] == b[1]
    assert (tmp_path / "a/network.net.xml").read_bytes() == (tmp_path / "b/network.net.xml").read_bytes()
    assert (tmp_path / "a/traffic.rou.xml").read_bytes() == (tmp_path / "b/traffic.rou.xml").read_bytes()
    assert len({m.vehicle_id for m in a[1]}) == len(a[1]) > 0
    assert all(m.route[-1] == m.destination and len(m.route) >= 2 for m in a[1])
    root = ET.parse(tmp_path / "a/traffic.rou.xml")
    assert root.find("vType").attrib == VEHICLE_TYPE
    assert not root.findall(".//stop")
    config = ET.parse(tmp_path / "a/simulation.sumocfg")
    assert config.find("processing/time-to-teleport").get("value") == "-1"
    assert config.find("processing/collision.check-junctions").get("value") == "true"
    assert config.find("time/step-length").get("value") == "0.5"
    assert a[0]["phenomenon_observed"] == "not_evaluated"


def test_seed_changes_explicit_departures_not_destinations(tmp_path):
    if not shutil.which("netconvert"):
        pytest.skip("netconvert absent.")
    missions = []
    for seed in (1, 2):
        output = tmp_path / str(seed)
        output.mkdir()
        missions.append(prepare_intersection(read_scenario("priority_wait"), output, seed)[1])
    assert [m.scheduled_s for m in missions[0]] != [m.scheduled_s for m in missions[1]]
    assert {m.vehicle_id: m.destination for m in missions[0]} == {m.vehicle_id: m.destination for m in missions[1]}


def test_controlled_options_do_not_silently_change_canonical_demand(tmp_path):
    for kwargs in ({"rate": 20}, {"duration_s": 10}, {"demand": "HIGH"}, {"config_path": "other.json"}):
        with pytest.raises(ValueError, match="réservés"):
            run_traffic(tmp_path / "absent", scenario="priority_wait", **kwargs)
    assert not (tmp_path / "absent").exists()


def test_real_negative_case_drains_and_cdrg_does_not_change_it(tmp_path):
    if not shutil.which("netconvert") or not shutil.which("sumo"):
        pytest.skip("SUMO absent ; scénario réel non validé.")
    off, on = tmp_path / "off", tmp_path / "on"
    a = run_traffic(off, scenario="priority_wait", seed=1)
    b = run_traffic(on, scenario="priority_wait", seed=1, crdg=True)
    assert a["status"] == b["status"] == "completed"
    assert a["counts"] == b["counts"]
    assert b["counts"]["arrived"] == b["counts"]["scheduled"] == 52
    assert b["counts"]["teleport_starts"] == b["counts"]["missing_without_arrival"] == 0
    assert a["collision_ids"] == b["collision_ids"] == []
    assert b["connection_closed"] and b["process_stopped"] and not b["cleanup_errors"]
    for name in ("network.net.xml", "traffic.rou.xml", "vehicles.csv", "timeline.csv", "observations.jsonl", "lanes.csv"):
        assert (off / name).read_bytes() == (on / name).read_bytes()
    summary = json.loads((on / "crdg_summary.json").read_text())
    assert summary["snapshots_with_closed_cycle_candidates"] == 0


def test_cli_preserves_kintambo_default_and_accepts_controlled_case(monkeypatch, tmp_path):
    import runpy
    import sys
    from unittest.mock import Mock
    from cav_recovery.simulation import traffic_run
    runner = Mock(return_value={"status": "completed", "counts": {}})
    monkeypatch.setattr(traffic_run, "run_traffic", runner)
    monkeypatch.setattr(sys, "argv", ["run_traffic.py", "--scenario", "mutual_yield", "--crdg", "--output-dir", str(tmp_path)])
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/run_traffic.py"))
    assert module["main"]() == 0
    assert runner.call_args.kwargs["scenario"] == "mutual_yield"
    assert runner.call_args.kwargs["crdg"] is True
