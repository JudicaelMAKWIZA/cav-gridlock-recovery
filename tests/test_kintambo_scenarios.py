"""Vérifie les missions ciblées sur Kintambo sans présumer leur résultat."""

import pytest

from cav_recovery.simulation.kintambo_scenarios import case_missions, case_names, read_case
from cav_recovery.simulation.road_network import read_config
from cav_recovery.simulation.traffic_run import run_traffic


def test_cases_do_not_modify_canonical_configuration():
    before = read_config()
    assert case_names() == ("clearance", "crossing", "spillback")
    for name, count in (("clearance", 24), ("crossing", 96), ("spillback", 210)):
        case = read_case(name)
        assert sum(row["count"] for row in case["streams"]) == count
        assert "evidence" in case and "magasin_nguma" in case["sector"]
    assert before == read_config()


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


def test_cli_exposes_kintambo_case_and_native_annotations(monkeypatch, tmp_path):
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
    from cav_recovery.simulation.crdg_gui import render_scene
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
    focus = sorted(data["readings"])[0]
    render_scene(folder, focus=focus)
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
