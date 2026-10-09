"""Vérifie les annotations, pas un diagnostic de blocage."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cav_recovery.simulation.crdg_gui import SceneAnnotations, focused_graph, configure_view, save_scene
from cav_recovery.simulation.traffic_run import run_traffic


def frame(time=2):
    return {"time_s": time, "nodes": [
        {"id": "vehicle:A", "node_type": "vehicle", "vehicle_id": "A"},
        {"id": "vehicle:B", "node_type": "vehicle", "vehicle_id": "B"},
        {"id": "resource:R", "node_type": "resource", "resource_type": "receiving_space", "candidate_lanes": ["out"],
         "max_free_space_m": 0, "required_space_m": 7.5}],
        "edges": [{"source": "vehicle:A", "target": "resource:R", "edge_type": "waits_for", "evidence": "receiving_occupancy"},
                  {"source": "resource:R", "target": "vehicle:B", "edge_type": "occupied_by", "lane": "out"}],
        "waiting_states": {}}


def setup():
    overlay = SceneAnnotations({"out": [(10, 10), (20, 20)]}, focus="A")
    readings = {"A": {"position": (5, 5)}, "B": {"position": (20, 20)}}
    return overlay, readings


def test_current_scene_and_exact_source_edges_are_used():
    overlay, readings = setup()
    current = frame()
    shown = overlay.show(current, readings)
    assert shown["edges"] == current["edges"]
    assert shown["positions"]["vehicle:A"] == readings["A"]["position"]
    assert shown["positions"]["resource:R"] == (10, 10)
    assert overlay.root.findall("poly") and overlay.root.findall("poi")
    connection = SimpleNamespace(simulation=Mock())
    connection.simulation.getTime.return_value = 1.5
    with pytest.raises(ValueError, match="même instant"):
        save_scene(connection, current, readings, "not-created", "A", 2, {})
    connection.simulation.saveState.assert_not_called()


def test_disappeared_edges_and_vehicles_do_not_stay_drawn():
    overlay, readings = setup()
    overlay.show(frame(), readings)
    shown = overlay.show({"time_s": 2.5, "nodes": [], "edges": []}, readings)
    assert shown["edges"] == [] and set(shown["positions"]) == {"vehicle:A"}
    assert not any("est occupée par" in poi.get("type", "") for poi in overlay.root.findall("poi"))


def test_unknown_resource_position_is_not_invented():
    overlay, readings = setup()
    overlay.shapes = {}
    shown = overlay.show(frame(), readings)
    assert shown["unplaced_nodes"] == ["resource:R"]
    assert "resource:R" not in shown["positions"]
    assert shown["edges"] == frame()["edges"]


@pytest.mark.parametrize("state,label", [("signal", "attente liée au signal"), ("unknown", "cause inconnue")])
def test_waiting_without_a_qualified_arc_stays_explicit(state, label):
    overlay, readings = setup()
    shown = overlay.show({"time_s": 2, "nodes": [], "edges": [], "waiting_states": {"A": state}}, readings)
    assert shown["edges"] == []
    assert label in shown["legend"][-1]


def test_absent_vehicle_is_not_created_for_the_drawing():
    overlay, readings = setup()
    with pytest.raises(ValueError, match="présent"):
        overlay.show(frame(), {})


def test_arrived_or_not_inserted_focus_does_not_interrupt_the_run(tmp_path):
    connection = SimpleNamespace(simulation=Mock())
    connection.simulation.getTime.return_value = 2
    from pathlib import Path
    connection.simulation.saveState.side_effect = lambda path: Path(path).write_bytes(b"unit-test-state")
    saved = save_scene(connection, {"time_s": 2, "nodes": [], "edges": []}, {}, tmp_path / "scene", "A", 2, {})
    assert saved["focus_present"] is False
    assert (tmp_path / "scene/scene.json").exists()
    assert not (tmp_path / "scene/annotations.add.xml").exists()


def test_large_neighborhood_is_explicitly_limited():
    current = {"nodes": [{"id": f"vehicle:{i}", "node_type": "vehicle", "vehicle_id": str(i)} for i in range(30)],
               "edges": [{"source": "vehicle:0", "target": f"vehicle:{i}", "edge_type": "leader"} for i in range(1, 30)]}
    shown = focused_graph(current, "0")
    assert len(shown["nodes"]) == 16 and len(shown["omitted_nodes"]) == 14
    assert len(current["edges"]) == 29


def test_view_only_enables_native_text(tmp_path):
    path = tmp_path / "view.xml"
    path.write_text('<viewsettings><scheme name="kintambo"><edges streetName_show="false"/></scheme></viewsettings>')
    configure_view(path)
    import xml.etree.ElementTree as ET
    scheme = ET.parse(path).find("scheme")
    assert scheme.find("pois").get("poiType_show") == "true"
    assert scheme.find("edges").get("streetName_show") == "false"


@pytest.mark.parametrize("options", [{"crdg_scene_times": (1.2,)}, {"crdg_focus": "A"}, {"crdg_scene_times": (-5,)}, {"crdg_depth": 4}])
def test_gui_options_are_refused_before_output_when_inapplicable(tmp_path, options):
    with pytest.raises(ValueError):
        run_traffic(tmp_path / "absent", **options)
    assert not (tmp_path / "absent").exists()
