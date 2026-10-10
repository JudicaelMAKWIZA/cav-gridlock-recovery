"""Le graphe reste visible pendant les sélections et commandes de caméra."""

import copy
import json
import os

import pytest

from cav_recovery.simulation.crdg_live import atomic_json
from cav_recovery.simulation.crdg_panel import InfoPanel


@pytest.fixture
def panel(tmp_path):
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("Affichage Tk absent ; interface réelle non vérifiée.")
    atomic_json(tmp_path / "ids.json", {"a": "V001", "b": "V002", "c": "V003", "d": "V004"})
    atomic_json(tmp_path / "control.json", {"focus": None, "highlighted": [], "paused": True, "labels": "selected"})
    atomic_json(tmp_path / "clock.json", {"time_s": 30, "status": "paused"})
    instance = InfoPanel(tmp_path)
    nodes = [{"id": "vehicle:" + item, "node_type": "vehicle", "vehicle_id": item} for item in "abcd"]
    edges = [{"source": "vehicle:" + a, "target": "vehicle:" + b, "edge_type": "leader", "dependency_id": a + b}
             for a, b in (("a", "b"), ("c", "d"))]
    instance.frame = {"snapshot": {"time_s": 30, "nodes": nodes, "edges": edges, "waiting_states": {}},
                      "groups": [{"id": "G001", "nodes": ["vehicle:a", "vehicle:b"], "categories": ["following"],
                                  "label": "File de 2 voitures", "vehicles": 2, "resources": 0, "edges": 1},
                                 {"id": "G002", "nodes": ["vehicle:c", "vehicle:d"], "categories": ["following"],
                                  "label": "File de 2 voitures", "vehicles": 2, "resources": 0, "edges": 1}],
                      "resources": {}}
    instance.clock = {"time_s": 30, "status": "paused", "present": list("abcd")}
    instance.fill_groups()
    instance.draw()
    instance.root.update_idletasks()
    try:
        yield instance
    finally:
        instance.root.destroy()


def test_two_groups_at_the_same_time_keep_the_graph_and_data(panel):
    before = copy.deepcopy(panel.frame)
    assert len(panel.canvas.find_all()) > 0
    for group, actors in (("G001", {"vehicle:a", "vehicle:b"}), ("G002", {"vehicle:c", "vehicle:d"})):
        panel.groups.selection_set(group)
        panel.choose_group()
        assert panel.selected_nodes() == actors
        assert len(panel.canvas.find_all()) >= 5
        assert panel.frame == before and panel.clock["time_s"] == 30


@pytest.mark.parametrize("kind", ["fit", "sector", "network", "manual", "follow"])
def test_navigation_in_pause_only_sends_native_camera_request(panel, kind):
    panel.focus = "a"
    panel.draw()
    glyphs = panel.canvas.find_all()
    panel.request_camera(kind)
    command = json.loads((panel.directory / "control.json").read_text())
    assert command["camera"]["kind"] == kind and command["paused"]
    assert panel.canvas.find_all() == glyphs and panel.canvas.winfo_viewable()
    assert not hasattr(panel, "physical") and not hasattr(panel, "views")
    assert panel.frame["snapshot"]["time_s"] == 30


def test_three_label_modes_leave_graph_and_aliases_unchanged(panel):
    original = copy.deepcopy(panel.frame)
    for label, mode in (("Sélection", "selected"), ("Toutes", "all"), ("Aucune", "none")):
        panel.label_mode.set(label)
        panel.change_labels()
        assert json.loads((panel.directory / "control.json").read_text())["labels"] == mode
        assert panel.frame == original


def test_new_group_does_not_restore_a_previously_inspected_relation(panel):
    panel.groups.selection_set("G001")
    panel.choose_group()
    panel.inspect(panel.frame["snapshot"]["edges"][0])
    assert panel.chosen_vehicles() == ["a", "b"]
    panel.groups.selection_set("G002")
    panel.choose_group()
    assert panel.inspected is None and panel.chosen_vehicles() == ["c", "d"]
    assert json.loads((panel.directory / "control.json").read_text())["highlighted"] == ["c", "d"]


def test_ordinary_native_id_can_be_looked_up_without_inventing_a_node(panel):
    panel.aliases["ordinary"] = "V005"
    panel.reverse["V005"] = "ordinary"
    panel.choice.set("ordinary")
    panel.choose_vehicle()
    assert panel.focus == "ordinary" and panel.choice.get() == "V005"
    assert panel.selected_nodes() == {row["id"] for row in panel.frame["snapshot"]["nodes"]}
    assert "Aucune dépendance qualifiée" in panel.description.get()


def test_graphical_legends_have_actual_colour_dots(panel):
    def canvases(widget):
        return [widget] if widget.winfo_class() == "Canvas" else [c for child in widget.winfo_children() for c in canvases(child)]
    dots = [c.itemcget(item, "fill") for c in canvases(panel.legend_box) for item in c.find_all() if c.type(item) == "oval"]
    assert dots == ["#1991dc", "#eb9b2d", "#916ec3", "#ffff00"]
    assert not hasattr(panel, "legend")
    assert not any(child.winfo_class() == "Toplevel" for child in panel.root.winfo_children())


def test_small_window_scrolls_without_losing_the_graph_or_controls(panel):
    before = copy.deepcopy(panel.frame)
    panel.root.geometry("520x420")
    panel.root.update()
    assert panel.root.resizable() == (1, 1)
    assert panel.viewport.xview()[1] < 1 and panel.viewport.yview()[1] < 1
    panel.viewport.yview_moveto(1)
    panel.viewport.xview_moveto(1)
    panel.root.update()
    assert panel.legend_box.winfo_viewable() and panel.canvas.find_all()
    assert panel.frame == before and panel.clock["time_s"] == 30


def test_mouse_sizegrip_changes_both_dimensions_without_a_simulation_command(panel):
    before = (panel.root.winfo_width(), panel.root.winfo_height())
    command = (panel.directory / "control.json").read_bytes()
    grip = panel.sizegrip
    x, y = grip.winfo_rootx() + 5, grip.winfo_rooty() + 5
    grip.event_generate("<ButtonPress-1>", x=5, y=5, rootx=x, rooty=y)
    grip.event_generate("<B1-Motion>", x=165, y=105, rootx=x + 160, rooty=y + 100)
    grip.event_generate("<ButtonRelease-1>", x=165, y=105, rootx=x + 160, rooty=y + 100)
    panel.root.update()
    assert panel.root.winfo_width() > before[0] and panel.root.winfo_height() > before[1]
    assert (panel.directory / "control.json").read_bytes() == command


def test_network_overview_places_separate_chains_in_separate_cells():
    import networkx as nx
    from cav_recovery.simulation.crdg_panel import graph_positions
    graph = nx.DiGraph([("a", "b"), ("b", "c"), ("d", "e"), ("e", "f"), ("g", "h")])
    positions = graph_positions(graph)
    assert positions == graph_positions(nx.DiGraph(list(reversed(list(graph.edges)))))
    assert min(((x - xx) ** 2 + (y - yy) ** 2) ** .5
               for node, (x, y) in positions.items() for other, (xx, yy) in positions.items() if node != other) > .2
