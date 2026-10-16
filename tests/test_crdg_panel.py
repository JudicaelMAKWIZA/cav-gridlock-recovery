"""Le graphe reste visible pendant les sélections et commandes de caméra."""

import copy
import json
import os
import time

import pytest

from cav_recovery.simulation.crdg_live import atomic_json
from cav_recovery.simulation.crdg_panel import InfoPanel
from cav_recovery.simulation.v2v import V2VExchange


def wait_size(panel, width, height):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        panel.root.update()
        if (panel.root.winfo_width(), panel.root.winfo_height()) == (width, height):
            panel.root.update_idletasks()
            return
        time.sleep(.01)
    pytest.fail(f"Taille demandée {width}x{height}, reçue {panel.root.geometry()}, "
                f"mesurée {panel.root.winfo_width()}x{panel.root.winfo_height()}, affichée={panel.root.winfo_ismapped()}.")


def resize(panel, width, height):
    panel.root.geometry(f"{width}x{height}")
    wait_size(panel, width, height)


@pytest.fixture
def panel(tmp_path):
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("Affichage Tk absent ; interface réelle non vérifiée.")
    atomic_json(tmp_path / "ids.json", {"a": "V001", "b": "V002", "c": "V003", "d": "V004"})
    atomic_json(tmp_path / "control.json", {"focus": None, "highlighted": [], "paused": True,
                                            "labels": "selected", "v2v_enabled": True})
    atomic_json(tmp_path / "clock.json", {"time_s": 30, "status": "paused", "present": list("abcd")})
    instance = InfoPanel(tmp_path)
    # WSLg peut annoncer la géométrie avant que la fenêtre soit réellement affichée.
    instance.root.wait_visibility()
    resize(instance, 780, 760)
    nodes = [{"id": "vehicle:" + item, "node_type": "vehicle", "vehicle_id": item} for item in "abcd"]
    edges = [{"source": "vehicle:" + a, "target": "vehicle:" + b, "edge_type": "leader", "dependency_id": a + b}
             for a, b in (("a", "b"), ("c", "d"))]
    instance.frame = {"snapshot": {"time_s": 30, "nodes": nodes, "edges": edges, "waiting_states": {},
                                   "vehicle_count": 4, "resource_count": 0, "edge_count": 2},
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
        for pending in instance.root.tk.call("after", "info"):
            instance.root.after_cancel(pending)
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


def test_v2v_section_shows_global_counts_local_provenance_and_range(panel):
    exchange = V2VExchange(range_m=20)
    def row(x):
        return {"position": (x, 0), "distance": 0, "speed": 0, "lane": "road_0",
                "route": ("road", "exit"), "route_index": 0}
    exchange.observe(0.5, {"a": row(0), "b": row(10), "c": row(80)})
    exchange.observe(1.0, {"a": row(0), "b": row(10), "c": row(80)})
    atomic_json(panel.directory / "v2v.json", exchange.snapshot())
    panel.toggle_v2v()
    panel.refresh()
    assert "3 CAV actifs" in panel.v2v_status.cget("text")
    panel.v2v_choice.set("V001")
    panel.render_v2v()
    shown = panel.v2v_text.get("1.0", "end")
    assert "V002" in shown and "V003" in shown
    assert "observation propre" in shown and "diagnostic du transport" in shown
    assert "hors de portée" in shown and "son contenu demeure inconnu" not in shown
    assert len(panel.canvas.find_all()) > 0
    panel.select_v2v_vehicle()
    assert json.loads((panel.directory / "control.json").read_text())["focus"] == "a"


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
    assert "Aucun lien observé" in panel.description.get()


def test_graphical_legends_have_actual_colour_dots(panel):
    def canvases(widget):
        return [widget] if widget.winfo_class() == "Canvas" else [c for child in widget.winfo_children() for c in canvases(child)]
    dots = [c.itemcget(item, "fill") for c in canvases(panel.legend_box) for item in c.find_all() if c.type(item) == "oval"]
    assert dots == ["#1991dc", "#eb9b2d", "#916ec3", "#ffff00"]
    assert not hasattr(panel, "legend")
    assert not any(child.winfo_class() == "Toplevel" for child in panel.root.winfo_children())


def test_small_window_scrolls_without_losing_the_graph_or_controls(panel):
    before = copy.deepcopy(panel.frame)
    resize(panel, 520, 420)
    assert panel.root.resizable() == (1, 1)
    assert panel.viewport.xview()[1] == 1 and panel.viewport.yview()[1] < 1
    assert panel.groups.xview()[1] < 1
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
    wait_size(panel, before[0] + 160, before[1] + 100)
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


def detection_row():
    return {"episode_id": "obstruction:000001", "junction": "J", "members": ["a", "b"],
            "first_seen_s": 10, "last_seen_s": 30, "observed_age_s": 20, "cycle_s": 90,
            "state": "persistent_immobility", "physical_resolution": "unknown", "right_censored": False,
            "current_halted_members": ["a", "b"], "persistent_members": ["a"], "remaining_members": ["a", "b"],
            "current_present_members": ["a", "b"], "advanced_members": [], "end_s": None,
            "cause_status": "occupation_only", "progress_m": {"a": 0, "b": 1}, "arrivals_s": {},
            "evidence": {"vehicles": {"a": {"low_progress_age_s": 100}, "b": {"low_progress_age_s": 10}},
                         "qualified_dependencies": [], "native_occupations": [{"ego_id": "a", "vehicle_id": "b"}],
                         "known_movements": [("in", "out")]}}


def enable_detection(panel):
    panel.frame["detection"] = {"time_s": 30, "episodes": [detection_row()], "completed_episodes": [],
                               "street_names": {"in": "Avenue Nguma", "out": "Avenue Colonel Mondjiba"}}
    panel.detection_button.configure(state="normal")
    panel.toggle_detector()
    panel.groups.selection_set("obstruction:000001")
    panel.choose_group()


def test_detector_selection_keeps_the_real_graph_and_existing_camera_protocol(panel):
    enable_detection(panel)
    before = copy.deepcopy(panel.frame)
    assert panel.selected_nodes() == {"vehicle:a", "vehicle:b"}
    assert panel.chosen_vehicles() == ["a", "b"]
    assert panel.canvas.find_all()
    panel.request_camera("fit")
    command = json.loads((panel.directory / "control.json").read_text())
    assert command["paused"] and command["camera"]["vehicles"] == ["a", "b"]
    assert panel.frame == before and panel.clock["time_s"] == 30


def test_detector_explanation_separates_duration_cause_and_unknowns(panel):
    enable_detection(panel)
    assert not panel.detail_box.winfo_manager()
    assert "10–100 s" in panel.description.get() and "gêne non confirmée" in panel.description.get()
    panel.technical.set(True)
    panel.details()
    text = panel.detail_text.get("1.0", "end")
    assert "V001" in text and "V002" in text
    assert "depuis son entrée ou son retour" in text and "Avenue Nguma" in text
    assert "Aucun gridlock confirmé" in text and "Aucune décision" in text


def test_detector_can_select_an_actor_without_inventing_a_crdg_node(panel):
    enable_detection(panel)
    panel.aliases["ordinary"], panel.reverse["V005"] = "V005", "ordinary"
    panel.clock["present"].append("ordinary")
    row = panel.frame["detection"]["episodes"][0]
    row.update(current_halted_members=["ordinary"], members=["ordinary"])
    panel.fill_groups()
    assert panel.chosen_vehicles() == ["ordinary"]
    assert not any(node["id"] == "vehicle:ordinary" for node in panel.frame["snapshot"]["nodes"])
    assert "0 dans le graphe" in panel.selection_notice.get()


def test_detector_resolved_history_remains_inspectable_without_fabricated_presence(panel):
    enable_detection(panel)
    row = copy.deepcopy(detection_row())
    row.update(state="resolved", physical_resolution="verified", current_halted_members=[], end_s=30)
    panel.frame["detection"].update(episodes=[], completed_episodes=[row])
    panel.clock["present"] = []
    panel.fill_groups()
    assert panel.detection_id == row["episode_id"] and panel.chosen_vehicles() == []
    assert "Passage dégagé" in panel.description.get() and "Historique à 30 s" in panel.selection_notice.get()


def test_detector_horizon_is_displayed_as_censored_not_permanent(panel):
    enable_detection(panel)
    panel.frame["detection"]["run_status"] = "horizon_reached"
    panel.fill_groups()
    assert "Fin ·" in panel.groups.item(panel.detection_id, "values")[1]
    assert "Simulation terminée ; issue inconnue" in panel.description.get()


def test_completed_run_waits_for_episode_closure_without_claiming_resolution(panel):
    enable_detection(panel)
    panel.frame["detection"]["run_status"] = "completed"
    panel.fill_groups()
    assert "Fin ·" in panel.groups.item(panel.detection_id, "values")[1]
    assert "à clôturer" in panel.group_count.cget("text")
    assert "dégagement à confirmer après fermeture" in panel.description.get()


def test_returning_to_crdg_restores_its_table_without_changing_observations(panel):
    enable_detection(panel)
    before = copy.deepcopy(panel.frame)
    panel.toggle_detector()
    assert panel.mode == "crdg" and set(panel.groups.get_children()) == {"G001", "G002"}
    assert panel.detection_nodes is None and panel.frame == before


def test_vehicle_inspection_keeps_detector_context_when_the_frame_updates(panel):
    enable_detection(panel)
    panel.choice.set("V001")
    panel.choose_vehicle()
    panel.fill_groups()
    assert panel.focus == "a" and panel.detection_id == "obstruction:000001"
    assert "gêne non confirmée" in panel.description.get()


def test_closed_history_is_not_censored_by_a_later_run_horizon():
    from cav_recovery.simulation.crdg_panel import detection_text
    row = detection_row()
    row.update(state="resolved", physical_resolution="verified", end_s=30)
    assert "issue inconnue" not in detection_text(row, {"a": "V001", "b": "V002"}, {}, terminal_status="horizon_reached")
    assert "Passage dégagé" in detection_text(row, {"a": "V001", "b": "V002"}, {})


def test_twenty_stopped_and_twelve_advanced_can_be_chosen_separately(panel):
    enable_detection(panel)
    ids = [f"car{i:02}" for i in range(35)]
    panel.aliases = {**{item: f"V{901+i:03}" for i, item in enumerate("abcd")},
                     **{item: f"V{i+1:03}" for i, item in enumerate(ids)}}
    panel.reverse = {value: item for item, value in panel.aliases.items()}
    panel.clock.update(time_s=95, present=ids)
    row = panel.frame["detection"]["episodes"][0]
    row.update(state="stopped", members=ids, last_seen_s=95, advanced_members=ids[:12],
               current_halted_members=ids[12:32], current_present_members=ids[6:], remaining_members=ids[6:],
               member_observations={item: {"measurement_since_s": 90, "first_progress_s": 95,
                                          "first_progress_m": 6} for item in ids[:12]})
    before = copy.deepcopy(panel.frame)
    panel.choice.set("V001")
    panel.fill_groups()
    assert set(panel.chosen_vehicles()) == set(ids[12:32]) and panel.choice.get() == ""
    panel.filter.set("Ont avancé")
    panel.change_filter()
    assert set(panel.chosen_vehicles()) == set(ids[:12]) and len(panel.chosen_vehicles()) == 12
    assert "Progrès enregistrés à 95–95 s" in panel.selection_notice.get()
    panel.technical.set(True)
    panel.details()
    text = panel.detail_text.get("1.0", "end")
    assert "V001" in text and "V012" in text and "6.000 m à 95 s" in text
    panel.filter.set("Encore ici")
    panel.change_filter()
    assert set(panel.chosen_vehicles()) == set(ids[6:])
    panel.filter.set("Tous")
    panel.change_filter()
    assert set(panel.chosen_vehicles()) == set(ids) and panel.frame == before


def test_old_progress_is_not_the_current_title_at_900(panel):
    enable_detection(panel)
    panel.aliases.update(a="V043", b="V048", c="V065", d="V072")
    panel.reverse = {value: key for key, value in panel.aliases.items()}
    panel.clock.update(time_s=900)
    row = panel.frame["detection"]["episodes"][0]
    row.update(state="observed_constraint", members=list("abcd"), last_seen_s=900, cycle_s=None,
               advanced_members=["a", "b"], current_halted_members=["c", "d"], current_present_members=["c", "d"],
               persistent_members=[], cause_status="qualified",
               member_observations={"a": {"first_progress_s": 115.5, "first_progress_m": 6.3},
                                    "b": {"first_progress_s": 116.5, "first_progress_m": 7.3}},
               evidence={"vehicles": {"c": {"low_progress_age_s": 765}, "d": {"low_progress_age_s": 763.5}},
                         "qualified_dependencies": [{"source_vehicle_id": "c", "edge_type": "waits_for", "target": "R"}]})
    panel.fill_groups()
    assert "Contrainte locale observée" in panel.description.get() and "763.5–765 s" in panel.description.get()
    assert "établi pour 1 véhicule" in panel.description.get()
    assert "reprise" not in panel.description.get().lower() and panel.chosen_vehicles() == ["c", "d"]
    panel.filter.set("Ont avancé")
    panel.change_filter()
    assert panel.chosen_vehicles() == ["a", "b"]
    assert "115.5–116.5 s" in panel.selection_notice.get() and "actuelle à 900 s" in panel.selection_notice.get()


def test_local_resolution_at_293_does_not_replay_vehicle_positions_at_300(panel):
    enable_detection(panel)
    row = panel.frame["detection"]["episodes"][0]
    row.update(state="resolved", junction="magasin_oua", end_s=293, last_seen_s=293,
               physical_resolution="verified", current_halted_members=[], remaining_members=[],
               current_present_members=[], advanced_members=["a", "b"])
    panel.frame["detection"].update(episodes=[], completed_episodes=[row])
    panel.clock.update(time_s=300)
    panel.fill_groups()
    assert "Magasin OUA — Passage dégagé" in panel.description.get()
    assert "Historique à 293 s · carte actuelle à 300 s" in panel.selection_notice.get()
    assert panel.filter.get() == "Tous" and panel.chosen_vehicles() == ["a", "b"]
    assert "Arrêtés" not in panel.filter["values"] and "Encore ici" not in panel.filter["values"]
    assert "Passé" in panel.groups.item(row["episode_id"], "values")[1]
    panel.request_camera("fit")
    command = json.loads((panel.directory / "control.json").read_text())
    assert command["camera"]["vehicles"] == ["a", "b"] and command["paused"]
    assert panel.clock["time_s"] == 300 and panel.frame["snapshot"]["time_s"] == 30


def test_plain_cause_never_turns_a_foe_presence_into_a_confirmed_blocker():
    from cav_recovery.simulation.crdg_panel import detection_cause
    row = detection_row()
    assert detection_cause(row) == "Passage occupé ; gêne non confirmée"
    row.update(cause_status="qualified", state="observed_constraint")
    row["evidence"]["vehicles"]["a"]["resources"] = [{"id": "R", "resource_type": "receiving_space"}]
    row["evidence"]["qualified_dependencies"] = [{"source_vehicle_id": "a", "edge_type": "waits_for", "target": "R"}]
    assert detection_cause(row) == "Voie suivante directe encombrée"


def test_receiving_constraint_does_not_claim_all_other_lanes_are_blocked():
    from cav_recovery.simulation.crdg_panel import detection_text

    row = detection_row()
    row.update(state="observed_constraint", cause_status="qualified", identified_members=["a"])
    row["evidence"]["vehicles"]["a"]["resources"] = [{"id": "R", "resource_type": "receiving_space"}]
    row["evidence"]["qualified_dependencies"] = [
        {"source_vehicle_id": "a", "edge_type": "waits_for", "target": "R"}
    ]
    text = detection_text(row, {}, {})
    assert "Contrainte locale observée" in text
    assert "Voie suivante directe encombrée" in text
    assert "Évitement par une autre voie non évalué ici" in text
    assert "Blocage local" not in text


def test_red_now_does_not_explain_a_whole_cycle_without_progress():
    from cav_recovery.simulation.crdg_panel import detection_cause
    row = detection_row()
    row.update(signal_wait_members=["a", "b"], cause_status="insufficient")
    assert detection_cause(row) == "Feu rouge maintenant ; longue attente à expliquer"


def test_identified_vehicle_is_selected_and_located_without_claiming_the_whole_cohort(panel):
    enable_detection(panel)
    row = panel.frame["detection"]["episodes"][0]
    row.update(state="observed_constraint", obstruction_verdict="verified", cause_status="qualified",
               identified_members=["a"], current_halted_members=["a", "b"])
    row["evidence"]["qualified_dependencies"] = [{"source_vehicle_id": "a", "edge_type": "connection_obstacle",
                                                  "target": "vehicle:b"}]
    panel.fill_groups()
    panel.groups.selection_set(row["episode_id"])
    panel.detection_id = None
    panel.choose_group()
    assert panel.filter.get() == "Gênés" and panel.chosen_vehicles() == ["a"]
    assert "établi pour 1 véhicule" in panel.description.get()
    command = json.loads((panel.directory / "control.json").read_text())
    assert command["highlighted"] == ["a"]
    assert command["camera"]["kind"] == "fit" and command["camera"]["vehicles"] == ["a"]
    panel.filter.set("Arrêtés")
    panel.change_filter()
    assert panel.chosen_vehicles() == ["a", "b"]


def test_detector_ranks_current_proven_constraint_above_red_wait_and_history(panel):
    from cav_recovery.simulation.crdg_panel import detection_priority
    proven = detection_row()
    proven.update(episode_id="obstruction:000003", obstruction_verdict="verified", state="observed_constraint")
    red = detection_row()
    red.update(episode_id="obstruction:000001", obstruction_verdict="not_confirmed", state="normal_wait")
    past = detection_row()
    past.update(episode_id="obstruction:000002", obstruction_verdict="verified", state="resolved", end_s=20)
    active = {proven["episode_id"], red["episode_id"]}
    ordered = sorted((red, past, proven), key=lambda row: detection_priority(row, active))
    assert [row["episode_id"] for row in ordered] == [proven["episode_id"], red["episode_id"], past["episode_id"]]


def test_missing_measurements_are_available_in_details_not_hidden(panel):
    enable_detection(panel)
    row = panel.frame["detection"]["episodes"][0]
    row.update(continuous_measurement=False, cycle_s=None)
    row["evidence"]["footprint_uncertainty"] = True
    panel.technical.set(True)
    panel.details()
    text = panel.detail_text.get("1.0", "end")
    assert "Mesures manquantes ou invalides" in text and "arrières inconnue" in text
    assert "durée de référence" in text and "ne sont pas toutes vérifiées" in text
    panel.raw_details.set(True)
    panel.details()
    assert '"continuous_measurement": false' in panel.detail_text.get("1.0", "end")


def test_main_detector_text_is_short_and_has_no_developer_vocabulary(panel):
    enable_detection(panel)
    text = panel.description.get()
    assert len(text.splitlines()) <= 4 and len(text) < 300
    assert not any(word in text.lower() for word in ("qualifi", "causal", "cohorte", "censure", "matériell", "descriptive"))
    assert not panel.detail_box.winfo_manager()


def test_small_detector_graph_uses_fewer_nodes_and_keeps_selected_row_visible(panel):
    enable_detection(panel)
    resize(panel, 520, 420)
    assert len([item for item in panel.canvas.find_all() if panel.canvas.type(item) == "text"]) <= 4
    assert panel.groups.bbox(panel.detection_id)
    panel.technical.set(True)
    panel.details()
    panel.root.update()
    assert str(panel.detail_box) in panel.panes.panes() and panel.detail_text.winfo_height() > 60
    assert panel.viewport.yview()[1] < 1
    panel.panes.sashpos(0, 120)
    panel.panes.sashpos(1, 230)
    panel.root.update()
    assert len([item for item in panel.canvas.find_all() if panel.canvas.type(item) == "text"]) <= 2


def test_individual_details_do_not_silently_show_the_entire_group(panel):
    enable_detection(panel)
    panel.technical.set(True)
    panel.choice.set("V001")
    panel.choose_vehicle()
    panel.details()
    text = panel.detail_text.get("1.0", "end")
    actor_text = text.split("Suivi du passage :")[0]
    assert "V001" in actor_text and "V002" not in actor_text


def test_relation_selection_names_its_actors_not_the_stopped_group(panel):
    enable_detection(panel)
    panel.technical.set(True)
    panel.choice.set("V001")
    panel.choose_vehicle()
    panel.inspect(panel.frame["snapshot"]["edges"][1])
    assert panel.chosen_vehicles() == ["c", "d"] and panel.focus is None
    assert panel.choice.get() == ""
    assert "Acteurs du lien : 2" in panel.selection_notice.get()
    assert "V003 → V004" in panel.detail_text.get("1.0", "end")
    assert "pas forcément tous les véhicules" in panel.detail_text.get("1.0", "end")
    panel.choose_group(object())
    assert panel.chosen_vehicles() == ["c", "d"]
    panel.fill_groups()
    assert panel.chosen_vehicles() == ["c", "d"] and "Acteurs du lien" in panel.selection_notice.get()


def test_arrived_history_keeps_its_summary_when_no_camera_target_remains(panel):
    enable_detection(panel)
    panel.clock["present"] = []
    panel.fill_groups()
    before = panel.description.get()
    command = (panel.directory / "control.json").read_bytes()
    panel.request_camera("fit")
    assert "Aucun membre encore présent" in panel.camera_label.cget("text")
    assert panel.description.get() == before and (panel.directory / "control.json").read_bytes() == command


def test_a_new_frame_waits_for_its_matching_physical_clock(panel):
    before = copy.deepcopy(panel.frame)
    future = copy.deepcopy(before)
    future["snapshot"]["time_s"] = 95
    future["snapshot"].update(vehicle_count=4, resource_count=0, edge_count=2)
    atomic_json(panel.directory / "frame.json", future)
    atomic_json(panel.directory / "clock.json", {"time_s": 94.5, "status": "running"})
    panel.refresh()
    assert panel.frame == before and "Mise à jour en cours" in panel.time_label.cget("text")
    atomic_json(panel.directory / "clock.json", {"time_s": 95, "status": "paused"})
    panel.refresh()
    assert panel.frame["snapshot"]["time_s"] == 95 and "âge 0 s" in panel.time_label.cget("text")
