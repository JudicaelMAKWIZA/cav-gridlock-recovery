"""Régressions automatiques du rendu ; l'essai clavier Windows reste distinct."""

import json
from types import SimpleNamespace

import pygame
import pytest

from cav_recovery.experiments.presentation import french, save_capture, wrapped
from cav_recovery.experiments.session import DetectionSession
from cav_recovery.experiments.view import ExperimentView
from cav_recovery.metadrive.native import make_env


@pytest.fixture
def view():
    env = make_env("crossing", sensors=False)
    env.reset(seed=1)
    inspector = ExperimentView(window=False)
    try:
        yield inspector
    finally:
        inspector.close()
        env.close()


def test_escape_closes_before_drawing_and_close_is_idempotent(view, monkeypatch):
    view.no_window = False
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, unicode="\x1b")
        ],
    )
    monkeypatch.setattr(view, "_draw", lambda: pytest.fail("dessin après fermeture"))
    assert view.render({}, to_image=False) is None
    assert view.closed and view.user_closed and not pygame.display.get_init()
    view.close()
    assert view.render({}) is None


@pytest.mark.parametrize(
    "key,character,direction",
    [
        (pygame.K_6, "-", -1),
        (pygame.K_EQUALS, "+", 1),
        (pygame.K_KP_PLUS, "", 1),
        (pygame.K_KP_MINUS, "", -1),
    ],
)
def test_azerty_and_numeric_keypad_change_cadence(
    view, monkeypatch, key, character, direction
):
    view.no_window = False
    initial = view.rate_index
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [pygame.event.Event(pygame.KEYDOWN, key=key, unicode=character)],
    )
    view._handle_event()
    assert view.rate_index == initial + direction
    assert "Cadence cible" in view.notice


def test_repeated_space_does_not_toggle_twice_and_layers_have_feedback(
    view, monkeypatch
):
    view.no_window = False
    events = [
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE, repeat=repeat)
        for repeat in (False, True, True)
    ]
    monkeypatch.setattr(pygame.event, "get", lambda: events)
    view._handle_event()
    assert view.paused
    assert not view.show_v2v
    for action, field in (
        ("v2v", "show_v2v"),
        ("graph", "show_graph"),
        ("resources", "show_resources"),
    ):
        before = getattr(view, field)
        view._action(action)
        assert getattr(view, field) != before and view.notice


def test_pause_serves_events_without_redrawing_an_unchanged_scene(view, monkeypatch):
    view.no_window = False
    view.paused = True
    draws = []
    pumps = []

    def pump():
        pumps.append(True)
        if len(pumps) == 12:
            view.paused = False

    monkeypatch.setattr(view, "_handle_event", pump)
    monkeypatch.setattr(
        view, "_redraw", lambda: (draws.append(True), setattr(view, "dirty", False))
    )
    view.clock = SimpleNamespace(tick=lambda _: None)
    frame = dict(at_s=0, diagnostics={}, graphs={}, deliveries=[])
    assert view.inspect(frame)
    assert len(pumps) == 12 and len(draws) == 1


def test_useful_v2v_is_received_relevant_deduplicated_and_bounded(view):
    keys = list(view.engine.agents)
    view.selected = keys[0]
    view.recent_links = {(keys[1], keys[0]): 0, (keys[0], keys[1]): 0}
    graph = dict(edges=[dict(state="CURRENT", blocker=keys[1])])
    assert view._useful_links(graph) == []
    view.v2v_mode = 1
    assert view._useful_links(graph) == [(keys[1], keys[0])]
    view.v2v_mode = 2
    assert len(view._useful_links(graph)) == 2
    view.frame = dict(at_s=0.8, deliveries=[])
    view._remember_deliveries()
    assert view._useful_links(graph) == []


def test_capture_keeps_pixels_provenance_and_previous_capture(view, tmp_path):
    metadata = dict(at_s=10.4, scenario="collective-1", seed=21019, commit="a" * 40)
    surface = view.screen_canvas
    first = save_capture(surface, tmp_path, metadata, 1)
    second = save_capture(surface, tmp_path, metadata, 1)
    assert first != second and first.exists()
    assert json.loads(second.with_suffix(".json").read_text("utf-8")) == metadata
    assert pygame.image.tostring(pygame.image.load(str(first)), "RGB") == (
        pygame.image.tostring(surface, "RGB")
    )


def test_panel_wraps_long_text_and_translates_without_changing_codes(view):
    pygame.font.init()
    font = pygame.font.Font(None, 23)
    code = "PERSISTENT_MOVEMENT_OBSTRUCTION"
    assert french(code) == "Mouvement physiquement obstrué dans la durée"
    assert code == "PERSISTENT_MOVEMENT_OBSTRUCTION"
    lines = list(wrapped(french(code) + " " + "x" * 150, font, 326))
    assert len(lines) > 2 and all(font.size(line)[0] <= 326 for line in lines)


def test_global_display_reference_is_optional_and_keeps_its_past(tmp_path):
    session = DetectionSession(tmp_path)
    session.inspect_reference(dict(time_s=0, witnesses=[]), {}, True)
    assert session.reference_stats is None
    assert len(session.reference_monitor.frames) == 1
    session.view = SimpleNamespace(show_reference=True)
    session.inspect_reference(dict(time_s=0.2, witnesses=[]), {}, True)
    assert len(session.reference_monitor.frames) == 2
    assert (
        session.reference_stats["source"]
        == "independent_native_reference_for_display_only"
    )


def display_frame(view, at_s=1.0):
    """Données de présentation contrôlées ; aucune qualification scientifique simulée."""
    diagnostics = {}
    for key in view.engine.agents:
        diagnostics[key] = dict(
            ego_id=key,
            movement_id="straight",
            current_occupants=[],
            status="NO_ALERT",
            reason="PROGRESSING",
            destination="a1_sink",
            own_state=dict(signed_speed_mps=1.0),
            own_controls=dict(requested_speed_mps=3.0, accepted_speed_mps=1.0),
            coverage=dict(series_start_s=None, effect_start_s=None, proof_times_s=[]),
            events=[],
        )
    keys = list(diagnostics)
    edge = dict(
        state="CURRENT",
        requester=keys[0],
        blocker=keys[1],
        level="SUSPECTED",
        resource_wkt="POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
    )
    diagnostics[keys[0]]["current_occupants"] = [
        dict(
            sender_id=keys[1],
            resource_polygon_wkt=edge["resource_wkt"],
            observed_at_s=at_s,
        )
    ]
    return dict(
        at_s=at_s,
        diagnostics=diagnostics,
        graphs={key: dict(groups=[], edges=[edge]) for key in diagnostics},
        deliveries=[],
        measurements={},
        costs_ms={},
        independent_reference=None,
    )


@pytest.mark.parametrize("size", [(800, 600), (960, 540), (1248, 616)])
def test_all_views_and_buttons_fit_after_resize(view, size):
    view._resize(size)
    frame = display_frame(view)
    for name in ("traffic", "graph", "v2v", "d01", "resources", "history"):
        view._action(name)
        view.inspect(frame)
        screen = view.screen_canvas.get_rect()
        assert screen.contains(view.scene_rect) and screen.contains(view.panel_rect)
        assert all(screen.contains(rect) for rect in view.buttons.values())
        assert view._screen_size == size
        assert view.label_positions  # Le trafic est dessiné dans chaque mode.
        assert view.scene_rect.right <= view.panel_rect.left
        if view.active_view != "traffic":
            assert view.panel_rect.contains(view.graph_rect)
            if view.details_rect.height:
                assert view.panel_rect.contains(view.details_rect)
    view._action("traffic")
    view._action("details")
    assert view.panel_rect.width == 0 and view.scene_rect.width == size[0]


def test_traffic_is_free_of_diagnostic_arrows_and_labels_stay_near_cars(
    view, monkeypatch
):
    frame = display_frame(view)
    view._action("traffic")
    monkeypatch.setattr(
        view, "_arrow", lambda *args: pytest.fail("flèche sur le trafic")
    )
    view.inspect(frame)
    assert view.active_view == "traffic" and view.label_font_size == 17
    assert all(
        view.label_body_rects[key].inflate(40, 40).colliderect(rect)
        for key, rect in view.label_positions.items()
    )


def test_view_shortcuts_toggle_dedicated_content_and_tab_selects(view, monkeypatch):
    view.frame = display_frame(view)
    view.selected = next(iter(view.frame["diagnostics"]))
    view.no_window = False
    for key, name in (
        (pygame.K_g, "graph"),
        (pygame.K_v, "v2v"),
        (pygame.K_d, "d01"),
        (pygame.K_h, "history"),
    ):
        event = pygame.event.Event(pygame.KEYDOWN, key=key, unicode="")
        monkeypatch.setattr(pygame.event, "get", lambda: [event])
        view._handle_event()
        assert view.active_view == name
        view._handle_event()
        assert view.active_view == "traffic"
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)],
    )
    view._handle_event()
    assert view.active_view == "graph" and view.show_resources
    before = view.selected
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB, unicode="\t")],
    )
    view._handle_event()
    assert before != view.selected and view.manual_selection


def test_d01_plot_keeps_only_past_samples_and_does_not_duplicate_paused_frame(view):
    for tick in range(310):
        view.frame = display_frame(view, tick * 0.2)
        view._remember_d01()
    view._remember_d01()
    assert all(
        len(rows) == 301 and rows[-1]["at_s"] == view.frame["at_s"]
        for rows in view.d01_history.values()
    )


def test_graph_keeps_complete_vehicle_resource_vehicle_chains_beside_traffic(view):
    view._resize((1248, 616))
    frame = display_frame(view)
    owner = next(iter(frame["diagnostics"]))
    template = frame["graphs"][owner]["edges"][0]
    frame["graphs"][owner]["edges"] = [
        dict(template, requester=f"agent{index}", blocker=f"agent{index + 1}")
        for index in range(0, 16, 2)
    ]
    view.selected = owner
    view.manual_selection = True
    view._action("graph")
    view.inspect(frame)
    assert view.label_positions and "diagram_prev" in view.buttons
    nodes = view.info_snapshot["nodes"]
    assert {nodes[key]["kind"] for key in view.graph_nodes} == {"vehicle", "resource"}
    for edge in view.visible_info_edges:
        assert nodes[edge["source"]]["kind"] != nodes[edge["target"]]["kind"]
    assert all(view.graph_rect.contains(rect) for rect in view.graph_nodes.values())
    assert view.graph_rect.left >= view.scene_rect.right


def test_resource_click_inspects_proof_without_changing_local_ego(view, monkeypatch):
    view._action("graph")
    view.inspect(display_frame(view))
    resource = next(key for key in view.graph_nodes if key.startswith("resource:"))
    selected = view.selected
    view.no_window = False
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN, button=1, pos=view.graph_nodes[resource].center
            )
        ],
    )
    view._handle_event()
    assert view.inspected_node == resource and view.selected == selected
    assert any("Occupé par" in text for text, _ in view._information_lines())


def test_diagnostic_details_and_help_leave_traffic_visible(view):
    view._action("graph")
    view.inspect(display_frame(view))
    view._action("details")
    assert view.scene_rect.width > 0 and view.graph_rect.width > 0
    assert view.details_rect.height == 0
    view._action("help")
    view.inspect(display_frame(view))
    assert view.label_positions and view.scene_rect.right <= view.panel_rect.left


def test_initial_window_reserves_space_for_windows_frame_and_taskbar():
    from cav_recovery.experiments.layout import available_size

    width, height = available_size()
    desktop_width, desktop_height = pygame.display.get_desktop_sizes()[0]
    assert width <= desktop_width - 32
    assert height <= desktop_height - 64


def emitted_frame(view, at_s=10.0):
    frame = display_frame(view, at_s)
    keys = list(frame["diagnostics"])
    packet = dict(
        sender_id=keys[0],
        message_id="wire-1",
        observed_at_s=at_s,
        sent_at_s=at_s,
        valid_until_s=at_s + 1.2,
        position_xy_m=[1.25, 2.5],
        signed_speed_mps=3.125,
    )
    frame["deliveries"] = [
        dict(
            event="emitted", channel="telemetry", message=packet, delivered_to=keys[1:]
        )
    ]
    view.selected = keys[0]
    view.manual_selection = True
    return frame


def test_v2v_defaults_to_correspondent_list_with_only_outgoing_packet_buttons(view):
    view._action("v2v")
    frame = emitted_frame(view)
    view.inspect(frame)
    assert view.v2v_presentation == "list" and view.v2v_mode == 2
    assert view.message_buttons and view.details_rect.height == 0
    assert all(
        view.graph_rect.contains(view.buttons[key]) for key in view.message_buttons
    )
    assert view.label_positions and view.scene_rect.right == view.panel_rect.left
    assert view.inspected_message is None


def test_message_click_opens_complete_payload_and_retains_it_after_expiry(
    view, monkeypatch
):
    view._action("v2v")
    frame = emitted_frame(view)
    view.inspect(frame)
    key = next(iter(view.message_buttons))
    view.no_window = False
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN, button=1, pos=view.buttons[key].center
            )
        ],
    )
    view._handle_event()
    view.no_window = True
    assert view.inspected_message["payload"] == frame["deliveries"][0]["message"]
    view.inspect(dict(frame, at_s=11.0, deliveries=[]))
    assert not view.recent_messages
    assert "3.125" in "\n".join(text for text, _ in view._information_lines())
    assert view.details_rect == view.panel_rect and view.label_positions
    view._action("raw_packet")
    assert '"signed_speed_mps": 3.125' in "\n".join(
        text for text, _ in view._information_lines()
    )
    view._action("v2v_presentation")
    assert view.inspected_message is None and view.v2v_presentation == "list"


def test_received_packets_are_grouped_without_automatically_opening_payload(
    view, monkeypatch
):
    view._action("v2v")
    frame = emitted_frame(view)
    view.selected = frame["deliveries"][0]["delivered_to"][0]
    view.inspect(frame)
    assert view.info_snapshot["nodes"] and not view.message_buttons
    assert view.inspected_message is None
    view._action("v2v_presentation")
    view.inspect(frame)
    middle = next(key for key in view.graph_nodes if key.startswith("message:"))
    view.no_window = False
    monkeypatch.setattr(
        pygame.event,
        "get",
        lambda: [
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN, button=1, pos=view.graph_nodes[middle].center
            )
        ],
    )
    view._handle_event()
    assert "MESSAGE REÇU" in view._information_lines()[0][0]


def test_v2v_payload_is_copied_without_mutating_or_reconstructing_backend_data(view):
    view._action("v2v")
    frame = emitted_frame(view)
    view.frame = frame
    view._remember_deliveries()
    frame["deliveries"][0]["message"]["signed_speed_mps"] = 999
    assert (
        next(iter(view.recent_messages.values()))["payload"]["signed_speed_mps"]
        == 3.125
    )


def test_automatic_ego_change_closes_the_previously_selected_packet(view):
    view._action("v2v")
    frame = emitted_frame(view)
    view.inspect(frame)
    sender = view.selected
    view._open_message(next(iter(view.message_buttons.values())))
    remaining = {
        key: value for key, value in frame["diagnostics"].items() if key != sender
    }
    view.inspect(dict(frame, diagnostics=remaining, deliveries=[]))
    assert view.selected != sender and view.inspected_message is None
