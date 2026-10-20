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
