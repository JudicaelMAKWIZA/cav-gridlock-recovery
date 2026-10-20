"""Contrats neutres et preuves physiques du socle MetaDrive, distincts de SUMO."""

from dataclasses import asdict
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip(
    "metadrive", reason="Installer l'environnement MetaDrive pour ces tests natifs."
)

from cav_recovery.metadrive.control import (
    MINIMUM_CLEARANCE_M,
    DriveCommand,
    apply_command,
    guard_blockers,
    path_poses,
    stopping_distance,
)
from cav_recovery.metadrive.cooperation import MetaDriveExchange
from cav_recovery.metadrive.native import (
    CASES,
    SubstepEvidence,
    geometry,
    make_env,
    missions_for,
)
from cav_recovery.metadrive.observation import Mission, body, observe, terminal_event
from cav_recovery.metadrive.relations import build_graph, movement_occupancy
from cav_recovery.metadrive.run import run_metadrive, validate_options
from cav_recovery.diagnostics.dependency_graph import snapshot, track_dependencies


@pytest.fixture
def native_env():
    env = make_env("crossing", sensors=False)
    try:
        env.reset(seed=1)
        yield env
    finally:
        env.close()


def test_native_map_missions_ids_and_reproducibility():
    records = []
    for _ in range(2):
        env = make_env("crossing", sensors=False)
        try:
            env.reset(seed=1)
            missions = missions_for(env)
            lanes, surface = geometry(env)
            rows = observe(env, 0, missions)
            assert set(rows) == set(CASES["crossing"])
            assert all(
                mission.destination == CASES["crossing"][key]["destination"]
                for key, mission in missions.items()
            )
            assert all(surface.covers(body(vehicle)) for vehicle in env.agents.values())
            records.append((lanes, rows))
        finally:
            env.close()
    assert records[0] == records[1]


def test_native_route_stays_within_requested_distance(native_env):
    for v in native_env.agents.values():
        poses, complete = path_poses(v, 0.5)
        length = sum(math.dist(a[0], b[0]) for a, b in zip(poses, poses[1:]))
        assert complete and length <= 0.5 + 1e-7


def test_native_next_lane_used_when_navigation_lags_at_connection(native_env):
    v = native_env.agents["agent0"]
    lane = v.navigation.current_ref_lanes[1]
    endpoint = lane.position(lane.length, 0)
    target = next(
        candidate
        for candidate in v.navigation.next_ref_lanes
        if math.dist(candidate.position(0, 0), endpoint) < 0.1
    )
    proxy = SimpleNamespace(
        position=target.position(1, 0),
        navigation=SimpleNamespace(
            current_ref_lanes=[lane], next_ref_lanes=v.navigation.next_ref_lanes
        ),
    )
    poses, complete = path_poses(proxy, 0.5)
    assert complete and math.dist(poses[0][0], proxy.position) < 1e-7


def test_mission_changes_are_errors(native_env):
    missions = missions_for(native_env)
    native_env.agents["agent0"].config["destination"] = "changed"
    with pytest.raises(RuntimeError, match="Destination"):
        observe(native_env, 0, missions)


def test_neutral_signed_state_and_no_fake_sumo_fields(native_env):
    missions = missions_for(native_env)
    row = observe(native_env, 0, missions)["agent0"]
    json.dumps(row, allow_nan=False)
    assert row["schema_version"] == "cgr-observation-1"
    assert not {"tls", "signal_state", "foes", "next_edge", "route_index"} & row.keys()


def test_reverse_then_independent_brake_retains_destination(native_env):
    vehicle = native_env.agents["agent0"]
    destination = vehicle.config["destination"]
    for _ in range(20):
        native_env.step(
            {
                key: apply_command(v, DriveCommand(-1 if key == "agent0" else 0))
                for key, v in native_env.agents.items()
            }
        )
    reverse = observe(native_env, 2, missions_for(native_env))["agent0"][
        "signed_speed_mps"
    ]
    assert reverse < -0.5
    for _ in range(25):
        native_env.step(
            {
                key: apply_command(v, DriveCommand(0))
                for key, v in native_env.agents.items()
            }
        )
    row = observe(native_env, 4.5, missions_for(native_env))["agent0"]
    assert abs(row["signed_speed_mps"]) < 0.01
    assert vehicle.config["destination"] == destination


def test_native_collision_is_detected_not_counted_as_arrival(native_env):
    # Fixture dangereuse au reset seulement, distincte des scènes routières admissibles.
    a, b = [native_env.agents[key] for key in sorted(native_env.agents)]
    a.set_position([30, 0])
    b.set_position([38, 0])
    a.set_heading_theta(0)
    b.set_heading_theta(math.pi)
    a.set_velocity([1, 0], 3)
    b.set_velocity([-1, 0], 3)
    missions = missions_for(native_env)
    _, surface = geometry(native_env)
    rows = []
    proof = SubstepEvidence(native_env, missions, surface, rows.append)
    try:
        for _ in range(30):
            _, _, _, _, info = native_env.step(
                {key: [0, 0.3] for key in native_env.agents}
            )
            if proof.contacts:
                break
        assert proof.contacts and any(row["contacts"] for row in rows)
        assert all(info[key]["crash_vehicle"] for key in ("agent0", "agent1"))
        event = terminal_event(
            "agent0",
            proof.calls * 0.02,
            info["agent0"],
            proof.last_readings.get("agent0"),
            missions["agent0"],
            False,
        )
        assert event["event"] == "collision"
    finally:
        proof.close()


def test_command_transition_brakes_before_reversing():
    v = SimpleNamespace(velocity=(1, 0), heading_theta=0)
    assert apply_command(v, DriveCommand(-1)) == [0, -1]
    assert v.explicit_brake and not v.enable_reverse


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_invalid_commands_rejected(value):
    with pytest.raises(ValueError):
        DriveCommand(value)


@pytest.mark.parametrize(
    "seed,horizon,delay", [(10, 1, 0), (1, 0, 0), (1, 0.15, 0), (1, 1, -1)]
)
def test_invalid_run_parameters(seed, horizon, delay):
    with pytest.raises(ValueError):
        validate_options(seed, horizon, delay)


@pytest.mark.parametrize(
    "info,corridor,expected",
    [
        ({"arrive_dest": True}, True, "arrived"),
        ({"arrive_dest": True}, False, "interrupted"),
        ({"arrive_dest": True, "crash_vehicle": True}, True, "collision"),
        ({"out_of_road": True}, False, "out_of_domain"),
        ({}, False, "interrupted"),
    ],
)
def test_termination_is_not_automatically_an_arrival(info, corridor, expected):
    mission = Mission(("a", "b", 0), "c", ("a", "c"), ("b", "c", 0), 50)
    assert (
        terminal_event("ego", 1, info, {"time_s": 1}, mission, corridor)["event"]
        == expected
    )


def test_current_geometry_builds_dependency_without_reference(native_env, monkeypatch):
    # Fixture de contrainte à 21 cm ; les démos routières gardent désormais une marge supérieure.
    vehicle = native_env.agents["agent0"]
    vehicle.set_position(vehicle.lane.position(10.1, 0))
    missions = missions_for(native_env)
    readings = observe(native_env, 0, missions)
    _, surface = geometry(native_env)
    for key, v in native_env.agents.items():
        poses, complete = path_poses(v, 0.5)
        readings[key]["desired_motion"] = dict(
            path_poses=poses, path_complete=complete, requested_forward=True
        )
    monkeypatch.setattr(
        Path, "read_text", lambda *a, **k: pytest.fail("Référence externe interdite")
    )
    graph, evidence = build_graph(0, readings, surface, {key: 0 for key in readings})
    assert evidence["agent0"]["state"] == "blocked"
    assert evidence["agent1"]["state"] == "free_on_tested_path"
    assert set(graph.edges) == {
        ("vehicle:agent0", next(n for n in graph if n.startswith("resource:"))),
        (next(n for n in graph if n.startswith("resource:")), "vehicle:agent1"),
    }
    history = {}
    track_dependencies(graph, history)
    json.dumps(snapshot(graph), allow_nan=False)
    empty, _ = build_graph(0.1, {"agent0": readings["agent0"]}, surface, {"agent0": 0})
    events = track_dependencies(empty, history)
    assert empty.number_of_edges() == 0
    assert sum(event["event"] == "disappeared" for event in events) == 2


def test_same_mission_does_not_merge_distinct_requested_spaces(native_env):
    missions = missions_for(native_env)
    readings = observe(native_env, 0, missions)
    _, surface = geometry(native_env)
    poses, complete = path_poses(native_env.agents["agent0"], 0.5)
    readings["agent0"]["desired_motion"] = dict(
        path_poses=poses, path_complete=complete, requested_forward=True
    )
    readings["agent2"] = {**readings["agent0"], "vehicle_id": "agent2"}
    readings["agent1"]["desired_motion"] = {}
    graph, _ = build_graph(0, readings, surface, {"agent0": 0, "agent2": 0})
    resources = [
        node for node, data in graph.nodes(data=True) if data["node_type"] == "resource"
    ]
    assert len(resources) == 2
    assert {graph.nodes[node]["requested_by"] for node in resources} == {
        "agent0",
        "agent2",
    }


def test_unknown_path_is_never_free(native_env):
    row = observe(native_env, 0, missions_for(native_env))["agent0"]
    row["desired_motion"] = {}
    _, surface = geometry(native_env)
    evidence = movement_occupancy(row, {}, surface)
    assert evidence["state"] == "unknown" and evidence["absolute_blockage"] == "unknown"


def test_moving_vehicle_is_not_a_stopped_dependency(native_env):
    readings = observe(native_env, 0, missions_for(native_env))
    _, surface = geometry(native_env)
    for key, row in readings.items():
        poses, complete = path_poses(native_env.agents[key], 0.5)
        row["desired_motion"] = dict(
            path_poses=poses, path_complete=complete, requested_forward=True
        )
        row["signed_speed_mps"] = 1.0
        row["velocity_xy_mps"] = [1.0, 0.0]
    graph, _ = build_graph(0, readings, surface, {})
    assert graph.number_of_edges() == 0


def test_lateral_motion_is_not_mistaken_for_standstill(native_env):
    readings = observe(native_env, 0, missions_for(native_env))
    _, surface = geometry(native_env)
    for key, row in readings.items():
        poses, complete = path_poses(native_env.agents[key], 0.5)
        row["desired_motion"] = dict(
            path_poses=poses, path_complete=complete, requested_forward=True
        )
        row["signed_speed_mps"] = 0.0
        row["velocity_xy_mps"] = [0.0, 1.0]
    graph, _ = build_graph(0, readings, surface, {})
    assert graph.number_of_edges() == 0


def own(x, speed=-1):
    return dict(
        position_xy_m=[x, 0],
        heading_rad=0,
        signed_speed_mps=speed,
        length_m=4.5,
        width_m=1.8,
        mission={"destination": "end"},
    )


def test_v2v_signed_speed_range_expiry_and_no_global_fields():
    exchange = MetaDriveExchange(range_m=30, interval_s=1, lifetime_s=2, step_s=0.1)
    for step in range(1, 11):
        exchange.observe(step / 10, {"a": own(0), "b": own(5)})
    knowledge = exchange.knowledge("a")
    message = knowledge["received"]["b"]["message"]
    assert message["signed_speed_mps"] == -1
    assert message["state_schema_version"] == "cgr-metadrive-state-1"
    assert (
        not {"participants", "reachable_neighbors", "transport_distances_m"}
        & knowledge.keys()
    )
    assert not exchange._deliver("a", message, 1)
    for step in range(11, 32):
        exchange.observe(step / 10, {"a": own(0), "b": own(100)})
    assert not exchange.knowledge("a")["received"]
    assert "b" in exchange.knowledge("a")["expired"]


def test_real_run_repeats_and_optional_observers_do_not_change_physics(tmp_path):
    results = []
    for name, enabled in (("all", True), ("none", False), ("repeat", True)):
        directory = tmp_path / name
        summary = run_metadrive(
            directory, horizon_s=2, sensors=enabled, v2v=enabled, crdg=enabled
        )
        assert summary["status"] == "horizon_reached" and summary["closed_normally"]
        assert len(summary["physical_episodes_open_censored"]) == 2
        assert not summary["errors"] and not summary["collision_contacts"]
        results.append((directory / "observations.jsonl").read_bytes())
    assert results[0] == results[1] == results[2]


def test_native_arrivals_and_initial_destinations(tmp_path):
    summary = run_metadrive(tmp_path / "arrival", case="crossing", horizon_s=90)
    assert summary["status"] == "completed" and len(summary["arrived"]) == 2
    assert summary["active"] == [] and summary["pending"] == 0
    assert all(event["corridor_verified"] for event in summary["events"])
    assert not summary["errors"] and summary["closed_normally"]


def test_secondary_cleanup_error_still_closes_engine(tmp_path, monkeypatch):
    from cav_recovery.metadrive.native import SensorReadings

    native = SensorReadings.close

    def broken_close(self):
        native(self)
        raise RuntimeError("fixture de fermeture")

    monkeypatch.setattr(SensorReadings, "close", broken_close)
    result = run_metadrive(tmp_path / "close_error", horizon_s=0.1)
    assert result["status"] == "failed" and result["closed_normally"]
    assert result["errors"][0]["event"] == "sensor_close_error"


def test_offscreen_topdown_is_native_2d_and_does_not_move_time(native_env):
    from cav_recovery.metadrive.view import TopDownView

    positions = {key: tuple(v.position) for key, v in native_env.agents.items()}
    native_env.engine.top_down_renderer = TopDownView(window=False)
    image = native_env.render(mode="topdown")
    assert image.shape == (760, 1000, 3) and image.min() < image.max()
    assert positions == {key: tuple(v.position) for key, v in native_env.agents.items()}


def test_user_quit_event_does_not_exit_the_process(native_env):
    import pygame
    from cav_recovery.metadrive.view import TopDownView

    # La fermeture est testée sans créer une fenêtre réelle pendant les tests unitaires.
    view = TopDownView(window=False)
    native_env.engine.top_down_renderer = view
    pygame.init()
    view.no_window = False
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    view._handle_event()
    assert view.user_closed


@pytest.mark.parametrize("speed", [0, 0.5, 1, 2, 2.2])
def test_braking_stays_within_guard_reserve(native_env, speed):
    vehicle = native_env.agents["agent0"]
    vehicle.set_position([30, 0])
    vehicle.set_heading_theta(0)
    vehicle.set_velocity([speed, 0])
    start = float(vehicle.position[0])
    maximum = 0.0
    for _ in range(35):
        native_env.step(
            {
                key: apply_command(v, DriveCommand(0))
                for key, v in native_env.agents.items()
            }
        )
        maximum = max(maximum, float(vehicle.position[0]) - start)
    # 10 cm couvrent mise en appui et discrétisation ; marge distincte de celle entre carrosseries.
    assert maximum <= stopping_distance(speed) + 0.1
    assert (
        abs(
            observe(native_env, 3.5, missions_for(native_env))["agent0"][
                "signed_speed_mps"
            ]
        )
        < 0.01
    )


def test_guard_reserves_crossing_vehicle_motion(native_env):
    blockers, complete = guard_blockers("agent1", native_env, 6)
    assert complete and not blockers
    native_env.agents["agent0"].set_velocity([2, 0])
    blockers, complete = guard_blockers("agent1", native_env, 6)
    assert complete and blockers == ["agent0"]


def test_substep_proof_records_initial_polygon_penetration(native_env):
    a, b = native_env.agents.values()
    a.set_position([30, 0])
    b.set_position([30.5, 0])
    missions = missions_for(native_env)
    _, surface = geometry(native_env)
    rows = []
    proof = SubstepEvidence(native_env, missions, surface, rows.append)
    try:
        proof.record(0)
        assert proof.overlaps and proof.minimum_gap_m == 0
        assert rows[0]["pairs"][0]["overlap_area_m2"] > 0
        assert proof.calls == 0
    finally:
        proof.close()


@pytest.mark.parametrize("heading", [0, math.pi / 2, -math.pi / 2, 0.37])
def test_vehicle_drawing_projects_the_full_physical_footprint(native_env, heading):
    from cav_recovery.metadrive.view import TopDownView

    vehicle = native_env.agents["agent0"]
    vehicle.set_heading_theta(heading)
    view = TopDownView(window=False)
    native_env.engine.top_down_renderer = view
    points = view.vehicle_pixels(vehicle)
    canvas = view._frame_canvas
    for pixel, (x, y) in zip(points, body(vehicle).exterior.coords):
        expected = (
            (x - canvas.origin[0]) * canvas.scaling,
            canvas.get_height() - (y - canvas.origin[1]) * canvas.scaling,
        )
        assert max(abs(a - b) for a, b in zip(pixel, expected)) <= 1


def test_crossing_three_clearance_and_rendering_do_not_change_physics(
    tmp_path, monkeypatch
):
    import cav_recovery.metadrive.view as view_module

    native_view = view_module.TopDownView
    monkeypatch.setattr(
        view_module, "TopDownView", lambda **kwargs: native_view(window=False, **kwargs)
    )
    outputs = []
    for name, gui in (("headless", False), ("rendered", True)):
        directory = tmp_path / name
        summary = run_metadrive(
            directory, case="crossing_three", horizon_s=100, gui=gui, gui_delay_ms=0
        )
        assert summary["status"] == "completed" and len(summary["arrived"]) == 3
        assert summary["minimum_body_gap_m"] >= MINIMUM_CLEARANCE_M
        assert (
            not summary["collision_contacts"]
            and not summary["footprint_overlaps"]
            and not summary["errors"]
        )
        assert summary["vehicle_labels"] == {
            "agent0": "V0",
            "agent1": "V1",
            "agent2": "V2",
        }
        assert (
            json.loads((directory / "vehicle_labels.json").read_text())
            == summary["vehicle_labels"]
        )
        outputs.append(
            tuple(
                (directory / name).read_bytes()
                for name in ("observations.jsonl", "physics.jsonl", "commands.jsonl")
            )
        )
    assert outputs[0] == outputs[1]


def test_convergence_still_completes_with_clearance_guard(tmp_path):
    summary = run_metadrive(tmp_path / "convergence", case="convergence", horizon_s=90)
    assert summary["status"] == "completed" and len(summary["arrived"]) == 2
    assert summary["minimum_body_gap_m"] >= MINIMUM_CLEARANCE_M
    assert not summary["errors"] and not summary["footprint_overlaps"]
