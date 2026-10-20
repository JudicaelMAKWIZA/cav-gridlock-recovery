"""Exécute les scènes MetaDrive sans appeler le pipeline SUMO."""

from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

from ..diagnostics.dependency_graph import snapshot, track_dependencies
from .control import (
    BRAKING_DECELERATION_MPS2,
    MINIMUM_CLEARANCE_M,
    PATH_TOLERANCE_M,
    REACTION_TIME_S,
    DriveCommand,
    apply_command,
    body,
    guard_blockers,
    path_poses,
    stopping_distance,
)
from .cooperation import MetaDriveExchange
from .native import (
    CASES,
    SensorReadings,
    SubstepEvidence,
    configuration_record,
    geometry,
    make_env,
    mission_at_destination,
    missions_for,
)
from .observation import observe, terminal_event
from .relations import build_graph
from .scenes import PHYSICAL_SCENES, PHYSICAL_POLICIES, agents_for


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_line(stream, value):
    stream.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def validate_options(seed, horizon_s, gui_delay_ms):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 10:
        raise ValueError("Seed MetaDrive attendue entre 0 et 9.")
    if (
        not math.isfinite(horizon_s)
        or horizon_s <= 0
        or abs(horizon_s / 0.1 - round(horizon_s / 0.1)) > 1e-7
    ):
        raise ValueError("L'horizon doit être positif et multiple de 0,1 s.")
    if (
        isinstance(gui_delay_ms, bool)
        or not isinstance(gui_delay_ms, int)
        or gui_delay_ms < 0
    ):
        raise ValueError("Délai graphique invalide.")


def commands_for(env, followers, time_s, priority, policy):
    import numpy as np

    ids = sorted(env.agents)
    if priority not in ids:
        priority = None
    if policy == "scheduled_passage" and time_s >= 10 and priority is None:
        for key in ids:
            blockers, covered = guard_blockers(key, env, 6)
            if covered and not blockers:
                priority = key
                break
    actions, desired, commands = {}, {}, {}
    for key, vehicle in sorted(env.agents.items()):
        follower = followers[key]
        follower.move_to_next_road()
        steering = float(
            np.clip(follower.steering_control(follower.routing_target_lane), -1, 1)
        )
        close = any(
            other != key and body(vehicle).distance(body(v)) < 2
            for other, v in env.agents.items()
        )
        requested_speed = 2.0
        reason = "native_route_following"
        if close:
            requested_speed, reason = 0.0, "symmetric_courtesy_not_physical_closure"
            if priority == key and policy == "scheduled_passage":
                requested_speed, reason = (
                    2.0,
                    "declared_central_scheduled_passage_not_autonomous_recovery",
                )
        lookahead = max(0.5, stopping_distance(math.hypot(*vehicle.velocity)))
        blockers, covered = guard_blockers(key, env, lookahead)
        accepted = 0.0 if blockers or not covered else requested_speed
        request = DriveCommand(requested_speed, steering)
        applied = DriveCommand(accepted, steering)
        actions[key] = apply_command(vehicle, applied)
        poses, complete = path_poses(vehicle, 0.5)
        desired[key] = dict(
            requested_forward=True,
            path_poses=poses,
            path_complete=complete,
            native_mission=vehicle.config["destination"],
        )
        commands[key] = dict(
            requested=asdict(request),
            accepted=asdict(applied),
            native_action=actions[key],
            explicit_brake=vehicle.explicit_brake,
            policy_reason=reason,
            guard_blockers=blockers,
            guard_path_complete=covered,
            guard_refused=accepted != requested_speed,
            guard_lookahead_m=lookahead,
            guard_source="central_global_execution_instrumentation_not_v2v",
        )
    return actions, desired, commands, priority


def run_metadrive(
    output,
    *,
    case="crossing",
    seed=1,
    horizon_s=90.0,
    gui=False,
    gui_delay_ms=30,
    policy=None,
    sensors=True,
    v2v=True,
    crdg=True,
):
    """Démonstrations M01 et expériences physiques à contrôleur global explicite."""
    validate_options(seed, horizon_s, gui_delay_ms)
    physical_case = case in PHYSICAL_SCENES
    policy = policy or ("road_following" if physical_case else "scheduled_passage")
    if (case not in CASES and not physical_case) or policy not in (
        PHYSICAL_POLICIES if physical_case else ("courtesy_wait", "scheduled_passage")
    ):
        raise ValueError("Scène ou politique inconnue.")
    scheduled = agents_for(case, CASES)
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Choisir un dossier de sortie absent ou vide.")
    output.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).parents[1]
    inputs = [
        Path(__file__).with_name(name)
        for name in (
            "__init__.py",
            "observation.py",
            "control.py",
            "native.py",
            "cooperation.py",
            "relations.py",
            "run.py",
            "cli.py",
            "view.py",
            "scenes.py",
        )
    ]
    inputs += [
        package / "diagnostics/dependency_graph.py",
        package / "communication/v2v.py",
        package / "cli.py",
    ]
    if physical_case:
        inputs += [
            Path(__file__).with_name(name)
            for name in ("passage.py", "physical.py", "maneuvers.py")
        ]
    code_hashes = {
        path.relative_to(package).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in inputs
    }
    begin = time.perf_counter()
    env = proof = captured = None
    view = None
    status, errors, events, arrived, missions = "horizon_reached", [], [], {}, {}
    readings, history, first_halted, evidence = {}, {}, {}, {}
    time_s, priority, setup_s = 0.0, None, None
    costs = dict(
        physics_s=0.0,
        integrity_s=0.0,
        observation_s=0.0,
        sensors_s=0.0,
        v2v_s=0.0,
        graph_s=0.0,
        render_s=0.0,
        control_s=0.0,
    )
    exchange = (
        MetaDriveExchange(range_m=30, interval_s=1, lifetime_s=2, step_s=0.1)
        if v2v
        else None
    )
    graph_counts = dict(
        observations=0,
        with_dependencies=0,
        max_edges=0,
        structural_cycles=0,
        closed_path_candidates=0,
    )
    config, lanes, last_image, image_stage = None, [], None, None
    closed = False
    labels = {key: f"V{index}" for index, key in enumerate(sorted(scheduled))}
    controller, certificate = None, None
    progress = {
        key: dict(
            travelled_m=0.0, reverse_distance_m=0.0, halted_s=0.0, last_position=None
        )
        for key in scheduled
    }
    with ExitStack() as stack:
        streams = {
            name: stack.enter_context(
                (output / (name + ".jsonl")).open("x", encoding="utf-8")
            )
            for name in (
                "observations",
                "physics",
                "commands",
                "crdg",
                "crdg_events",
                "sensors",
                "v2v",
                "events",
            )
            + (("physical_reference",) if physical_case else ())
        }
        try:
            from metadrive.policy.idm_policy import IDMPolicy

            env = make_env(case, sensors=sensors)
            env.reset(seed=seed)
            missions = missions_for(env)
            native_ids = {key: vehicle.id for key, vehicle in env.agents.items()}
            if set(missions) != set(scheduled):
                raise RuntimeError(
                    "Les identités initiales diffèrent des missions programmées."
                )
            if any(
                m.destination != scheduled[key]["destination"]
                or m.origin_lane != tuple(scheduled[key]["spawn_lane_index"])
                for key, m in missions.items()
            ):
                raise RuntimeError(
                    "Une mission native diffère de la mission programmée avant reset."
                )
            lanes, surface = geometry(env)
            if case == "shared_passage":
                # L’argument géométrique utilise les voies exactes, sans la tolérance de surface M01.
                from shapely.geometry import LineString
                from shapely.ops import unary_union

                surface = unary_union(
                    [
                        LineString(lane["centers"]).buffer(
                            lane["width_m"] / 2, cap_style=2
                        )
                        for lane in lanes
                    ]
                )
            config = configuration_record(env, seed)
            write_json(output / "configuration.json", config)
            write_json(
                output / "missions.json",
                {key: asdict(m) for key, m in missions.items()},
            )
            write_json(output / "vehicle_labels.json", labels)
            write_json(
                output / "geometry.json",
                dict(
                    lanes=lanes,
                    surface_wkt=surface.wkt,
                    scope="native_lane_envelope_not_permission_or_full_manoeuvre_validation",
                ),
            )
            proof = SubstepEvidence(
                env, missions, surface, lambda row: write_line(streams["physics"], row)
            )
            proof.record(0.0)
            captured = SensorReadings(env) if sensors else None
            followers = {
                key: IDMPolicy(vehicle, seed) for key, vehicle in env.agents.items()
            }
            if physical_case:
                from .maneuvers import FeasibilityController
                from .physical import physical_snapshot, passage_certificate

                controller = FeasibilityController(case, policy, surface)
                write_json(output / "scenario.json", PHYSICAL_SCENES[case])
            if gui:
                from .view import TopDownView

                if physical_case:
                    xmin, ymin, xmax, ymax = surface.bounds
                    view = TopDownView(
                        labels=labels,
                        camera_position=((xmin + xmax) / 2, (ymin + ymax) / 2),
                        scaling=min(
                            7, 900 / (xmax - xmin + 10), 650 / (ymax - ymin + 10)
                        ),
                    )
                else:
                    view = TopDownView(labels=labels)
                env.engine.top_down_renderer = view
            readings = observe(env, 0, missions)
            if physical_case:
                write_line(
                    streams["physical_reference"],
                    physical_snapshot(env, readings, surface, missions, 0),
                )
                if case == "shared_passage":
                    certificate = passage_certificate(readings, surface)
                    write_json(output / "forward_only_certificate.json", certificate)
                for key, row in readings.items():
                    progress[key]["last_position"] = row["position_xy_m"]
            write_line(streams["observations"], dict(time_s=0, vehicles=readings))
            setup_s = time.perf_counter() - begin
            for step in range(round(horizon_s / 0.1)):
                start = time.perf_counter()
                actions, desired, commands, priority = (
                    controller.commands(env, followers, time_s)
                    if physical_case
                    else commands_for(env, followers, time_s, priority, policy)
                )
                costs["control_s"] += time.perf_counter() - start
                before = observe(env, time_s, missions, desired)
                write_line(streams["commands"], dict(time_s=time_s, commands=commands))
                proof.commands = commands
                _, _, terminated, truncated, info = env.step(actions)
                time_s = round((step + 1) * 0.1, 8)
                start = time.perf_counter()
                readings = observe(env, time_s, missions)
                if physical_case:
                    for key, vehicle in env.agents.items():
                        mission = missions[key]
                        if (
                            vehicle.id != native_ids[key]
                            or tuple(vehicle.navigation.checkpoints)
                            != mission.checkpoints
                            or tuple(vehicle.navigation.final_lane.index)
                            != mission.final_lane
                            or tuple(vehicle.config["spawn_lane_index"])
                            != mission.origin_lane
                        ):
                            raise RuntimeError(
                                f"Identité ou route initiale modifiée : {key}"
                            )
                    # Les dernières poses physiques sont enregistrées avant le retrait natif des arrivants.
                    for key, row in proof.last_readings.items():
                        metric = progress[key]
                        distance = math.dist(
                            row["position_xy_m"], metric["last_position"]
                        )
                        metric["travelled_m"] += distance
                        if row["signed_speed_mps"] < -0.05:
                            metric["reverse_distance_m"] += distance
                        if math.hypot(*row["velocity_xy_mps"]) < 0.1:
                            metric["halted_s"] += 0.1
                            first_halted.setdefault(key, time_s)
                        else:
                            first_halted.pop(key, None)
                        metric["last_position"] = row["position_xy_m"]
                    for key in set(first_halted) - set(readings):
                        first_halted.pop(key)
                    if step == 0 or (step + 1) % 10 == 0 or not readings:
                        write_line(
                            streams["physical_reference"],
                            physical_snapshot(env, readings, surface, missions, time_s),
                        )
                write_line(
                    streams["observations"], dict(time_s=time_s, vehicles=readings)
                )
                costs["observation_s"] += time.perf_counter() - start
                for key in sorted(before):
                    native = info.get(key, {})
                    bad = any(
                        native.get(flag)
                        for flag in (
                            "crash_vehicle",
                            "crash_object",
                            "crash_sidewalk",
                            "crash_building",
                            "out_of_road",
                        )
                    )
                    ended = (
                        key not in env.agents
                        or native.get("arrive_dest")
                        or terminated.get(key)
                        or truncated.get(key)
                    )
                    if bad or ended:
                        last = proof.last_readings.get(key)
                        verified = mission_at_destination(last, missions[key], lanes)
                        event = terminal_event(
                            key, time_s, native, last, missions[key], verified
                        )
                        events.append(event)
                        write_line(streams["events"], event)
                        if event["event"] == "arrived":
                            arrived[key] = time_s
                        else:
                            errors.append(event)
                if set(readings) | set(arrived) != set(missions):
                    errors.append(
                        dict(event="identity_or_disappearance", time_s=time_s)
                    )
                if proof.contacts or proof.overlaps or proof.outside:
                    errors.append(
                        dict(
                            event="substep_integrity",
                            contacts=proof.contacts,
                            overlaps=proof.overlaps,
                            outside=proof.outside,
                        )
                    )
                if captured:
                    write_line(
                        streams["sensors"],
                        dict(time_s=time_s, local=captured.snapshot(env)),
                    )
                if exchange:
                    start = time.perf_counter()
                    messages = exchange.observe(time_s, readings)
                    write_line(
                        streams["v2v"],
                        dict(
                            time_s=time_s,
                            events=messages,
                            local_knowledge={
                                key: exchange.knowledge(key) for key in readings
                            },
                        ),
                    )
                    costs["v2v_s"] += time.perf_counter() - start
                if crdg:
                    start = time.perf_counter()
                    for key in set(first_halted) - set(readings):
                        first_halted.pop(key)
                    graph_rows = {}
                    for key, row in readings.items():
                        if math.hypot(*row["velocity_xy_mps"]) < 0.1:
                            first_halted.setdefault(key, time_s)
                        else:
                            first_halted.pop(key, None)
                        poses, complete = path_poses(env.agents[key], 0.5)
                        graph_rows[key] = {
                            **row,
                            "desired_motion": dict(
                                requested_forward=True,
                                path_poses=poses,
                                path_complete=complete,
                            ),
                        }
                    graph, evidence = build_graph(
                        time_s, graph_rows, surface, first_halted
                    )
                    for event in track_dependencies(graph, history):
                        write_line(streams["crdg_events"], event)
                    current = snapshot(graph)
                    graph_counts["observations"] += 1
                    graph_counts["with_dependencies"] += bool(current["edge_count"])
                    graph_counts["max_edges"] = max(
                        graph_counts["max_edges"], current["edge_count"]
                    )
                    graph_counts["structural_cycles"] += bool(
                        current["cycle_candidates"]
                    )
                    graph_counts["closed_path_candidates"] += bool(
                        current["closed_cycle_candidates"]
                    )
                    if step == 0 or (step + 1) % 10 == 0 or errors or not readings:
                        write_line(
                            streams["crdg"],
                            {
                                **current,
                                "physical_evidence": evidence,
                                "scope": "assigned_forward_paths_only_no_gridlock_diagnosis",
                            },
                        )
                    costs["graph_s"] += time.perf_counter() - start
                if gui:
                    start = time.perf_counter()
                    last_image = env.render(
                        mode="topdown",
                        text={"Temps (s)": time_s, "Véhicules": len(readings)},
                    )
                    costs["render_s"] += time.perf_counter() - start
                    if step in (0, 99, 199):
                        from PIL import Image
                        import numpy as np

                        Image.fromarray(np.asarray(last_image)).save(
                            output / f"topdown_{time_s:g}s.png"
                        )
                    if physical_case and controller.stage != image_stage:
                        from PIL import Image
                        import numpy as np

                        image_stage = controller.stage
                        Image.fromarray(np.asarray(last_image)).save(
                            output / f"topdown_stage_{image_stage}_{time_s:g}s.png"
                        )
                    if view.user_closed:
                        status = "user_closed"
                        break
                    if gui_delay_ms:
                        time.sleep(gui_delay_ms / 1000)
                if errors:
                    status = "failed"
                    break
                if not readings:
                    status = "completed"
                    break
            if last_image is not None:
                from PIL import Image
                import numpy as np

                Image.fromarray(np.asarray(last_image)).save(output / "topdown.png")
        except KeyboardInterrupt:
            status = "user_closed"
        except Exception as error:
            status = "failed"
            errors.append(
                dict(
                    event="technical_error",
                    type=type(error).__name__,
                    detail=str(error),
                )
            )
        finally:
            if captured:
                costs["sensors_s"] = captured.wall_s
                try:
                    captured.close()
                except Exception as error:
                    status = "failed"
                    errors.append(dict(event="sensor_close_error", detail=str(error)))
            if proof:
                costs.update(physics_s=proof.physics_s, integrity_s=proof.integrity_s)
                try:
                    proof.close()
                except Exception as error:
                    status = "failed"
                    errors.append(dict(event="observer_close_error", detail=str(error)))
            if env is not None:
                try:
                    env.close()
                    closed = True
                except Exception as error:
                    status = "failed"
                    errors.append(dict(event="close_error", detail=str(error)))
        summary = dict(
            schema_version="cgr-metadrive-run-1",
            scenario=case,
            seed=seed,
            policy=policy,
            preparation="native_legal_lane_initial_poses_not_spontaneous_gridlock",
            status=status,
            scheduled=len(scheduled),
            departed=len(missions),
            arrived=arrived,
            active=sorted(readings),
            pending=len(scheduled) - len(missions),
            simulation_time_s=time_s,
            horizon_s=horizon_s,
            physics_step_s=0.02,
            decision_interval_s=0.1,
            collision_contacts=proof.contacts if proof else [],
            outside_surface=proof.outside if proof else [],
            minimum_body_gap_m=proof.minimum_gap_m if proof else None,
            events=events,
            errors=errors,
            minimum_body_gap=proof.minimum_pair if proof else None,
            footprint_overlaps=proof.overlaps if proof else [],
            vehicle_labels=labels,
            guard_safety_model=dict(
                minimum_clearance_m=MINIMUM_CLEARANCE_M,
                braking_deceleration_mps2=BRAKING_DECELERATION_MPS2,
                reaction_time_s=REACTION_TIME_S,
                path_tolerance_m=PATH_TOLERANCE_M,
                scope="sampled_global_forward_guard_other_braking_envelope_not_universal_safety",
            ),
            closed_normally=closed,
            wall_s=time.perf_counter() - begin,
            costs=costs,
            setup_s=setup_s,
            active_state="last_successful_observation",
            substeps=proof.calls if proof else 0,
            sensor_calls=captured.calls if captured else 0,
            v2v=exchange.snapshot() if exchange else None,
            crdg=graph_counts,
            dependency_statistics=history.get("statistics", {}),
            dependency_episodes_open_censored=list(history.get("active", {}).values()),
            physical_episodes_open_censored=[
                dict(
                    vehicle_id=key,
                    last_observed_s=time_s,
                    first_halted_s=first_halted.get(key),
                )
                for key in readings
            ],
            scientific_status="prepared_movement_interaction_absolute_blockage_unknown_no_D01_port_no_gridlock",
        )
        if physical_case:
            summary.update(
                preparation=PHYSICAL_SCENES[case]["formation"],
                progress=progress,
                forward_only_certificate=certificate,
                controller_stage=controller.stage if controller else None,
                scientific_status="physical_feasibility_experiment_no_D01_global_control",
                recovery_admissibility=(
                    "joint_missions_completed_without_integrity_failure"
                    if status == "completed" and not errors
                    else "failed_not_admissible"
                    if errors
                    else "partial_execution_joint_completion_unverified"
                ),
                still_open=bool(readings),
                dynamic_limits="fixed low speed commands; no real vehicle calibration; 0.02s sampled integrity",
            )
            if case in ("shared_passage", "lane_merge"):
                summary["guard_safety_model"] = dict(
                    minimum_sampled_gap_m=0.7,
                    road_margin_m=0.15,
                    wheelbase_reference_m=2.7,
                    braking_deceleration_mps2=BRAKING_DECELERATION_MPS2,
                    reaction_time_s=REACTION_TIME_S,
                    minimum_lookahead_m=0.35,
                    scope="kinematic_sampled_global_sweep_with_neighbor_stopping_reserve_not_universal_safety",
                )
        write_json(output / "summary.json", summary)
    manifest = dict(
        code=code_hashes,
        code_changed_during_run=any(
            hashlib.sha256(path.read_bytes()).hexdigest()
            != code_hashes[path.relative_to(package).as_posix()]
            for path in inputs
        ),
        outputs={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    )
    write_json(output / "manifest.json", manifest)
    return summary
