"""Simulation dense native, conduite locale réactive et référence physique indépendante."""

from contextlib import ExitStack
from dataclasses import asdict
import gzip
import json
import math
from pathlib import Path
import time
from ..platform_support import peak_rss_kib

import numpy as np
import shapely
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
from metadrive.component.block.base_block import BaseBlock
from metadrive.component.lane.point_lane import PointLane
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.map.pg_map import PGMap
from metadrive.component.pgblock.pg_block import PGBlock
from metadrive.component.road_network import Road, NodeRoadNetwork
from metadrive.component.sensors.lidar import Lidar
from metadrive.constants import PGLineType, PGLineColor
from metadrive.component.pg_space import ParameterSpace
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.manager.pg_map_manager import PGMapManager

from ..metadrive.control import DriveCommand, apply_command
from ..metadrive.observation import body, rectangle, signed_speed
from ..metadrive.native import missions_for, configuration_record
from ..metadrive.cooperation import MetaDriveExchange
from .qualify import qualify


class DatasetBlock(BaseBlock):
    ID = "D"
    PARAMETER_SPACE = ParameterSpace({})
    pre_block_socket_index = 0

    @property
    def block_network_type(self):
        return NodeRoadNetwork

    def _sample_topology(self):
        spec = json.loads(self.engine.global_config["dataset_spec_json"])
        # Le centre partagé appartient à la géométrie routière native.
        # Ce n’est ni une route supplémentaire ni un agrandissement purement graphique.
        corners = []
        for road in spec["roads"]:
            if road["kind"] in ("in", "out"):
                p = road["points"][-1 if road["kind"] == "in" else 0]
                a, b = road["points"][0], road["points"][-1]
                h = math.atan2(b[1] - a[1], b[0] - a[0])
                w = road["width_m"] / 2
                corners.extend(
                    [
                        (p[0] + sign * w * math.sin(h), p[1] - sign * w * math.cos(h))
                        for sign in (-1, 1)
                    ]
                )
        centre = Polygon(corners).convex_hull
        centre_added = False
        for road in sorted(
            spec["roads"], key=lambda r: (r["start"], r["end"], r["lane"])
        ):
            points = road["points"]
            lane = (
                StraightLane(
                    points[0],
                    points[-1],
                    width=road["width_m"],
                    line_types=(PGLineType.NONE, PGLineType.NONE),
                )
                if len(points) == 2
                else PointLane(points, width=road["width_m"])
            )
            lane.line_colors = (PGLineColor.GREY, PGLineColor.GREY)
            if len(points) > 2 and not centre_added:
                lane._polygon = np.array(
                    Polygon(lane.polygon).union(centre).exterior.coords
                )
                centre_added = True
            self.block_network.add_lane(road["start"], road["end"], lane)
        self._respawn_roads = [
            Road(r["start"], r["end"])
            for r in spec["roads"]
            if r["kind"] == "in" and r["lane"] == 0
        ]
        return True

    create_in_world = PGBlock.create_in_world
    _construct_lane_line_in_block = PGBlock._construct_lane_line_in_block


class DatasetMap(PGMap):
    def _generate(self):
        block = DatasetBlock(0, self.road_network, random_seed=1)
        block.construct_block(self.engine.worldNP, self.engine.physics_world)
        self.blocks.append(block)


class DatasetMapManager(PGMapManager):
    def reset(self):
        road_map = self.spawn_object(
            DatasetMap,
            map_config=self.engine.global_config["map_config"],
            random_seed=None,
        )
        self.load_map(road_map)
        road_map.spawn_roads = self.engine.global_config["spawn_roads"]


class DatasetEnv(MultiAgentMetaDrive):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update(dict(dataset_spec_json=""), allow_add_new_key=True)
        return config

    def setup_engine(self):
        super().setup_engine()
        self.engine.update_manager("map_manager", DatasetMapManager())


def vehicle_config(agent):
    return dict(
        spawn_lane_index=(
            f"a{agent['origin']}_source",
            f"a{agent['origin']}_entry",
            agent["lane"],
        ),
        spawn_longitude=agent["longitude"],
        spawn_lateral=0,
        destination=f"a{agent['destination']}_sink",
    )


def environment(spec):
    return DatasetEnv(
        dict(
            dataset_spec_json=json.dumps(spec),
            num_agents=1,
            num_scenarios=1,
            start_seed=spec["seed"],
            allow_respawn=False,
            delay_done=0,
            horizon=1000000,
            traffic_density=0,
            log_level=40,
            decision_repeat=10,
            physics_world_step_size=0.02,
            force_seed_spawn_manager=True,
            use_render=False,
            agent_configs={"agent0": vehicle_config(spec["agents"][0])},
            spawn_roads=[
                Road(r["start"], r["end"])
                for r in spec["roads"]
                if r["kind"] == "in" and r["lane"] == 0
            ],
            map_config=dict(
                lane_num=spec["topology"]["lane_num"],
                exit_length=70,
                lane_width=spec["topology"]["lane_width_m"],
            ),
            sensors=dict(lidar=(Lidar,)),
            vehicle_config=dict(
                vehicle_model="cgr_independent_brake",
                enable_reverse=False,
                lidar=dict(num_lasers=16, distance=30, num_others=0),
                side_detector=dict(num_lasers=0),
                lane_line_detector=dict(num_lasers=0),
            ),
        )
    )


def introduce(env, agent):
    """Crée le véhicule à l’emplacement admis, sans réinitialiser ensuite sa pose."""
    manager = env.agent_manager
    # La création native lit config[agent0] ; la mission est fournie
    # avant création afin d’éviter la réaffectation ultérieure du réappariteur standard.
    config = env.config["vehicle_config"].copy()
    config.update(vehicle_config(agent))
    env.engine.global_config["agent_configs"]["agent0"] = config
    original = manager._init_observations["agent0"]
    manager._init_observations["agent0"] = env.get_single_observation()
    try:
        key, vehicle, _ = manager.propose_new_vehicle()
    finally:
        manager._init_observations["agent0"] = original
    if key != agent["id"]:
        raise ValueError("Native introduction identity mismatch")
    env.dones[key] = False
    return vehicle


def dump(stream, value):
    stream.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")


def measured(values):
    """Arrondit le stockage au micromètre ou microradian, sans modifier l’état moteur."""
    return [round(float(value), 6) for value in values]


def forward_clearance_hits(route, first, last, predicted, vehicle_length_m):
    """Préserve les réserves avant et latérales sans retenir inutilement l’arrière.

    Un polygone entièrement derrière le plan arrière d’une empreinte future
    ne peut la recouper. Seule sa réserve arrière est écartée ; toutes les
    intersections de carrosserie et les marges avant ou latérales sont conservées."""
    polygons = np.array(list(predicted.values()), dtype=object)
    hits = shapely.intersects(route.guard_grid[first:last, None], polygons[None, :])
    headings = route.grid_headings[first:last]
    directions = np.column_stack((np.cos(headings), np.sin(headings)))
    origins = np.sum(route.grid_positions[first:last] * directions, axis=1)
    for column in np.flatnonzero(np.any(hits, axis=0)):
        vertices = shapely.get_coordinates(polygons[column])
        frontmost = np.max(vertices @ directions.T, axis=0) - origins
        entirely_behind = frontmost < -vehicle_length_m / 2 - 1e-8
        hits[:, column] &= ~entirely_behind
    return hits


class Route:
    """Référence curviligne construite sur les trois voies natives connectées."""

    def __init__(self, env, agent):
        network = env.current_map.road_network
        i, j, lane = agent["origin"], agent["destination"], agent["lane"]
        self.indices = [
            (f"a{i}_source", f"a{i}_entry", lane),
            (f"a{i}_entry", f"a{j}_exit", lane),
            (f"a{j}_exit", f"a{j}_sink", lane),
        ]
        self.lanes = [network.get_lane(index) for index in self.indices]
        points = []
        self.boundaries = [0.0]
        for native in self.lanes:
            if points and math.dist(points[-1], native.position(0, 0)) > 0.01:
                raise ValueError("Disconnected native route")
            for s in np.linspace(
                0, native.length, max(2, math.ceil(native.length / 0.4) + 1)
            ):
                p = native.position(float(s), 0).tolist()
                if not points or math.dist(points[-1], p) > 1e-7:
                    points.append(p)
            self.boundaries.append(self.boundaries[-1] + native.length)
        self.line = LineString(points)
        self.length = self.line.length
        self.agent = agent
        self.grid = np.arange(0, self.length + 0.49, 0.5)
        self.grid_positions = np.array([self.position(float(s)) for s in self.grid])
        self.grid_headings = np.array([self.heading(float(s)) for s in self.grid])
        self.body_grid = None
        self.guard_grid = None

    def prepare_bodies(self, length, width):
        self.body_grid = np.array(
            [
                rectangle(p, h, length, width)
                for p, h in zip(self.grid_positions, self.grid_headings)
            ],
            dtype=object,
        )
        self.guard_grid = shapely.buffer(self.body_grid, self.agent["clearance_m"])

    def position(self, s):
        p = self.line.interpolate(max(0, min(s, self.length)))
        return p.x, p.y

    def heading(self, s):
        a, b = self.position(s - 0.15), self.position(s + 0.15)
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def poses(self, s, distance, spacing=0.7):
        end = min(self.length, s + distance)
        return [
            (*self.position(float(p)), self.heading(float(p)))
            for p in np.linspace(
                max(0, s), end, max(2, math.ceil(distance / spacing) + 1)
            )
        ]

    def sweep(self, s, distance, length, width):
        return unary_union(
            [
                rectangle(p[:2], p[2], length, width)
                for p in self.reference_poses(s, distance)
            ]
        )

    def reference_poses(self, s, distance):
        first = max(0, int(math.ceil(s * 2)))
        last = min(len(self.grid), int(math.floor((s + distance) * 2)) + 1)
        poses = [
            (*self.position(s), self.heading(s)),
            *[
                (*map(float, p), float(h))
                for p, h in zip(
                    self.grid_positions[first:last], self.grid_headings[first:last]
                )
            ],
            (*self.position(s + distance), self.heading(s + distance)),
        ]
        return [measured(p) for p in poses]

    def resource(self, s):
        segment = (
            "approach"
            if s < self.boundaries[1]
            else "central"
            if s < self.boundaries[2]
            else "exit"
        )
        return (
            "junction:central"
            if segment == "central"
            else f"arm:{self.agent['origin'] if segment == 'approach' else self.agent['destination']}:lane:{self.agent['lane']}:{segment}"
        )


def simulate(spec, folder, contract):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    cpu_start = time.process_time()
    frames, errors, arrived, introduced, metadata, configuration = (
        [],
        [],
        {},
        {},
        {},
        {},
    )
    env = None
    native_step = None
    physics_calls = 0
    minimum_gap = None
    last_t = 0.0
    samples = 0
    previous_velocity = {}
    last_native_states = {}
    routes = {}
    missions = {}
    native_ids = {}
    counts = []
    pending_fine = None
    agent_specs = {a["id"]: a for a in spec["agents"]}
    pending = list(spec["agents"][1:])
    deferred = {}
    max_tracking = 0.0
    with ExitStack() as stack:
        streams = {
            name: stack.enter_context(
                gzip.open(
                    folder / (name + ".jsonl.gz"),
                    "wt",
                    encoding="utf-8",
                    compresslevel=3,
                )
            )
            for name in (
                "physics",
                "observations",
                "reference",
                "commands",
                "native_events",
            )
        }
        try:
            env = environment(spec)
            env.reset(seed=spec["seed"])
            configuration = configuration_record(env, spec["seed"])
            surface = unary_union(
                [
                    Polygon(l.polygon).buffer(0)
                    for a in env.current_map.road_network.graph.values()
                    for ls in a.values()
                    for l in ls
                ]
            )
            surface_tolerance = surface.buffer(1e-5)
            shapely.prepare(surface_tolerance)
            metadata = dict(
                vehicles={},
                physics_columns=["x_m", "y_m", "heading_rad", "vx_mps", "vy_mps"],
                physics_step_s=0.02,
                sensor=dict(type="native_lidar", rays=16, range_m=30),
                feature_scope="ego_measurements_and_received_v2v_only",
                trace_decimal_places=6,
                trace_maximum_scalar_rounding_error=5e-7,
                controller_perception="ideal_local_object_positions_and_velocities_within_30m_no_occlusion",
                surface_wkt=surface.wkt,
                intersection_count=1,
                route_lanes={},
                initial_progress_m={},
            )

            def register(key, t):
                v = env.agents[key]
                routes[key] = Route(env, agent_specs[key])
                m = missions_for(env)[key]
                missions[key] = m
                native_ids[key] = v.id
                introduced[key] = t
                routes[key].prepare_bodies(v.LENGTH, v.WIDTH)
                metadata["vehicles"][key] = dict(
                    length_m=float(v.LENGTH),
                    width_m=float(v.WIDTH),
                    native_id=v.id,
                    mission=asdict(m),
                    introduced_at_s=t,
                    movement=agent_specs[key]["movement_id"],
                )
                metadata["route_lanes"][key] = [
                    dict(
                        index=list(index),
                        length_m=float(l.length),
                        points=[
                            list(map(float, l.position(float(s), 0)))
                            for s in np.linspace(
                                0, l.length, max(2, math.ceil(l.length / 0.4) + 1)
                            )
                        ],
                    )
                    for index, l in zip(routes[key].indices, routes[key].lanes)
                ]
                metadata["initial_progress_m"][key] = float(
                    routes[key].line.project(Point(v.position))
                )
                dump(
                    streams["native_events"],
                    dict(
                        time_s=t,
                        vehicle_id=key,
                        event="introduced",
                        native_id=v.id,
                        scheduled_at_s=agent_specs[key]["departure_s"],
                        occupancy_checked=True,
                        initial_pose=measured(
                            [*v.position, v.heading_theta, *v.velocity]
                        ),
                    ),
                )

            register("agent0", 0.0)
            exchange = MetaDriveExchange(
                range_m=30, interval_s=0.4, lifetime_s=1.2, step_s=0.2
            )

            def record_physics(t):
                nonlocal minimum_gap, pending_fine
                states = {}
                polygons = {}
                for key, v in sorted(env.agents.items()):
                    states[key] = measured([*v.position, v.heading_theta, *v.velocity])
                    polygons[key] = body(v)
                    if not surface_tolerance.covers(polygons[key]):
                        errors.append(
                            dict(kind="outside_surface", time_s=t, vehicle=key)
                        )
                ids = list(states)
                pairs = []
                xy = np.array([states[k][:2] for k in ids])
                for i, left in enumerate(ids):
                    distances = np.sum((xy[i + 1 :] - xy[i]) ** 2, axis=1)
                    for offset in np.flatnonzero(distances < 18**2):
                        right = ids[i + 1 + offset]
                        gap = polygons[left].distance(polygons[right])
                        minimum_gap = (
                            gap if minimum_gap is None else min(minimum_gap, gap)
                        )
                        contacts = (
                            env.engine.physics_world.dynamic_world.contactTestPair(
                                env.agents[left].chassis.node(),
                                env.agents[right].chassis.node(),
                            ).getNumContacts()
                            if gap < 0.1
                            else 0
                        )
                        overlap = (
                            polygons[left].intersection(polygons[right]).area
                            if gap == 0
                            else 0.0
                        )
                        if contacts or overlap > 1e-8:
                            errors.append(
                                dict(
                                    kind="contact_or_overlap",
                                    time_s=t,
                                    pair=[left, right],
                                )
                            )
                        pairs.append(
                            [left, right, round(gap, 6), contacts, round(overlap, 9)]
                        )
                if pending_fine is not None and pending_fine["time_s"] < t:
                    dump(streams["physics"], pending_fine)
                elif pending_fine is not None and pending_fine["time_s"] > t:
                    raise ValueError("Native trace time reversal")
                pending_fine = dict(
                    time_s=t, states=states, pairs=pairs, pair_center_cutoff_m=18.0
                )
                last_native_states.clear()
                last_native_states.update(states)

            native_step = env.engine.step_physics_world

            def physics():
                nonlocal physics_calls
                native_step()
                physics_calls += 1
                record_physics(round(physics_calls * 0.02, 8))

            env.engine.step_physics_world = physics
            record_physics(0.0)
            for step in range(round(spec["horizon_s"] / 0.2) + 1):
                t = round(step * 0.2, 8)
                last_t = t
                if pending and not errors:
                    # L’ordre d’admission est conservé ; une place occupée retarde la création sans déplacement.
                    agent = pending[0]
                    if (
                        t >= agent["departure_s"]
                        and len(env.agents) < spec["active_limit"]
                    ):
                        lane = env.current_map.road_network.get_lane(
                            vehicle_config(agent)["spawn_lane_index"]
                        )
                        p = lane.position(agent["longitude"], 0)
                        heading = lane.heading_theta_at(agent["longitude"])
                        spawn = rectangle(p, heading, 4.515, 1.852).buffer(1.5)
                        occupied = any(
                            spawn.intersects(body(v)) for v in env.agents.values()
                        )
                        if not occupied:
                            introduce(env, agent)
                            pending.pop(0)
                            register(agent["id"], t)
                            record_physics(t)
                        else:
                            deferred[agent["id"]] = deferred.get(agent["id"], 0) + 1
                current, oracle, actions, commands, v2v_rows = {}, {}, {}, {}, {}
                oracle = []
                polygons = {key: body(v) for key, v in env.agents.items()}
                pose_count = dict(
                    time_s=t,
                    active=len(env.agents),
                    central=0,
                    approach=0,
                    exit=0,
                    moving=0,
                    physically_constrained=0,
                )
                for key, v in sorted(env.agents.items()):
                    a = agent_specs[key]
                    route = routes[key]
                    mission = missions[key]
                    if (
                        v.id != native_ids[key]
                        or v.config["destination"] != mission.destination
                        or tuple(v.navigation.checkpoints) != mission.checkpoints
                    ):
                        raise ValueError("Mission or identity changed")
                    s = float(route.line.project(Point(v.position)))
                    tracking = route.line.distance(Point(v.position))
                    max_tracking = max(max_tracking, tracking)
                    if tracking > spec["topology"]["lane_width_m"] * 0.55:
                        errors.append(
                            dict(
                                kind="route_tracking_failure",
                                time_s=t,
                                vehicle=key,
                                error_m=tracking,
                            )
                        )
                    velocity = tuple(map(float, v.velocity))
                    speed = signed_speed(v)
                    prior = previous_velocity.get(key, velocity)
                    acceleration = [(velocity[i] - prior[i]) / 0.2 for i in (0, 1)]
                    previous_velocity[key] = velocity
                    current[key] = dict(
                        progress_m=s - metadata["initial_progress_m"][key],
                        speed_mps=speed,
                        acceleration_xy_mps2=acceleration,
                        movement_id=a["movement_id"],
                        resource_id=route.resource(s),
                    )
                    segment = route.resource(s).split(":")[-1]
                    pose_count[segment] += 1
                    pose_count["moving"] += abs(speed) > 0.5
                    v2v_rows[key] = dict(
                        vehicle_id=key,
                        position_xy_m=measured(v.position),
                        heading_rad=round(float(v.heading_theta), 6),
                        signed_speed_mps=round(speed, 6),
                        velocity_xy_mps=measured(velocity),
                        length_m=float(v.LENGTH),
                        width_m=float(v.WIDTH),
                        mission=dict(
                            destination=mission.destination, movement=a["movement_id"]
                        ),
                    )
                    distance = contract.movement_vehicle_lengths * v.LENGTH
                    requested = route.sweep(s, distance, v.LENGTH, v.WIDTH).union(
                        polygons[key]
                    )
                    complete = s + distance < route.length - 1e-7
                    admissible = complete and surface_tolerance.covers(requested)
                    constrained = False
                    for other, p in polygons.items():
                        if (
                            other == key
                            or math.dist(v.position, env.agents[other].position)
                            > distance + 6
                        ):
                            continue
                        conflict = requested.intersection(p)
                        if conflict.area > contract.minimum_intersection_area_m2:
                            constrained = True
                            oracle.append(
                                dict(
                                    requester=key,
                                    blocker=other,
                                    c1_expected=True,
                                    body_intersection=True,
                                    intersection_area_m2=conflict.area,
                                    road_admissible=admissible,
                                    no_guard_input=True,
                                    reference_safety_margin_m=contract.reference_margin_m,
                                    safety_envelope_area_m2=requested.buffer(
                                        contract.reference_margin_m
                                    )
                                    .intersection(p)
                                    .area,
                                    resource=dict(
                                        id=route.resource(s),
                                        polygon_wkt=shapely.set_precision(
                                            conflict, 1e-6
                                        ).wkt,
                                    ),
                                    mechanism=(
                                        "longitudinal_occupancy"
                                        if math.cos(
                                            v.heading_theta
                                            - env.agents[other].heading_theta
                                        )
                                        > 0.8
                                        else "opposing_occupancy"
                                        if math.cos(
                                            v.heading_theta
                                            - env.agents[other].heading_theta
                                        )
                                        < -0.8
                                        else "transverse_occupancy"
                                        if abs(
                                            math.cos(
                                                v.heading_theta
                                                - env.agents[other].heading_theta
                                            )
                                        )
                                        < 0.4
                                        else "turning_shared_space"
                                    ),
                                    movement=dict(
                                        id=a["movement_id"],
                                        kind=a["movement_kind"],
                                        distance_m=distance,
                                        reference_poses=route.reference_poses(
                                            s, distance
                                        ),
                                    ),
                                    geometric_ablation=dict(
                                        blocker_removed_releases_this_intersection=True,
                                        scope="spatial_ablation_not_dynamic_intervention",
                                    ),
                                    alternatives="not_evaluated",
                                )
                            )
                    pose_count["physically_constrained"] += constrained
                    # Le conducteur n’observe que les véhicules dans sa portée locale de 30 m.
                    nearby = [
                        o
                        for o in polygons
                        if o != key
                        and math.dist(v.position, env.agents[o].position) < 30.0
                    ]
                    predicted = {
                        o: unary_union(
                            [
                                polygons[o],
                                rectangle(
                                    env.agents[o].position
                                    + np.array(env.agents[o].velocity)
                                    * a["reaction_s"],
                                    env.agents[o].heading_theta,
                                    env.agents[o].LENGTH,
                                    env.agents[o].WIDTH,
                                ),
                            ]
                        ).convex_hull
                        for o in nearby
                    }
                    free = 25.0
                    blockers = []
                    first = max(0, int(math.ceil(s * 2)))
                    last = min(len(route.grid), int(math.ceil((s + 25) * 2)) + 1)
                    if nearby:
                        hits = forward_clearance_hits(
                            route, first, last, predicted, v.LENGTH
                        )
                        occupied = np.flatnonzero(np.any(hits, axis=1))
                        if len(occupied):
                            idx = int(occupied[0])
                            free = max(0.0, float(route.grid[first + idx] - s))
                            blockers = [nearby[i] for i in np.flatnonzero(hits[idx])]
                    curvature = (
                        abs(
                            math.atan2(
                                math.sin(route.heading(s + 4) - route.heading(s)),
                                math.cos(route.heading(s + 4) - route.heading(s)),
                            )
                        )
                        / 4
                    )
                    curve_speed = min(
                        a["target_speed_mps"], math.sqrt(1.4 / max(0.001, curvature))
                    )
                    accepted = min(
                        curve_speed,
                        max(
                            0.0,
                            math.sqrt(2.0 * max(0.0, free - 0.5))
                            - max(0.0, speed) * a["reaction_s"],
                        ),
                    )
                    # Le suivi pure pursuit agit sur les commandes, sans affecter pose ou vitesse.
                    lookahead = 1.5 + max(0.0, speed) * 0.35
                    tx, ty = route.position(s + lookahead)
                    alpha = (
                        math.atan2(ty - v.position[1], tx - v.position[0])
                        - v.heading_theta
                    )
                    alpha = math.atan2(math.sin(alpha), math.cos(alpha))
                    wheelbase = float(v.FRONT_WHEELBASE + v.REAR_WHEELBASE)
                    steer_angle = math.atan2(2 * wheelbase * math.sin(alpha), lookahead)
                    steering = float(
                        np.clip(
                            steer_angle / math.radians(v.config["max_steering"]), -1, 1
                        )
                    )
                    actions[key] = apply_command(v, DriveCommand(accepted, steering))
                    commands[key] = dict(
                        requested_speed_mps=a["target_speed_mps"],
                        accepted_speed_mps=accepted,
                        action=actions[key],
                        guard_blockers=blockers,
                        reason="local_predicted_occupancy"
                        if blockers
                        else "free_native_route",
                        observed_neighbors=nearby,
                        sensing_range_m=30.0,
                        route_tracking_error_m=tracking,
                    )
                frame = dict(time_s=t, states=current, witnesses=oracle)
                frames.append(frame)
                counts.append(pose_count)
                dump(streams["reference"], frame)
                dump(streams["commands"], dict(time_s=t, commands=commands))
                if step:
                    exchange.observe(t, v2v_rows)
                for key, v in sorted(env.agents.items()):
                    rays = env.engine.get_sensor("lidar").perceive(
                        v,
                        env.engine.physics_world.dynamic_world,
                        num_lasers=16,
                        distance=30,
                        show=False,
                    )[0]
                    knowledge = exchange.knowledge(key)
                    dump(
                        streams["observations"],
                        dict(
                            time_s=t,
                            vehicle_id=key,
                            position_xy_m=measured(v.position),
                            heading_rad=round(float(v.heading_theta), 6),
                            signed_speed_mps=round(current[key]["speed_mps"], 6),
                            velocity_xy_mps=measured(v.velocity),
                            acceleration_xy_mps2=measured(
                                current[key]["acceleration_xy_mps2"]
                            ),
                            mission_progress_m=round(current[key]["progress_m"], 6),
                            lidar_normalized=measured(rays),
                            received_v2v=knowledge,
                            missing_mask=dict(lidar=False, v2v=not bool(knowledge)),
                        ),
                    )
                samples += 1
                if (
                    errors
                    or (not env.agents and not pending)
                    or step == round(spec["horizon_s"] / 0.2)
                ):
                    break
                before = set(env.agents)
                _, _, _, _, info = env.step(actions)
                for key in before:
                    event = info.get(key, {})
                    flags = {
                        f: bool(event.get(f))
                        for f in (
                            "arrive_dest",
                            "crash_vehicle",
                            "crash_object",
                            "crash_sidewalk",
                            "crash_building",
                            "out_of_road",
                        )
                    }
                    dump(
                        streams["native_events"],
                        dict(
                            time_s=round(t + 0.2, 8),
                            vehicle_id=key,
                            flags=flags,
                            disappeared=key not in env.agents,
                            terminal_pose=last_native_states.get(key)
                            if key not in env.agents
                            else None,
                        ),
                    )
                    if any(flags[f] for f in flags if f != "arrive_dest"):
                        errors.append(
                            dict(
                                kind="native_collision_or_exit",
                                time_s=round(t + 0.2, 8),
                                vehicle=key,
                                flags=flags,
                            )
                        )
                    if key not in env.agents:
                        last = last_native_states.get(key)
                        route = routes[key]
                        corridor = (
                            last is not None
                            and route.line.project(Point(last[:2]))
                            >= route.length - 5.1
                            and route.line.distance(Point(last[:2]))
                            < spec["topology"]["lane_width_m"] / 2
                        )
                        if event.get("arrive_dest") and corridor:
                            arrived[key] = round(t + 0.2, 8)
                        else:
                            errors.append(
                                dict(
                                    kind="unexplained_disappearance",
                                    vehicle=key,
                                    time_s=round(t + 0.2, 8),
                                )
                            )
                if set(env.agents) | set(arrived) != set(introduced):
                    errors.append(dict(kind="identity_integrity"))
        except Exception as error:
            import traceback

            errors.append(
                dict(
                    kind="technical_error",
                    type=type(error).__name__,
                    detail=str(error),
                    traceback=traceback.format_exc(),
                )
            )
        finally:
            if pending_fine is not None:
                dump(streams["physics"], pending_fine)
            if env is not None:
                if native_step is not None:
                    env.engine.step_physics_world = native_step
                env.close()
    integrity = dict(
        errors=errors,
        all_arrived=len(arrived) == len(spec["agents"]),
        complete_measurements=bool(metadata) and not errors,
        arrivals=arrived,
        introduced=introduced,
        not_introduced=[a["id"] for a in pending],
        minimum_body_gap_m=minimum_gap,
        substeps=physics_calls,
        time_s=last_t,
        maximum_route_tracking_error_m=max_tracking,
    )
    traffic = dict(
        total_planned=len(spec["agents"]),
        total_introduced=len(introduced),
        arrived=len(arrived),
        peak_active=max((c["active"] for c in counts), default=0),
        peak_central=max((c["central"] for c in counts), default=0),
        peak_approach=max((c["approach"] for c in counts), default=0),
        spawn_deferrals=deferred,
        timeline=counts,
    )
    result = qualify(frames, metadata, integrity, contract)
    result.update(
        id=spec["id"],
        family=spec["family"],
        parent_id=spec["parent_id"],
        phase=spec["phase"],
        integrity=integrity,
        traffic=traffic,
        wall_s=time.perf_counter() - start,
        samples=samples,
        cpu_s=time.process_time() - cpu_start,
        peak_rss_kib=peak_rss_kib(),
    )
    for name, value in (
        ("spec", spec),
        ("metadata", metadata),
        ("configuration", configuration),
        ("annotation", result),
    ):
        (folder / (name + ".json")).write_text(
            json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
    return result
