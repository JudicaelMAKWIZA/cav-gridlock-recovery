"""Accès natifs à MetaDrive : carte, capteurs et contrôles aux sous-pas Bullet."""

import math
import time

import numpy as np
from shapely.geometry import LineString
from shapely.ops import unary_union
from metadrive.envs.marl_envs.marl_intersection import MultiAgentIntersectionEnv
from metadrive.component.sensors.lidar import Lidar
from metadrive.component.sensors.distance_detector import SideDetector, LaneLineDetector

from . import METADRIVE_COMMIT
from .control import (
    BrakeVehicle,
)  # Enregistre le véhicule avant la création de l'environnement.
from .observation import Mission, body, observe
from .scenes import PHYSICAL_SCENES, agents_for


CASES = {
    "crossing": {
        # Corrige l'écart initial de 21 cm, commun aux deux démonstrations de traversée.
        "agent0": dict(
            spawn_lane_index=(">>>", "1X1_0_", 1),
            spawn_longitude=9.0,
            spawn_lateral=0,
            destination="1X1_1_",
        ),
        "agent1": dict(
            spawn_lane_index=("-1X2_0_", "1X0_0_", 0),
            spawn_longitude=17.5,
            spawn_lateral=0,
            destination="1X0_1_",
        ),
    },
    "convergence": {
        "agent0": dict(
            spawn_lane_index=(">>>", "1X1_0_", 0),
            spawn_longitude=29.5,
            spawn_lateral=0,
            destination="1X1_1_",
        ),
        "agent1": dict(
            spawn_lane_index=("-1X2_0_", "1X1_0_", 0),
            spawn_longitude=19,
            spawn_lateral=0,
            destination="1X1_1_",
        ),
    },
}
CASES["crossing_three"] = {
    **CASES["crossing"],
    "agent2": dict(
        spawn_lane_index=(">>>", "1X1_0_", 1),
        spawn_longitude=2.0,
        spawn_lateral=0,
        destination="1X1_1_",
    ),
}


def make_env(case, *, sensors=True):
    if case not in CASES and case not in PHYSICAL_SCENES:
        raise ValueError(f"Scène MetaDrive inconnue : {case}")
    agents = agents_for(case, CASES)
    config = dict(
        num_agents=len(agents),
        num_scenarios=10,
        allow_respawn=False,
        delay_done=0,
        horizon=1000000,
        traffic_density=0,
        log_level=40,
        decision_repeat=5,
        physics_world_step_size=0.02,
        force_seed_spawn_manager=True,
        use_render=False,
        agent_configs=agents,
        vehicle_config=dict(vehicle_model="cgr_independent_brake", enable_reverse=True),
    )
    if sensors:
        config["sensors"] = dict(
            lidar=(Lidar,),
            side_detector=(SideDetector,),
            lane_line_detector=(LaneLineDetector,),
        )
        config["vehicle_config"].update(
            lidar=dict(num_lasers=72, distance=40, num_others=0),
            side_detector=dict(num_lasers=12, distance=20),
            lane_line_detector=dict(num_lasers=12, distance=20),
        )
    else:
        config["vehicle_config"].update(
            lidar=dict(num_lasers=0, num_others=0),
            side_detector=dict(num_lasers=0),
            lane_line_detector=dict(num_lasers=0),
        )
    if case == "lane_merge":
        from metadrive.envs.marl_envs.marl_bottleneck import MultiAgentBottleneckEnv

        config["map_config"] = dict(
            bottle_lane_num=2,
            lane_num=2,
            neck_lane_num=1,
            neck_length=20,
            exit_length=60,
        )
        return MultiAgentBottleneckEnv(config)
    if case == "shared_passage":
        from metadrive.component.road_network import Road
        from .passage import SharedPassageEnv

        config["spawn_roads"] = [Road("west", "east"), Road("east", "west")]
        config["map_config"] = dict(lane_num=1, exit_length=70, lane_width=3.4)
        return SharedPassageEnv(config)
    return MultiAgentIntersectionEnv(config)


def missions_for(env):
    return {
        key: Mission(
            tuple(v.config["spawn_lane_index"]),
            v.config["destination"],
            tuple(v.navigation.checkpoints),
            tuple(v.navigation.final_lane.index),
            float(v.navigation.final_lane.length),
        )
        for key, v in sorted(env.agents.items())
    }


def geometry(env):
    lanes, surfaces = [], []
    for start, links in sorted(env.current_map.road_network.graph.items()):
        for end, values in sorted(links.items()):
            for index, lane in enumerate(values):
                points = [
                    tuple(map(float, lane.position(s, 0)))
                    for s in np.linspace(0, lane.length, max(3, int(lane.length) + 1))
                ]
                lanes.append(
                    dict(
                        index=[start, end, index],
                        length_m=float(lane.length),
                        width_m=float(lane.width),
                        centers=points,
                    )
                )
                surfaces.append(LineString(points).buffer(lane.width / 2, cap_style=2))
    return lanes, unary_union(surfaces).buffer(0.05)


def mission_at_destination(row, mission, lanes):
    if row is None or row["mission"]["destination"] != mission.destination:
        return False
    from shapely.geometry import Point

    point = Point(row["position_xy_m"])
    for lane in lanes:
        if tuple(lane["index"][:2]) != mission.final_lane[:2]:
            continue
        line = LineString(lane["centers"])
        if (
            line.distance(point) <= lane["width_m"] / 2 + 0.05
            and line.project(point) >= lane["length_m"] - 5.1
        ):
            return True
    return False


class SensorReadings:
    """Garde les rayons déjà calculés par MetaDrive, sans exposer ses objets voisins."""

    def __init__(self, env):
        self.originals = []
        self.latest = {}
        self.calls = 0
        self.wall_s = 0.0
        for name in ("lidar", "side_detector", "lane_line_detector"):
            sensor = env.engine.get_sensor(name)
            native = sensor.perceive
            self.originals.append((sensor, native))

            def measured(vehicle, *args, _name=name, _native=native, **kwargs):
                begin = time.perf_counter()
                result = _native(vehicle, *args, **kwargs)
                values = list(map(float, result[0]))
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                    raise RuntimeError("Rayon de capteur invalide.")
                self.latest.setdefault(vehicle.id, {})[_name] = dict(
                    normalized_ranges=values,
                    range_m=40 if _name == "lidar" else 20,
                    source="native_raycast",
                    neighbor_identity="unknown",
                )
                self.calls += 1
                self.wall_s += time.perf_counter() - begin
                return result

            sensor.perceive = measured

    def snapshot(self, env):
        return {key: self.latest.get(v.id, {}) for key, v in sorted(env.agents.items())}

    def close(self):
        for sensor, native in self.originals:
            sensor.perceive = native


class SubstepEvidence:
    """Vérifie chaque sous-pas sans changer l'intégration Bullet."""

    def __init__(self, env, missions, surface, write):
        self.env, self.missions, self.surface, self.write = (
            env,
            missions,
            surface,
            write,
        )
        self.native = env.engine.step_physics_world
        self.calls = 0
        self.physics_s = self.integrity_s = 0.0
        self.last_readings = {}
        self.contacts = []
        self.overlaps = []
        self.outside = []
        self.minimum_gap_m = None
        self.minimum_pair = None
        self.commands = {}
        env.engine.step_physics_world = self.step

    def step(self):
        begin = time.perf_counter()
        self.native()
        self.physics_s += time.perf_counter() - begin
        begin = time.perf_counter()
        self.calls += 1
        self.record(round(self.calls * 0.02, 9))
        self.integrity_s += time.perf_counter() - begin

    def record(self, time_s):
        """Mesure aussi le reset ; les drapeaux véhicule natifs datent du dernier after_step."""
        readings = observe(self.env, time_s, self.missions)
        self.last_readings = readings
        polygons = {key: body(v) for key, v in sorted(self.env.agents.items())}
        contacts, overlaps, outside, pairs = [], [], [], []
        ids = sorted(polygons)
        for i, left in enumerate(ids):
            if not self.surface.covers(polygons[left]):
                outside.append(left)
            for right in ids[i + 1 :]:
                contact = self.env.engine.physics_world.dynamic_world.contactTestPair(
                    self.env.agents[left].chassis.node(),
                    self.env.agents[right].chassis.node(),
                )
                if contact.getNumContacts():
                    contacts.append([left, right])
                gap = polygons[left].distance(polygons[right])
                area = polygons[left].intersection(polygons[right]).area
                pairs.append(
                    dict(
                        pair=[left, right],
                        body_gap_m=gap,
                        overlap_area_m2=area,
                        bullet_contacts=contact.getNumContacts(),
                    )
                )
                if area > 1e-8:
                    overlaps.append([left, right])
                if self.minimum_gap_m is None or gap < self.minimum_gap_m:
                    self.minimum_gap_m = gap
                    self.minimum_pair = dict(
                        time_s=time_s, pair=[left, right], body_gap_m=gap
                    )
        flags = {
            key: {
                flag: bool(getattr(v, flag))
                for flag in (
                    "crash_vehicle",
                    "crash_object",
                    "crash_sidewalk",
                    "crash_building",
                )
            }
            for key, v in sorted(self.env.agents.items())
        }
        row = dict(
            time_s=time_s,
            contacts=contacts,
            overlaps=overlaps,
            pairs=pairs,
            native_collision_flags_last_decision=flags,
            commands=self.commands,
            outside_surface=outside,
            vehicles=readings,
        )
        self.write(row)
        self.contacts.extend(dict(time_s=row["time_s"], pair=pair) for pair in contacts)
        self.overlaps.extend(dict(time_s=row["time_s"], pair=pair) for pair in overlaps)
        self.outside.extend(
            dict(time_s=row["time_s"], vehicle_id=key) for key in outside
        )

    def close(self):
        self.env.engine.step_physics_world = self.native


def configuration_record(env, seed):
    import importlib.metadata
    import json
    from pathlib import Path
    import subprocess
    from urllib.parse import unquote, urlparse
    from urllib.request import url2pathname

    metadata = importlib.metadata.distribution("metadrive-simulator")
    direct = json.loads(metadata.read_text("direct_url.json") or "{}")
    revision = direct.get("vcs_info", {}).get("commit_id")
    if not revision and direct.get("url", "").startswith("file:"):
        source = Path(url2pathname(urlparse(direct["url"]).path))
        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            revision = result.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
        )
        if dirty.returncode or dirty.stdout.strip():
            raise RuntimeError("La source MetaDrive doit être propre et épinglée.")
    if revision != METADRIVE_COMMIT:
        raise RuntimeError(f"Version source MetaDrive non validée : {revision!r}.")

    config = env.config.get_dict()

    # Le JSON décrit les classes par leur nom, sans sérialiser d'objet moteur.
    def convert(value):
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, type):
            return value.__module__ + "." + value.__qualname__
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return repr(value)

    return dict(
        seed=seed,
        metadrive_commit=revision,
        config=convert(config),
        versions={
            name: importlib.metadata.version(name)
            for name in (
                "metadrive-simulator",
                "Panda3D",
                "numpy",
                "gymnasium",
                "shapely",
                "networkx",
            )
        },
    )
