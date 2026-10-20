"""Observations métriques ; aucune permission SUMO n'est fabriquée."""

from dataclasses import asdict, dataclass
import math

from . import OBSERVATION_VERSION


@dataclass(frozen=True)
class Mission:
    origin_lane: tuple
    destination: str
    checkpoints: tuple
    final_lane: tuple
    final_lane_length_m: float


@dataclass(frozen=True)
class VehicleObservation:
    vehicle_id: str
    time_s: float
    position_xy_m: tuple[float, float]
    heading_rad: float
    signed_speed_mps: float
    velocity_xy_mps: tuple[float, float]
    length_m: float
    width_m: float
    body_xy_m: tuple
    mission: Mission
    lane: tuple
    desired_motion: dict
    source: str = "metadrive_global_instrumentation_not_v2v"
    coordinate_frame: str = "metadrive_local_xy_m"
    schema_version: str = OBSERVATION_VERSION

    def to_dict(self):
        return asdict(self)


def signed_speed(vehicle):
    x, y = vehicle.velocity
    return float(
        x * math.cos(vehicle.heading_theta) + y * math.sin(vehicle.heading_theta)
    )


def rectangle(position, heading, length, width, margin=0):
    from shapely.geometry import Polygon

    forward = (math.cos(heading), math.sin(heading))
    side = (-forward[1], forward[0])
    return Polygon(
        [
            (
                position[0]
                + a * forward[0] * (length / 2 + margin)
                + b * side[0] * (width / 2 + margin),
                position[1]
                + a * forward[1] * (length / 2 + margin)
                + b * side[1] * (width / 2 + margin),
            )
            for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))
        ]
    )


def body(vehicle, margin=0):
    return rectangle(
        vehicle.position, vehicle.heading_theta, vehicle.LENGTH, vehicle.WIDTH, margin
    )


def observe(env, time_s, missions, desired=None):
    rows = {}
    for key, vehicle in sorted(env.agents.items()):
        if vehicle.config["destination"] != missions[key].destination:
            raise RuntimeError(f"Destination modifiée : {key}")
        row = VehicleObservation(
            key,
            time_s,
            tuple(map(float, vehicle.position)),
            float(vehicle.heading_theta),
            signed_speed(vehicle),
            tuple(map(float, vehicle.velocity)),
            float(vehicle.LENGTH),
            float(vehicle.WIDTH),
            tuple(tuple(map(float, xy)) for xy in body(vehicle).exterior.coords),
            missions[key],
            tuple(vehicle.lane_index),
            (desired or {}).get(
                key,
                dict(
                    kind="assigned_road_mission_not_local_control_command",
                    destination=missions[key].destination,
                ),
            ),
        )
        if not all(
            math.isfinite(value)
            for value in (
                *row.position_xy_m,
                row.heading_rad,
                row.signed_speed_mps,
                *row.velocity_xy_mps,
                row.length_m,
                row.width_m,
            )
        ):
            raise RuntimeError(f"État physique non fini : {key}")
        rows[key] = row.to_dict()
    return rows


def terminal_event(vehicle_id, time_s, info, last_row, mission, corridor_verified):
    """Une fin native n'est une arrivée que si sa destination est corroborée."""
    collision = any(
        info.get(flag)
        for flag in (
            "crash_vehicle",
            "crash_object",
            "crash_sidewalk",
            "crash_building",
        )
    )
    if collision:
        kind = "collision"
    elif info.get("out_of_road"):
        kind = "out_of_domain"
    elif info.get("arrive_dest") and corridor_verified and last_row is not None:
        kind = "arrived"
    else:
        kind = "interrupted"
    return dict(
        event=kind,
        vehicle_id=vehicle_id,
        time_s=time_s,
        destination=mission.destination,
        native_arrive_dest=bool(info.get("arrive_dest")),
        corridor_verified=corridor_verified,
        last_observation=last_row,
    )
