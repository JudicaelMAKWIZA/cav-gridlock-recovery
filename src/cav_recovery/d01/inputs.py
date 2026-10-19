"""Strict, immutable, one-ego boundary. This module performs no I/O."""

from dataclasses import dataclass
import math
from types import MappingProxyType
from collections.abc import Mapping

EPS_TIME = 1e-7
FRAME = "metadrive_local_xy_m"


class InputError(ValueError):
    """Malformed, privileged, future or unsupported observation."""


def _keys(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields.split()):
        raise InputError(f"{name}: expected exactly {fields}")


def _number(value, name, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InputError(f"{name}: finite number required")
    if (low is not None and value < low) or (high is not None and value > high):
        raise InputError(f"{name}: outside supported range")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value or len(value) > 256:
        raise InputError(f"{name}: nonempty bounded string required")


def _vector(value, size, name, low=None, high=None):
    if not isinstance(value, list) or len(value) != size:
        raise InputError(f"{name}: expected {size} components")
    for item in value:
        _number(item, name, low, high)


def _times(block, decision, name):
    a = _number(block["observed_at_s"], name)
    b = _number(block["available_at_s"], name)
    if a > b + EPS_TIME or b > decision + EPS_TIME:
        raise InputError(f"{name}: future observation or availability")


def _equal(value, expected, name):
    if type(value) is not type(expected) or value != expected:
        raise InputError(f"{name}: unsupported value")


def freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(freeze(v) for v in value)
    return value


def plain(value):
    """Independent serializable snapshot, including immutable parsed inputs."""
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


@dataclass(frozen=True)
class LocalInput:
    data: Mapping

    @property
    def time_s(self):
        return self.data["decision_at_s"]

    @property
    def ego_id(self):
        return self.data["ego_id"]


def adapt(value):
    """Validate all nested keys before constructing a detached immutable input.

    Invalid input is explicitly rejected, never silently made into a negative.
    Missing LiDAR/V2V is represented by null; missing essential ego/navigation
    fields is an input error. A valid old measurement remains dated and old.
    """
    _keys(value, "schema_version contract_id ego_id decision_at_s ego mission map lidar own_controls v2v", "input")
    _equal(value["schema_version"], "F01-input-1", "schema")
    _equal(value["contract_id"], "K04_APPROVED", "contract")
    _text(value["ego_id"], "ego_id")
    t = _number(value["decision_at_s"], "decision")
    ego = value["ego"]
    _keys(ego, "observed_at_s available_at_s position_xy_m heading_rad signed_speed_mps velocity_xy_mps acceleration_xy_mps2 mission_progress_m length_m width_m", "ego")
    _times(ego, t, "ego")
    for key in ("position_xy_m", "velocity_xy_mps"):
        _vector(ego[key], 2, key)
    if ego["acceleration_xy_mps2"] is not None:
        _vector(ego["acceleration_xy_mps2"], 2, "acceleration")
    for key in ("heading_rad", "signed_speed_mps", "mission_progress_m"):
        _number(ego[key], key)
    for key in ("length_m", "width_m"):
        _number(ego[key], key, 0.01, 30)
    mission = value["mission"]
    _keys(mission, "observed_at_s available_at_s version origin destination own_route_xy_m movement_id demand_forward", "mission")
    _times(mission, t, "mission")
    for key in ("version", "origin", "destination", "movement_id"):
        _text(mission[key], key)
    if type(mission["demand_forward"]) is not bool:
        raise InputError("demand_forward must be boolean")
    route = mission["own_route_xy_m"]
    if not isinstance(route, list) or not 2 <= len(route) <= 10000:
        raise InputError("own route: 2..10000 points required")
    for point in route:
        _vector(point, 2, "own route point")
    road = value["map"]
    _keys(road, "observed_at_s available_at_s version coordinate_frame drivable_polygons_xy_m lane_width_m rule_context", "map")
    _times(road, t, "map")
    _text(road["version"], "map version")
    _equal(road["coordinate_frame"], FRAME, "frame")
    _number(road["lane_width_m"], "lane width", 1, 10)
    if road["rule_context"] not in ("not_modelled", "ordinary_wait", "unknown"):
        raise InputError("rule_context: unsupported rule (no inferred permission)")
    polygons = road["drivable_polygons_xy_m"]
    if not isinstance(polygons, list) or not 1 <= len(polygons) <= 1024:
        raise InputError("map: bounded explicit polygons required")
    for polygon in polygons:
        if not isinstance(polygon, list) or not 3 <= len(polygon) <= 10000:
            raise InputError("invalid map polygon")
        for point in polygon:
            _vector(point, 2, "map point")
    controls = value["own_controls"]
    _keys(controls, "observed_at_s available_at_s requested_speed_mps accepted_speed_mps native_action", "own_controls")
    _times(controls, t, "controls")
    for key in ("requested_speed_mps", "accepted_speed_mps"):
        _number(controls[key], key)
    _vector(controls["native_action"], 2, "native action", -1, 1)
    lidar = value["lidar"]
    if lidar is not None:
        _keys(lidar, "observed_at_s available_at_s source num_lasers range_m height_m phase_offset_rad normalized_ranges missing", "lidar")
        _times(lidar, t, "lidar")
        for key, expected in (("source", "native_lidar"), ("num_lasers", 16), ("range_m", 30.0), ("height_m", 1.2), ("phase_offset_rad", 0.0)):
            if key in ("range_m", "height_m", "phase_offset_rad"):
                _number(lidar[key], key)
                if abs(lidar[key] - expected) > 1e-12:
                    raise InputError(f"lidar: unsupported {key}")
            else:
                _equal(lidar[key], expected, key)
        if not isinstance(lidar["missing"], bool):
            raise InputError("lidar missing must be boolean")
        rays = lidar["normalized_ranges"]
        if not isinstance(rays, list) or len(rays) != 16:
            raise InputError("16 rays required")
        for ray in rays:
            if ray is not None:
                _number(ray, "ray", 0, 1)
    box = value["v2v"]
    if box is not None:
        _keys(box, "available_at_s received expired", "v2v")
        available = _number(box["available_at_s"], "box availability")
        if available > t + EPS_TIME:
            raise InputError("future mailbox")
        if any(not isinstance(box[k], list) or len(box[k]) > 256 for k in ("received", "expired")):
            raise InputError("bounded mailbox lists required")
        senders = set()
        for entry in box["received"]:
            _keys(entry, "received_at_s message", "received entry")
            m = entry["message"]
            _keys(m, "protocol_version message_id sender_id sequence_number observed_at_s sent_at_s valid_until_s state_schema_version coordinate_frame position_xy_m heading_rad signed_speed_mps length_m width_m intended_destination progress_delta_m progress_kind progress_window_s source confidence", "message")
            for key, expected in (("protocol_version", 1), ("state_schema_version", "cgr-metadrive-state-1"), ("coordinate_frame", FRAME), ("progress_kind", "unsigned_distance_not_mission_progress"), ("source", "metadrive_ego_emulation_not_radio"), ("confidence", "simulated_exact_ego")):
                _equal(m[key], expected, key)
            for key in ("message_id", "sender_id", "intended_destination"):
                _text(m[key], key)
            if m["sender_id"] == value["ego_id"] or m["sender_id"] in senders:
                raise InputError("self or duplicate sender")
            senders.add(m["sender_id"])
            if type(m["sequence_number"]) is not int or m["sequence_number"] < 1:
                raise InputError("positive sequence required")
            if m["message_id"] != f'{m["sender_id"]}:{m["sequence_number"]}':
                raise InputError("message ID/sequence mismatch")
            times = [m["observed_at_s"], m["sent_at_s"], entry["received_at_s"], available]
            for x in times:
                _number(x, "message time")
            until = _number(m["valid_until_s"], "valid until")
            if any(a > b + EPS_TIME for a, b in zip(times, times[1:])) or entry["received_at_s"] > until + EPS_TIME:
                raise InputError("invalid message chronology")
            _vector(m["position_xy_m"], 2, "message position")
            for key in ("heading_rad", "signed_speed_mps"):
                _number(m[key], key)
            for key in ("length_m", "width_m"):
                _number(m[key], key, 0.01, 30)
            for key in ("progress_delta_m", "progress_window_s"):
                if m[key] is not None:
                    _number(m[key], key, 0)
            if (m["progress_delta_m"] is None) != (m["progress_window_s"] is None):
                raise InputError("delta and its window must be missing together")
        for expired in box["expired"]:
            _keys(expired, "sender_id message_id last_received_at_s valid_until_s", "expired")
            for k in ("sender_id", "message_id"):
                _text(expired[k], k)
            a = _number(expired["last_received_at_s"], "last receipt")
            b = _number(expired["valid_until_s"], "expiry")
            if a > b + EPS_TIME or b >= t - EPS_TIME:
                raise InputError("expiry not yet observed")
    # Receipt list order carries no causal information in the ideal transport.
    # Canonicalize detached containers, never the caller's objects.
    if box is not None:
        value={**value,"v2v":{**box,
            "received":sorted(box["received"],key=lambda e:e["message"]["sender_id"]),
            "expired":sorted(box["expired"],key=lambda e:e["sender_id"])}}
    return LocalInput(freeze(value))


def message_age(entry, decision_at_s):
    m = entry["message"]
    if decision_at_s > m["valid_until_s"] + EPS_TIME:
        return "expired"
    return "current" if abs(decision_at_s - m["observed_at_s"]) <= EPS_TIME else "historical"
