"""Génère un carrefour connecté avec voies réalistes et arrivées échelonnées."""

import math
import random
from .contract import digest

FAMILIES = ("four_arm", "t_junction", "oblique", "asymmetric")
DENSITIES = (8, 16, 24, 30)


def candidate(index, *, phase="production", variants=4):
    if index < 0 or phase not in ("production", "calibration") or variants < 1:
        raise ValueError("Invalid candidate identity.")
    parent, variant = divmod(index, variants)
    family = FAMILIES[parent % len(FAMILIES)]
    rng = random.Random(f"dense-v3:{phase}:{parent}")
    jitter = random.Random(f"dense-v3:{phase}:{parent}:{variant}")
    width = rng.choice((3.3, 3.5, 3.7))
    lane_num = rng.choice((2, 3))
    layout = (
        "inner_left_middle_through_outer_right"
        if lane_num == 3 and rng.random() < 0.5
        else "inner_left_outer_right_shared_through"
    )
    radius = rng.uniform(20.0, 24.0) + (lane_num - 2) * 4
    angles = dict(
        four_arm=[0, 90, 180, 270],
        t_junction=[0, 90, 180],
        oblique=[0, 70, 180, 250],
        asymmetric=[0, 85, 190, 270],
    )[family]
    rotation = rng.uniform(-math.pi, math.pi)
    angles = [math.radians(a) + rotation for a in angles]
    cap = DENSITIES[(parent // len(FAMILIES)) % len(DENSITIES)]
    headway = jitter.uniform(0.5, 1.3) if cap >= 24 else jitter.uniform(0.7, 2.0)
    right_share = jitter.uniform(0.15, 0.95)
    approach_length = rng.uniform(42.0, 55.0)
    roads, endpoints = [], {}
    for arm, angle in enumerate(angles):
        u = (math.cos(angle), math.sin(angle))
        n = (-u[1], u[0])

        def at(distance, offset):
            return [round(distance * u[k] + offset * n[k], 7) for k in (0, 1)]

        for lane in range(lane_num):
            offset = (lane + 0.5) * width
            incoming = [at(radius + approach_length, offset), at(radius, offset)]
            outgoing = [at(radius, -offset), at(radius + approach_length, -offset)]
            endpoints[arm, lane] = (incoming, outgoing)
            for direction, points in (("in", incoming), ("out", outgoing)):
                nodes = (
                    (f"a{arm}_source", f"a{arm}_entry")
                    if direction == "in"
                    else (f"a{arm}_exit", f"a{arm}_sink")
                )
                roads.append(
                    dict(
                        start=nodes[0],
                        end=nodes[1],
                        lane=lane,
                        points=points,
                        width_m=width,
                        kind=direction,
                        arm=arm,
                    )
                )
    movements = {}
    for origin in range(len(angles)):
        for destination in range(len(angles)):
            if destination == origin:
                continue
            delta = (angles[destination] - angles[origin]) % (2 * math.pi)
            kind = (
                "straight"
                if abs(delta - math.pi) < 0.45
                else "left"
                if delta > math.pi
                else "right"
            )
            for lane in range(lane_num):
                incoming = endpoints[origin, lane][0]
                outgoing = endpoints[destination, lane][1]
                p0, p3 = incoming[-1], outgoing[0]
                incoming_u = (-math.cos(angles[origin]), -math.sin(angles[origin]))
                outgoing_u = (
                    math.cos(angles[destination]),
                    math.sin(angles[destination]),
                )
                turn = abs(
                    math.atan2(
                        math.sin(angles[destination] - angles[origin] - math.pi),
                        math.cos(angles[destination] - angles[origin] - math.pi),
                    )
                )
                if turn < 0.01:
                    first = last = math.dist(p0, p3) / 3
                else:
                    det = incoming_u[0] * outgoing_u[1] - incoming_u[1] * outgoing_u[0]
                    dx, dy = p3[0] - p0[0], p3[1] - p0[1]
                    first = (dx * outgoing_u[1] - dy * outgoing_u[0]) / det
                    last = (incoming_u[0] * dy - incoming_u[1] * dx) / det
                    ratio = (4 / 3) * math.tan(turn / 4) / math.tan(turn / 2)
                    first *= ratio
                    last *= ratio
                    if min(first, last) <= 0:
                        raise ValueError("Infeasible junction connector tangents")
                p1 = [p0[k] + first * incoming_u[k] for k in (0, 1)]
                p2 = [p3[k] - last * outgoing_u[k] for k in (0, 1)]
                points = [
                    [
                        round(
                            (1 - t) ** 3 * p0[k]
                            + 3 * (1 - t) ** 2 * t * p1[k]
                            + 3 * (1 - t) * t * t * p2[k]
                            + t**3 * p3[k],
                            7,
                        )
                        for k in (0, 1)
                    ]
                    for t in (i / 80 for i in range(81))
                ]
                movement = f"a{origin}_a{destination}_l{lane}"
                roads.append(
                    dict(
                        start=f"a{origin}_entry",
                        end=f"a{destination}_exit",
                        lane=lane,
                        points=points,
                        width_m=width,
                        kind=kind,
                        movement_id=movement,
                    )
                )
                movements[movement] = dict(
                    origin=origin, destination=destination, lane=lane, kind=kind
                )
    agents = []
    total = cap + max(8, cap // 2)
    for i in range(total):
        origin = i % len(angles)
        choices = [a for a in range(len(angles)) if a != origin]
        right = min(choices, key=lambda a: (angles[a] - angles[origin]) % (2 * math.pi))
        destination = right if jitter.random() < right_share else jitter.choice(choices)
        kind = movements[f"a{origin}_a{destination}_l0"]["kind"]
        lane = (
            0
            if kind == "left"
            else lane_num - 1
            if kind == "right"
            else 1
            if layout == "inner_left_middle_through_outer_right"
            else jitter.randrange(lane_num)
        )
        movement = f"a{origin}_a{destination}_l{lane}"
        agents.append(
            dict(
                id=f"agent{i}",
                origin=origin,
                destination=destination,
                lane=lane,
                movement_id=movement,
                movement_kind=movements[movement]["kind"],
                longitude=4.0,
                departure_s=round(i * headway + jitter.uniform(0, 0.3), 3)
                if i
                else 0.0,
                target_speed_mps=round(jitter.uniform(3.2, 5.0), 4),
                reaction_s=round(jitter.uniform(0.25, 0.55), 4),
                clearance_m=round(jitter.uniform(0.55, 0.9), 4),
                vehicle_model="cgr_independent_brake",
            )
        )
    lineage = digest(
        [family, width, lane_num, round(radius / 3), round(approach_length / 8)]
    )[:20]
    spec = dict(
        schema="cgr-dense-junction-4",
        nominal_controller="local_forward_clearance_v4",
        phase=phase,
        parent_id=f"{phase}:{parent:07d}",
        lineage_group=lineage,
        family=family,
        seed=1000 + index,
        roads=roads,
        agents=agents,
        intersection_count=1,
        topology=dict(
            arms=len(angles),
            angles_rad=angles,
            lane_num=lane_num,
            lane_width_m=width,
            radius_m=radius,
            approach_length_m=approach_length,
            lane_movement_layout=layout,
        ),
        active_limit=cap,
        density_level=cap,
        demand=dict(headway_s=headway, right_turn_preference=right_share),
        horizon_s=110.0,
        variant=variant,
        formation="staggered_occupancy_checked_native_spawns_then_reactive_route_following",
    )
    spec["id"] = digest(spec)[:24]
    return spec
