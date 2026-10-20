"""Références géométriques indépendantes du garde de sécurité.

Les sondes gardent les autres corps fixes : une intersection ne témoigne
que du chemin sondé. Une sonde libre n’est pas un certificat dynamique.
Les commandes non sondées et les mouvements futurs restent inconnus."""

import math

from shapely.geometry import LineString, Polygon, box

from .observation import body, rectangle
from .control import path_poses


def probe(poses, length, width, surface, neighbors):
    if not poses:
        return dict(
            status="indeterminate_no_path_samples",
            tested_distance_m=0.0,
            clear_sample_distance_m=None,
            blockers=[],
            outside_surface=None,
        )
    previous = None
    distance = 0.0
    last_clear = 0.0
    for x, y, heading in poses:
        if previous is not None:
            distance += math.dist((x, y), previous)
        footprint = rectangle((x, y), heading, length, width)
        blockers = sorted(
            key
            for key, polygon in neighbors.items()
            if footprint.intersection(polygon).area > 1e-8
        )
        outside = not surface.covers(footprint)
        if blockers or outside:
            return dict(
                status="sampled_path_obstructed",
                tested_distance_m=distance,
                clear_sample_distance_m=last_clear,
                blockers=blockers,
                outside_surface=outside,
                witness_pose=[x, y, heading],
            )
        previous = (x, y)
        last_clear = distance
    return dict(
        status="geometric_candidate_dynamics_unverified",
        tested_distance_m=distance,
        clear_sample_distance_m=last_clear,
        blockers=[],
        outside_surface=False,
    )


def passage_certificate(
    readings, surface, *, width=3.4, bay_end_x=22.0, length=70.0, bay_center_y=3.0
):
    """Argument suffisant d’impossibilité d’échange en marche avant dans cette bande.

    Les disques inscrits interdisent l’échange d’ordre longitudinal si la largeur
    est inférieure à deux largeurs de voiture. La longueur interdit un cap
    transversal ; le véhicule orienté vers l’est ne peut rejoindre la baie ouest
    sans reculer. L’argument concerne cette géométrie et ces rectangles rigides,
    pas un détecteur universel de circulation."""
    ids = sorted(readings)
    established = False
    checks = {}
    if len(ids) == 2:
        a, b = (readings[key] for key in ids)
        dims = [dict(length=row["length_m"], width=row["width_m"]) for row in (a, b)]
        checks = dict(
            surface_matches_declared_geometry=surface.symmetric_difference(
                box(0, -width / 2, length, width / 2).union(
                    box(
                        0, bay_center_y - width / 2, bay_end_x, bay_center_y + width / 2
                    )
                )
            ).area
            < 1e-8,
            cannot_exchange_order_in_strip=width < sum(d["width"] for d in dims),
            cannot_turn_transverse=all(d["length"] > width for d in dims),
            both_bodies_east_of_bay=all(
                Polygon(row["body_xy_m"]).bounds[0] > bay_end_x for row in (a, b)
            ),
            initial_bodies_on_road=all(
                surface.covers(Polygon(row["body_xy_m"])) for row in (a, b)
            ),
            opposing_order=a["position_xy_m"][0] < b["position_xy_m"][0]
            and abs(a["heading_rad"]) < 0.05
            and abs(abs(b["heading_rad"]) - math.pi) < 0.05,
            opposite_fixed_destinations=a["mission"]["destination"] == "east"
            and b["mission"]["destination"] == "west",
        )
        established = all(checks.values())
    return dict(
        status="westward_retreat_required_for_joint_completion"
        if established
        else "indeterminate",
        reverse_gear_necessity="conditional_on_nonholonomic_no_sideslip_model"
        if established
        else "indeterminate",
        checks=checks,
        strip_width_m=width,
        bay_end_x_m=bay_end_x,
        assumptions=[
            "continuous planar rigid rectangles",
            "no lateral slip control or pose edits",
            "whole bodies remain on declared road",
            "no removal before native destination",
            "no additional bay east of the initial cars",
        ],
        scope="geometric_order_argument_not_D01_not_exhaustive_Bullet_reachability",
    )


def physical_snapshot(env, readings, surface, missions, time_s):
    polygons = {key: body(v) for key, v in env.agents.items()}
    rows = {}
    paths = {}
    for key, vehicle in sorted(env.agents.items()):
        row = readings[key]
        poses, complete = path_poses(vehicle, 4.0, spacing=0.05)
        poses = [(xy[0], xy[1], angle) for xy, angle in poses]
        neighbors = {other: poly for other, poly in polygons.items() if other != key}
        x, y = row["position_xy_m"]
        heading = row["heading_rad"]
        probes = {}
        probes["assigned_forward_4m"] = {
            **probe(poses, vehicle.LENGTH, vehicle.WIDTH, surface, neighbors),
            "path_complete": complete,
            "start_offset_from_body_center_m": math.dist((x, y), poses[0][:2])
            if poses
            else None,
        }
        assigned = probes["assigned_forward_4m"]
        if (
            not complete
            and assigned["status"] == "geometric_candidate_dynamics_unverified"
        ):
            assigned["status"] = "indeterminate_incomplete_assigned_path"
        assigned["start_connector"] = (
            "near_assigned_path"
            if poses and assigned["start_offset_from_body_center_m"] < 0.1
            else "unverified_body_to_assigned_path_connector"
        )
        # Les sondes sont cinématiques ; elles ne certifient ni dynamique ni limites de braquage.
        for name, sign, curvature in (
            ("reverse_straight_4m", -1, 0),
            ("forward_left_arc_4m", 1, 0.08),
            ("forward_right_arc_4m", 1, -0.08),
        ):
            samples = []
            for index in range(81):
                s = sign * index * 0.05
                angle = heading + curvature * s
                if curvature:
                    px = x + (math.sin(angle) - math.sin(heading)) / curvature
                    py = y - (math.cos(angle) - math.cos(heading)) / curvature
                else:
                    px, py = x + s * math.cos(heading), y + s * math.sin(heading)
                samples.append((px, py, angle))
            probes[name] = probe(
                samples, vehicle.LENGTH, vehicle.WIDTH, surface, neighbors
            )
        if poses:
            paths[key] = LineString([(p[0], p[1]) for p in poses]).buffer(
                vehicle.WIDTH / 2
            )
        final = env.current_map.road_network.get_lane(missions[key].final_lane)
        longitudinal, lateral = final.local_coordinates(vehicle.position)
        rows[key] = dict(
            probes=probes,
            remaining_final_lane_projection_m=missions[key].final_lane_length_m
            - float(longitudinal),
            final_lane_lateral_m=float(lateral),
            signed_longitudinal_speed_mps=row["signed_speed_mps"],
            unverified_alternatives=[
                "other steering histories",
                "coordinated moving neighbors",
                "full dynamic reachability",
            ],
            interpretation="frozen_neighbor_sampled_paths_only_not_absolute_blockage",
        )
    resources = []
    for index, left in enumerate(sorted(paths)):
        for right in sorted(paths)[index + 1 :]:
            shared = paths[left].intersection(paths[right])
            if shared.area > 1e-8:
                resources.append(
                    dict(
                        id=f"requested_path_overlap:{left}:{right}",
                        users=[left, right],
                        area_m2=shared.area,
                        polygon_wkt=shared.wkt,
                        occupants=[
                            key for key, p in polygons.items() if p.intersects(shared)
                        ],
                    )
                )
    return dict(
        time_s=time_s,
        vehicles=rows,
        resources=resources,
        reference="independent_current_geometry_not_guard_or_D01",
        sample_spacing_m=0.05,
        continuous_sweep_safety="not_certified_by_probes_native_execution_checked_every_0.02s",
    )
