"""Preuves géométriques prudentes issues du plan propre et des poses reçues."""

import math
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union


class GeometryError(ValueError):
    pass


def planned_poses(data, distance):
    """Lit uniquement le plan propre explicite, avec origine et longueur vérifiées."""
    plan = data["own_controls"].get("planned_path_xyh_m")
    if plan is None:
        return None
    ego = data["ego"]
    if math.dist(plan[0][:2], ego["position_xy_m"]) > 2e-6:
        raise GeometryError("OWN_PLAN_ORIGIN_MISMATCH")
    if (
        abs(
            math.atan2(
                math.sin(plan[0][2] - ego["heading_rad"]),
                math.cos(plan[0][2] - ego["heading_rad"]),
            )
        )
        > 2e-6
    ):
        raise GeometryError("OWN_PLAN_HEADING_MISMATCH")
    result = [(tuple(plan[0][:2]), plan[0][2])]
    travelled = 0.0
    for first, last in zip(plan, plan[1:]):
        segment = math.dist(first[:2], last[:2])
        if not 0 < segment <= 0.500002:
            raise GeometryError("OWN_PLAN_SAMPLING_UNSUPPORTED")
        remaining = distance - travelled
        fraction = min(1.0, remaining / segment)
        rotation = math.atan2(
            math.sin(last[2] - first[2]), math.cos(last[2] - first[2])
        )
        direction = math.atan2(last[1] - first[1], last[0] - first[0])
        middle_heading = first[2] + rotation / 2
        rounding_angle = 2e-6 / segment + 1e-6
        if (
            abs(
                math.atan2(
                    math.sin(direction - middle_heading),
                    math.cos(direction - middle_heading),
                )
            )
            > rounding_angle
        ):
            raise GeometryError("OWN_PLAN_NONHOLONOMIC_MISMATCH")
        controls = data["own_controls"]
        if "wheelbase_m" in controls:
            limit = math.tan(controls["max_steering_rad"]) / controls["wheelbase_m"]
            if abs(rotation) / segment > limit + rounding_angle / segment:
                raise GeometryError("OWN_PLAN_STEERING_LIMIT")
        result.append(
            (
                tuple(first[k] + fraction * (last[k] - first[k]) for k in (0, 1)),
                first[2] + fraction * rotation,
            )
        )
        travelled += segment
        if travelled >= distance - 2e-6:
            return result
    raise GeometryError("INCOMPLETE_OWN_MOVEMENT")


def footprint(position, heading, length, width):
    c, s = math.cos(heading), math.sin(heading)
    return Polygon(
        [
            (position[0] + c * x - s * y, position[1] + s * x + c * y)
            for x, y in (
                (-length / 2, -width / 2),
                (length / 2, -width / 2),
                (length / 2, width / 2),
                (-length / 2, width / 2),
            )
        ]
    )


def pose_error(length, width):
    """Borne d’arrondi des poses simulées à six décimales et réserve de calcul.

    Cette borne ne constitue ni une calibration de capteur ni une prédiction de mouvement."""
    return math.sqrt(2) * 0.5e-6 + math.hypot(length, width) * 0.25e-6 + 1e-7


class MovementGeometry:
    def __init__(self):
        self.static_key = self.pose_key = None

    def _route_poses(self, station, distance, heading):
        """Échantillonne le plan statique lorsque le conducteur ne déclare pas de trajectoire."""

        def at(s):
            q = self.route.interpolate(s)
            a = self.route.interpolate(max(0, s - 0.001))
            b = self.route.interpolate(min(self.route.length, s + 0.001))
            return (q.x, q.y), math.atan2(b.y - a.y, b.x - a.x)

        _, tangent = at(station)
        if (
            abs(
                math.atan2(
                    math.sin(tangent - heading),
                    math.cos(tangent - heading),
                )
            )
            > 0.1
        ):
            raise GeometryError("OWN_HEADING_OFF_ROUTE")
        # Les sommets du trajet complètent les échantillons curvilignes réguliers.
        count = max(1, math.ceil(distance / 0.25))
        stations = {station + distance * i / count for i in range(count + 1)}
        arc = 0.0
        coords = list(self.route.coords)
        for a, b in zip(coords, coords[1:]):
            arc += math.dist(a, b)
            if station < arc < station + distance:
                stations.add(arc)
        poses = [at(s) for s in sorted(stations)]
        return poses, tangent

    def build(self, data, movement_lengths):
        ego, mission, road = (data[k] for k in ("ego", "mission", "map"))
        key = (mission["own_route_xy_m"], road["drivable_polygons_xy_m"])
        if self.static_key != key:
            route = LineString(mission["own_route_xy_m"])
            polygons = [Polygon(points) for points in road["drivable_polygons_xy_m"]]
            if (
                route.length <= 0
                or not route.is_simple
                or any(not p.is_valid or p.area <= 0 for p in polygons)
            ):
                raise GeometryError("INVALID_STATIC_GEOMETRY")
            self.route, self.surface = route, unary_union(polygons)
            self.static_key, self.pose_key = key, None
        pose = (
            ego["position_xy_m"],
            ego["heading_rad"],
            ego["length_m"],
            ego["width_m"],
            movement_lengths,
            data["own_controls"].get("planned_path_xyh_m"),
            data["own_controls"].get("wheelbase_m"),
            data["own_controls"].get("max_steering_rad"),
        )
        if pose == self.pose_key:
            return self.result
        point = Point(ego["position_xy_m"])
        station = self.route.project(point)
        distance = movement_lengths * ego["length_m"]
        if station + distance > self.route.length + 1e-7:
            raise GeometryError("INCOMPLETE_OWN_MOVEMENT")
        own_plan = planned_poses(data, distance)
        if own_plan is None and self.route.distance(point) > 0.05:
            # Aucun raccord inventé ni affectation cachée de route n’est accepté.
            raise GeometryError("OWN_ROUTE_TRACKING_UNCERTAIN")

        if own_plan is None:
            poses, tangent = self._route_poses(station, distance, ego["heading_rad"])
        else:
            poses, tangent = own_plan, ego["heading_rad"]
        straight = all(
            abs(math.atan2(math.sin(h - tangent), math.cos(h - tangent))) < 1e-10
            for _, h in poses
        )
        if straight:
            poses = [poses[0], poses[-1]]
        bodies = [footprint(p, h, ego["length_m"], ego["width_m"]) for p, h in poses]
        own = footprint(
            ego["position_xy_m"], ego["heading_rad"], ego["length_m"], ego["width_m"]
        )
        # La preuve intérieure utilise les empreintes aux positions définies sur le plan propre.
        # L’admissibilité utilise une enveloppe extérieure prudente de translation et rotation.
        radius = math.hypot(ego["length_m"], ego["width_m"]) / 2
        outer = []
        for i, (a, b) in enumerate(zip(bodies, bodies[1:])):
            dh = abs(
                math.atan2(
                    math.sin(poses[i + 1][1] - poses[i][1]),
                    math.cos(poses[i + 1][1] - poses[i][1]),
                )
            )
            if dh > 0.15:
                raise GeometryError("SHARP_ROUTE_CORNER_UNSUPPORTED")
            outer.append(
                a.union(b).convex_hull.buffer(
                    radius * dh + pose_error(ego["length_m"], ego["width_m"])
                )
            )
        if not self.surface.covers(unary_union([own, *outer])):
            raise GeometryError("MOVEMENT_ROAD_ADMISSIBILITY_UNKNOWN")
        # En translation droite, l’enveloppe est le balayage continu exact ;
        # elle ne remplace pas un trajet tournant par une corde.
        sweep = (
            bodies[0].union(bodies[-1]).convex_hull if straight else unary_union(bodies)
        )
        self.result = dict(
            own=own,
            sweep=sweep,
            required=sweep.difference(own),
            station_m=station,
            distance_m=distance,
            error_m=pose_error(ego["length_m"], ego["width_m"]),
            origin_xy_m=ego["position_xy_m"],
            sweep_radius_m=max(
                math.dist(position, ego["position_xy_m"]) for position, _ in poses
            )
            + radius,
            eroded_geometries={},
        )
        self.pose_key = pose
        return self.result


def body_evidence(message, geometry):
    other_radius = math.hypot(message["length_m"], message["width_m"]) / 2
    if (
        math.dist(message["position_xy_m"], geometry["origin_xy_m"])
        > geometry["sweep_radius_m"] + other_radius + 1e-5
    ):
        return None
    other = footprint(
        message["position_xy_m"],
        message["heading_rad"],
        message["length_m"],
        message["width_m"],
    )
    error = geometry["error_m"] + pose_error(message["length_m"], message["width_m"])
    if error not in geometry["eroded_geometries"]:
        if len(geometry["eroded_geometries"]) >= 16:
            geometry["eroded_geometries"].clear()
        geometry["eroded_geometries"][error] = (
            geometry["own"].buffer(-error),
            geometry["required"].buffer(-error),
        )
    own_inner, required_inner = geometry["eroded_geometries"][error]
    other_inner = other.buffer(-error)
    if own_inner.intersects(other_inner):
        raise GeometryError("CURRENT_BODY_OVERLAP")
    overlap = required_inner.intersection(other_inner)
    if overlap.area <= 0:
        return None
    return dict(
        sender_id=message["sender_id"],
        message_id=message["message_id"],
        observed_at_s=message["observed_at_s"],
        intersection_area_m2=overlap.area,
        intersection_envelope_xy_m=[
            list(p) for p in overlap.convex_hull.exterior.coords
        ],
        resource_polygon_wkt=overlap.wkt,
        position_xy_m=list(message["position_xy_m"]),
        heading_rad=message["heading_rad"],
        length_m=message["length_m"],
        width_m=message["width_m"],
        source=message["source"],
        confidence=message["confidence"],
        error_bound_m=error,
    ), overlap


def anonymous_returns(data, geometry):
    lidar = data["lidar"]
    t = data["decision_at_s"]
    if lidar is None or lidar["missing"] or abs(lidar["observed_at_s"] - t) > 1e-7:
        return []
    ego = data["ego"]
    result = []
    for i, r in enumerate(lidar["normalized_ranges"]):
        if r is None or r >= 1:
            continue
        angle = ego["heading_rad"] + lidar["phase_offset_rad"] + 2 * math.pi * i / 16
        distance = lidar["range_m"] * r
        p = Point(
            ego["position_xy_m"][0] + distance * math.cos(angle),
            ego["position_xy_m"][1] + distance * math.sin(angle),
        )
        error = geometry["error_m"] + lidar["range_m"] * 0.5e-6 + distance * 0.5e-6
        if geometry["required"].contains(p.buffer(error)):
            result.append(
                dict(
                    ray=i,
                    position_xy_m=[p.x, p.y],
                    observed_at_s=t,
                    error_bound_m=error,
                    category="unknown",
                    identity=None,
                )
            )
    return result
