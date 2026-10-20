"""Preuves géométriques prudentes issues du plan propre et des poses reçues."""

import math
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union


class GeometryError(ValueError):
    pass


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
        )
        if pose == self.pose_key:
            return self.result
        point = Point(ego["position_xy_m"])
        station = self.route.project(point)
        distance = movement_lengths * ego["length_m"]
        if station + distance > self.route.length + 1e-7:
            raise GeometryError("INCOMPLETE_OWN_MOVEMENT")
        if self.route.distance(point) > 0.05:
            # Aucun raccord inventé ni affectation cachée de route n’est accepté.
            raise GeometryError("OWN_ROUTE_TRACKING_UNCERTAIN")

        def at(s):
            q = self.route.interpolate(s)
            a = self.route.interpolate(max(0, s - 0.001))
            b = self.route.interpolate(min(self.route.length, s + 0.001))
            return (q.x, q.y), math.atan2(b.y - a.y, b.x - a.x)

        _, tangent = at(station)
        if (
            abs(
                math.atan2(
                    math.sin(tangent - ego["heading_rad"]),
                    math.cos(tangent - ego["heading_rad"]),
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
        )
        self.pose_key = pose
        return self.result


def body_evidence(message, geometry):
    other = footprint(
        message["position_xy_m"],
        message["heading_rad"],
        message["length_m"],
        message["width_m"],
    )
    error = geometry["error_m"] + pose_error(message["length_m"], message["width_m"])
    if geometry["own"].buffer(-error).intersects(other.buffer(-error)):
        raise GeometryError("CURRENT_BODY_OVERLAP")
    overlap = geometry["required"].buffer(-error).intersection(other.buffer(-error))
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
