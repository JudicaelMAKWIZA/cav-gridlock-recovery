"""Plan propre à court horizon du conducteur nominal, sans données voisines."""

import math


def requested_path(
    route, position, heading, speed, wheelbase, max_steering_rad, distance
):
    """Déclare le mouvement demandé avec le même suivi pure pursuit que le conducteur.

    Ce plan cinématique commence à la pose mesurée. Il ne téléporte pas le CAV
    sur l'axe de voie et ne prédit pas le mouvement réel sous freinage. Sa
    courbure est bornée par le gabarit de direction du véhicule propre.
    """
    from shapely.geometry import Point

    x, y = map(float, position)
    angle = float(heading)
    poses = [[x, y, angle]]
    lookahead = 1.5 + max(0.0, speed) * 0.35
    count = math.ceil(distance / 0.25)
    step = distance / count
    maximum_curvature = math.tan(max_steering_rad) / wheelbase
    for _ in range(count):
        station = float(route.line.project(Point(x, y)))
        target_x, target_y = route.position(station + lookahead)
        error = math.atan2(target_y - y, target_x - x) - angle
        curvature = max(
            -maximum_curvature, min(maximum_curvature, 2 * math.sin(error) / lookahead)
        )
        middle_heading = angle + curvature * step / 2
        x += step * math.cos(middle_heading)
        y += step * math.sin(middle_heading)
        angle += curvature * step
        poses.append([x, y, angle])
    return [[round(value, 6) for value in pose] for pose in poses]
