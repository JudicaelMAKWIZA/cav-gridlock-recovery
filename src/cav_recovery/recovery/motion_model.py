"""Prévision REC-MOTION-2 : le centre du châssis n'est pas l'essieu arrière."""

import math
import numpy as np
from shapely.ops import unary_union

from ..metadrive.observation import rectangle


def center_pose(own, action, distance, *, rear_axle_offset_m):
    """Évolution cinématique à courbure constante, rendue au centre du châssis.

    La distance signée est longitudinale à l'essieu arrière. Le déport est une
    propriété propre du véhicule, pas une valeur ajustée sur l'issue du scénario.
    Ceci ne prédit ni le temps de parcours ni le glissement de Bullet.
    """
    if not 0 < rear_axle_offset_m < own["wheelbase_m"] or distance < 0:
        raise ValueError("Géométrie ou distance de manœuvre invalide.")
    heading = own["heading_rad"]
    curvature = math.tan(action.steering * own["max_steering_rad"]) / own["wheelbase_m"]
    signed = (-1 if action.speed_mps < 0 else 1) * distance
    angle = heading + curvature * signed
    rear_x = own["position"][0] - rear_axle_offset_m * math.cos(heading)
    rear_y = own["position"][1] - rear_axle_offset_m * math.sin(heading)
    if abs(curvature) < 1e-8:
        rear_x += signed * math.cos(heading)
        rear_y += signed * math.sin(heading)
    else:
        rear_x += (math.sin(angle) - math.sin(heading)) / curvature
        rear_y -= (math.cos(angle) - math.cos(heading)) / curvature
    return [
        rear_x + rear_axle_offset_m * math.cos(angle),
        rear_y + rear_axle_offset_m * math.sin(angle),
        angle,
    ]


def center_sweep(own, action, distance, *, rear_axle_offset_m):
    """Enveloppe cinématique continue approchée ; garde natif toujours séparé."""
    bodies = []
    steps = max(3, math.ceil(distance / 0.05) + 1)
    for station in np.linspace(0, distance, steps):
        pose = center_pose(own, action, station, rear_axle_offset_m=rear_axle_offset_m)
        bodies.append(rectangle(pose[:2], pose[2], own["length_m"], own["width_m"]))
    curvature = abs(
        math.tan(action.steering * own["max_steering_rad"]) / own["wheelbase_m"]
    )
    if curvature < 1e-8:
        return bodies[0].union(bodies[-1]).convex_hull
    angular_step = curvature * distance / (steps - 1)
    radius = (
        1 / curvature
        + rear_axle_offset_m
        + math.hypot(own["length_m"], own["width_m"]) / 2
    )
    sagitta = radius * (1 - math.cos(angular_step / 2))
    return unary_union(
        [a.union(b).convex_hull.buffer(sagitta) for a, b in zip(bodies, bodies[1:])]
    )
