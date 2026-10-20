"""Commandes physiques issues des essais F02, sans calibration véhicule réelle."""

from dataclasses import dataclass
import math

from metadrive.component.vehicle.vehicle_type import (
    StaticDefaultVehicle,
    vehicle_type,
    vehicle_class_to_type,
)

from .observation import body, rectangle, signed_speed


MINIMUM_CLEARANCE_M = 1.0
BRAKING_DECELERATION_MPS2 = 1.0
REACTION_TIME_S = 0.1
PATH_TOLERANCE_M = 0.1


def stopping_distance(speed_mps):
    """Réserve expérimentale : un pas de réaction puis freinage à 1 m/s².

    Cette enveloppe doit être vérifiée pour le véhicule et le sol utilisés ;
    ce n'est pas une calibration du frein ni une garantie pour d'autres scènes.
    """
    if not math.isfinite(speed_mps):
        raise ValueError("Vitesse de freinage non finie.")
    speed = abs(speed_mps)
    return speed * REACTION_TIME_S + speed**2 / (2 * BRAKING_DECELERATION_MPS2)


class BrakeVehicle(StaticDefaultVehicle):
    """Le frein des quatre roues fonctionne aussi pendant le recul."""

    explicit_brake = False
    brake_gain = 10.0

    def _apply_throttle_brake(self, value):
        super()._apply_throttle_brake(value)
        if self.explicit_brake:
            force = min(
                self.config["max_brake_force"], 2 + self.brake_gain * self.speed
            )
            for wheel in range(4):
                self.system.applyEngineForce(0, wheel)
                self.system.setBrake(force, wheel)


# Enregistrement local prévu par MetaDrive ; aucun fichier installé n'est modifié.
vehicle_type["cgr_independent_brake"] = BrakeVehicle
vehicle_class_to_type[BrakeVehicle] = "cgr_independent_brake"


@dataclass(frozen=True)
class DriveCommand:
    target_speed_mps: float
    steering: float = 0.0

    def __post_init__(self):
        if (
            not math.isfinite(self.target_speed_mps)
            or not math.isfinite(self.steering)
            or abs(self.steering) > 1
        ):
            raise ValueError("Commande de conduite invalide.")


def apply_command(vehicle, command):
    speed = signed_speed(vehicle)
    target = command.target_speed_mps
    braking = (
        abs(target) < 1e-8
        or speed * target < -0.001
        or (target > 0 and speed > target + 0.05)
        or (target < 0 and speed < target - 0.05)
    )
    vehicle.explicit_brake = braking
    vehicle.enable_reverse = target < 0 and not braking
    force = -1.0 if braking else max(-0.4, min(0.4, 0.35 * (target - speed)))
    return [command.steering, force]


def path_poses(vehicle, distance, spacing=0.2):
    """Suit la route native courante et son raccord suivant, sans extrapolation."""
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("Portée du chemin invalide.")
    lanes = vehicle.navigation.current_ref_lanes
    if not lanes:
        return [], False
    lane = min(lanes, key=lambda item: abs(item.local_coordinates(vehicle.position)[1]))
    longitude = float(lane.local_coordinates(vehicle.position)[0])
    next_lanes = vehicle.navigation.next_ref_lanes
    # La navigation peut garder le raccord précédent alors que le centre est déjà sur le suivant.
    if longitude > lane.length and next_lanes:
        end = lane.position(lane.length, 0)
        matching = [
            candidate
            for candidate in next_lanes
            if math.dist(candidate.position(0, 0), end) <= 0.1
        ]
        if len(matching) != 1:
            return [], False
        candidate = matching[0]
        s, lateral = candidate.local_coordinates(vehicle.position)
        if s < 0 or abs(lateral) > candidate.width / 2 + 0.05:
            return [], False
        lane, longitude, next_lanes = candidate, float(s), None
    if longitude < -0.05 or longitude > lane.length + 0.05:
        return [], False
    longitude = max(0.0, min(longitude, lane.length))
    remaining, poses = float(distance), []
    for _ in range(2):
        used = min(remaining, lane.length - longitude)
        offsets = [i * spacing for i in range(int(used / spacing) + 1)]
        if not offsets or used - offsets[-1] > 1e-8:
            offsets.append(used)
        for offset in offsets:
            s = longitude + offset
            poses.append(
                (
                    tuple(map(float, lane.position(s, 0))),
                    float(lane.heading_theta_at(s)),
                )
            )
        remaining -= used
        if remaining <= 1e-8:
            return poses, True
        if not next_lanes:
            break
        end = lane.position(lane.length, 0)
        distances = [
            (math.dist(candidate.position(0, 0), end), candidate)
            for candidate in next_lanes
        ]
        distances.sort(key=lambda pair: pair[0])
        if distances[0][0] > 0.1 or (
            len(distances) > 1 and abs(distances[0][0] - distances[1][0]) < 1e-7
        ):
            break
        lane, longitude, next_lanes = distances[0][1], 0.0, None
    return poses, False


def guard_blockers(vehicle_id, env, distance):
    """Protège le trajet avant et réserve aussi le freinage des autres CAV.

    L'enveloppe adverse est isotrope : elle couvre aussi un mouvement transversal,
    au prix de refus conservateurs. Le garde reste global et échantillonné.
    """
    from shapely.ops import unary_union

    vehicle = env.agents[vehicle_id]
    poses, complete = path_poses(vehicle, distance)
    if not poses or not complete:
        return [], False
    margin = (MINIMUM_CLEARANCE_M + PATH_TOLERANCE_M) / 2
    sweep = unary_union(
        [
            rectangle(xy, heading, vehicle.LENGTH, vehicle.WIDTH, margin)
            for xy, heading in poses
        ]
    )
    return sorted(
        key
        for key, other in env.agents.items()
        if key != vehicle_id
        and sweep.intersects(
            body(other).buffer(margin + stopping_distance(math.hypot(*other.velocity)))
        )
    ), True
