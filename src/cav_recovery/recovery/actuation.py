"""Actionneur REC-PHYS-2 : sortir du repos numérique sans imposer de vitesse."""

from ..metadrive.observation import signed_speed
from ..metadrive.control import BrakeVehicle
from metadrive.component.vehicle.vehicle_type import vehicle_type, vehicle_class_to_type


class RecoveryVehicle(BrakeVehicle):
    """Même gabarit et frein mesuré que M01, dans un domaine identifié."""


vehicle_type["cgr_recovery_v2"] = RecoveryVehicle
vehicle_class_to_type[RecoveryVehicle] = "cgr_recovery_v2"


def command(vehicle, action, contract):
    """Freine un vrai changement de sens ; tolère le résidu d'arrêt mesuré.

    Le domaine à 1 cm/s est explicite et propre à la récupération. La mesure
    brute, K04 et l'actionneur historique M02 restent intacts. La force reste
    bornée et Bullet décide du mouvement réel.
    """
    speed = signed_speed(vehicle)
    target = action.speed_mps
    changing_direction = (
        speed * target < 0 and abs(speed) > contract.direction_change_speed_mps
    )
    braking = (
        abs(target) < 1e-8
        or changing_direction
        or target > 0
        and speed > target + 0.05
        or target < 0
        and speed < target - 0.05
    )
    vehicle.explicit_brake = braking
    vehicle.enable_reverse = target < 0 and not braking
    force = -1.0 if braking else max(-0.4, min(0.4, 0.35 * (target - speed)))
    return [action.steering, force]
