"""Contrats versionnés, fixés avant les comparaisons de récupération."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Action:
    """Une demande propre ; elle ne commande jamais directement un voisin."""

    speed_mps: float
    steering: float = 0.0
    reason: str = "nominal"
    attempt: str | None = None

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.speed_mps, self.steering)):
            raise ValueError("Commande non finie.")
        if abs(self.steering) > 1:
            raise ValueError("Braquage hors limites.")


@dataclass(frozen=True)
class RecoveryContract:
    """Domaine commun au nominal, à la baseline et aux futurs acteurs."""

    version: str = "recovery-physical-2"
    step_s: float = 0.2
    physics_s: float = 0.02
    maximum_forward_mps: float = 2.0
    maximum_reverse_mps: float = 0.6
    reverse_budget_m: float = 8.0
    margin_m: float = 0.15
    perception_max_age_s: float = 0.21
    message_lifetime_s: float = 0.6
    range_m: float = 30.0
    rays: int = 360
    minimum_rays: int = 180
    minimum_vehicle_width_m: float = 1.7
    stationary_speed_mps: float = 0.02
    stall_s: float = 35.0
    recovery_window_s: float = 10.0
    relapse_followup_s: float = 30.0
    effect_window_s: float = 5.0
    direction_change_speed_mps: float = 0.01
    neighbor_acceleration_bound_mps2: float = 6.0
    unknown_object_speed_bound_mps: float = 2.1

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name != "version" and (
                isinstance(value, bool) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError("Paramètre de récupération invalide : " + name)
        if (
            type(self.rays) is not int
            or type(self.minimum_rays) is not int
            or self.rays < self.minimum_rays
        ):
            raise ValueError("Résolution LiDAR non prise en charge.")
        if self.step_s != 0.2 or self.physics_s != 0.02:
            raise ValueError("Cette version impose les cadences natives 0,2/0,02 s.")

    def record(self):
        return asdict(self)
