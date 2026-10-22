"""Garde local d'exécution : perception reçue et carte propre, jamais env.agents."""

from dataclasses import dataclass
import math

import numpy as np
from shapely.geometry import Polygon
import shapely
from shapely.ops import unary_union

from ..metadrive.observation import rectangle
from .contracts import Action, RecoveryContract


@dataclass
class LocalView:
    ego: str
    at_s: float
    own: dict
    scan: dict
    messages: list
    road: object
    route: object
    reverse_used_m: float = 0.0
    own_commitment_until_s: float = 0.0
    own_controls: dict | None = None
    diagnostic: dict | None = None
    graph: dict | None = None
    measurement_contract: str | None = None


def trajectory(own, action, distance):
    """Court balayage du braquage demandé ; aucune recherche de dégagement."""
    x, y = own["position"]
    heading = own["heading_rad"]
    sign = -1 if action.speed_mps < 0 else 1
    curvature = math.tan(action.steering * own["max_steering_rad"]) / own["wheelbase_m"]
    poses = []
    for distance in np.linspace(
        0, sign * distance, max(3, math.ceil(distance / 0.05) + 1)
    ):
        angle = heading + curvature * distance
        if abs(curvature) < 1e-8:
            px, py = x + distance * math.cos(heading), y + distance * math.sin(heading)
        else:
            px = x + (math.sin(angle) - math.sin(heading)) / curvature
            py = y - (math.cos(angle) - math.cos(heading)) / curvature
        poses.append(rectangle((px, py), angle, own["length_m"], own["width_m"]))
    return unary_union(poses)


class LocalGuard:
    """Refuse ou réduit une commande, sans choisir l'agent ou sa priorité.

    Domaine expérimental : CAV de largeur >=1,7 m, rayons 360° coplanaires,
    carte connue, commandes bornées. Ce n'est pas un certificat universel.
    """

    def __init__(self, contract=None):
        self.contract = contract or RecoveryContract()

    def check(self, view, action, *, require_agreement=True):
        c = self.contract
        if action.speed_mps == 0:
            return dict(
                accepted=True, reason="HOLD", requested=0.0, accepted_speed_mps=0.0
            )
        if view.at_s < view.own_commitment_until_s - 1e-7:
            return self._refuse(action, "OWN_HOLD_COMMITMENT")
        if not -c.maximum_reverse_mps <= action.speed_mps <= c.maximum_forward_mps:
            return self._refuse(action, "ACTION_LIMIT")
        if action.speed_mps < 0 and view.reverse_used_m >= c.reverse_budget_m - 0.02:
            return self._refuse(action, "REVERSE_BUDGET")
        scan = view.scan
        if (
            scan.get("missing", False)
            or not 0 <= view.at_s - scan["at_s"] <= c.perception_max_age_s
        ):
            return self._refuse(action, "PERCEPTION_UNAVAILABLE")
        rays = scan["ranges"]
        if len(rays) < c.minimum_rays or not all(
            math.isfinite(r) and 0 < r <= 1 for r in rays
        ):
            return self._refuse(action, "PERCEPTION_COVERAGE")
        if scan.get("full_circle") is not True:
            return self._refuse(action, "PERCEPTION_COVERAGE")
        own = view.own
        if (
            action.speed_mps * own["speed_mps"] < 0
            and abs(own["speed_mps"]) > c.direction_change_speed_mps
        ):
            # Pendant l'arrêt, le corps continue dans son sens actuel. Ne pas
            # autoriser son enveloppe opposée comme si le signe avait déjà changé.
            return self._refuse(action, "SIGN_CHANGE_REQUIRES_STOP")
        speed = abs(own["speed_mps"])
        if (
            action.speed_mps < 0
            and c.reverse_budget_m - view.reverse_used_m
            < speed * c.step_s + speed**2 / 2 + 0.02
        ):
            return self._refuse(action, "REVERSE_STOPPING_BUDGET")
        length = max(
            0.25,
            speed * c.step_s + speed**2 / 2 + abs(action.speed_mps) * c.step_s + 0.15,
        )
        sweep = trajectory(own, action, length)
        if not view.road.buffer(1e-5).covers(sweep.buffer(0.02)):
            return self._refuse(action, "ROAD_ENVELOPE")
        xy = np.asarray(own["position"])
        angles = own["heading_rad"] + np.arange(len(rays)) * (2 * math.pi / len(rays))
        distances = np.asarray(rays) * scan["range_m"]
        # Les bords entre rayons ne sont pas une visibilité parfaite : retirer
        # la réserve angulaire et une marge d'exécution avant d'autoriser.
        endpoints = (
            xy + np.column_stack((np.cos(angles), np.sin(angles))) * distances[:, None]
        )
        visible = Polygon(endpoints)
        if not visible.is_valid:
            visible = visible.buffer(0)
        # L'erreur angulaire est bornée sur la zone balayée à vérifier,
        # pas sur les 30 m du capteur : la distance transverse entre rayons
        # croît avec le rayon du point testé. La carte visible reste entière.
        polygons = list(sweep.geoms) if hasattr(sweep, "geoms") else [sweep]
        radius = max(
            math.dist(xy, point)
            for polygon in polygons
            for point in polygon.exterior.coords
        )
        angular = min(scan["range_m"], radius + 0.5) * math.sin(math.pi / len(rays))
        horizon = c.step_s + view.at_s - scan["at_s"]
        received = [
            m
            for m in view.messages
            if m["kind"] == "STATE"
            and m["expires_s"] >= view.at_s
            and 0 <= view.at_s - m["sent_s"] <= c.perception_max_age_s
        ]
        reported_bodies = [
            rectangle(
                m["position"],
                m["payload"]["heading_rad"],
                m["payload"]["length_m"],
                m["payload"]["width_m"],
            )
            for m in received
        ]
        relevant_hits = distances < min(8.0, scan["range_m"] - 0.01)
        matched = (
            shapely.dwithin(
                shapely.points(endpoints), unary_union(reported_bodies), 0.15
            )
            if reported_bodies
            else np.zeros(len(rays), dtype=bool)
        )
        unknown_hit = bool(np.any(relevant_hits & ~matched))
        # Un corps communiqué immobile n'a pas instantanément la vitesse
        # maximale : réserve d'accélération 4 m/s², à vérifier dans le domaine.
        known_speed = max(
            (
                abs(m["payload"]["speed_mps"])
                for m in received
                if math.dist(m["position"], own["position"]) < 10.0
            ),
            default=0.0,
        )
        dynamic_reserve = (
            c.unknown_object_speed_bound_mps * horizon
            if unknown_hit
            else known_speed * horizon
            + 0.5 * c.neighbor_acceleration_bound_mps2 * horizon**2
        )
        clearance = visible.buffer(-(c.margin_m + angular + dynamic_reserve))
        present = rectangle(xy, own["heading_rad"], own["length_m"], own["width_m"])
        required = sweep.difference(present.buffer(0.03))
        if not clearance.covers(required):
            return self._refuse(action, "RAY_OCCUPANCY_OR_UNKNOWN")
        for message in view.messages:
            if message["kind"] != "STATE" or message["expires_s"] < view.at_s:
                continue
            state = message["payload"]
            age = view.at_s - message["sent_s"]
            if age < -1e-7:
                return self._refuse(action, "FUTURE_MESSAGE")
            other = rectangle(
                message["position"],
                state["heading_rad"],
                state["length_m"],
                state["width_m"],
            )
            reserve = (
                c.margin_m
                + abs(state["speed_mps"]) * (age + c.step_s)
                + state["speed_mps"] ** 2 / 2
                + 0.5 * c.neighbor_acceleration_bound_mps2 * (age + c.step_s) ** 2
            )
            if sweep.intersects(other.buffer(reserve)):
                return self._refuse(action, "RECEIVED_OCCUPANCY")
        if action.speed_mps < 0 and require_agreement:
            partners = {
                m["sender"]
                for m in view.messages
                if m["kind"] == "STATE"
                and math.dist(m["position"], own["position"]) <= 10.0
            }
            agreed = {
                m["sender"]
                for m in view.messages
                if m["kind"] == "AGREE_HOLD"
                and m["attempt"] == action.attempt
                and m["payload"]["until_s"] >= view.at_s
                and m["expires_s"] >= view.at_s
            }
            if partners - agreed:
                return self._refuse(action, "COOPERATIVE_AGREEMENT_MISSING")
        return dict(
            accepted=True,
            reason="LOCAL_ENVELOPE",
            requested=action.speed_mps,
            accepted_speed_mps=action.speed_mps,
            tested_distance_m=length,
            scope="sampled_local_guard_experimental_not_global_certificate",
        )

    @staticmethod
    def _refuse(action, reason):
        return dict(
            accepted=False,
            reason=reason,
            requested=action.speed_mps,
            accepted_speed_mps=0.0,
        )

    def limit(self, view, action):
        """Même masque pour chaque méthode ; ralentir ne réordonne pas les CAV."""
        attempts = [action.speed_mps]
        if action.speed_mps > 0.5:
            attempts += [min(action.speed_mps, 0.8), 0.4]
        receipt = None
        for speed in dict.fromkeys(attempts):
            receipt = self.check(
                view, Action(speed, action.steering, action.reason, action.attempt)
            )
            if receipt["accepted"]:
                return Action(
                    speed, action.steering, action.reason, action.attempt
                ), receipt
        return Action(0.0, action.steering, "guard_hold"), receipt
