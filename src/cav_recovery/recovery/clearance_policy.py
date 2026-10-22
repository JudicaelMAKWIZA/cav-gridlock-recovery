"""Variante locale : évaluer le dégagement final, puis demander toute sa place."""

from shapely import wkt
from shapely.geometry import Point
from shapely.ops import unary_union

from ..metadrive.observation import rectangle
from .cooperative import CooperativePolicy, option_bank


class ClearancePolicy(CooperativePolicy):
    """REC-COOP-3 expérimental, sans lecture d'un plan ou d'une population oracle.

    Le classement privilégie une libération complète prévue. La demande arrière
    couvre 4,5 m et la durée maximale est 40 s. Le garde contrôle chaque commande
    réellement exécutée : une prévision cinématique n'est pas un certificat.
    """

    def __init__(self, ego, endpoint, contract):
        super().__init__(ego, endpoint, contract)
        self.request_distance_m = 4.5
        self.option_duration_s = 40.0

    def candidates(self, view, guard):
        own = view.own
        present = rectangle(
            own["position"], own["heading_rad"], own["length_m"], own["width_m"]
        )
        spaces = [
            wkt.loads(m["payload"]["space_wkt"])
            for m in view.messages
            if m["kind"] == "CLEAR_REQUEST" and m["target"] in (None, self.ego)
        ]
        requested = unary_union([p for p in spaces if p.intersects(present)])
        station = view.route.line.project(Point(own["position"]))
        options = []
        for action in option_bank():
            if (
                not action.speed_mps
                or not guard.check(view, action, require_agreement=False)["accepted"]
            ):
                continue
            best = None
            for distance in [0.5, 1.0, 2.5, 4.5]:
                if (
                    action.speed_mps < 0
                    and view.reverse_used_m + distance > self.contract.reverse_budget_m
                ):
                    continue
                if not view.road.covers(self.sweep_predictor(own, action, distance)):
                    continue
                pose = self.pose_predictor(own, action, distance)
                end = rectangle(pose[:2], pose[2], own["length_m"], own["width_m"])
                gain = (
                    present.intersection(requested).area
                    - end.intersection(requested).area
                )
                if action.speed_mps < 0 and gain <= 0.02:
                    continue
                complete = not requested.is_empty and not end.buffer(0.15).intersects(
                    requested
                )
                progress = view.route.line.project(Point(pose[:2])) - station
                score = 25 * complete + 2 * gain + progress - 0.2 * distance
                score -= 0.3 * view.route.line.distance(Point(pose[:2]))
                best = score if best is None else max(best, score)
            if best is not None:
                options.append((best, action))
        return [
            action
            for _, action in sorted(options, key=lambda item: item[0], reverse=True)
        ]
