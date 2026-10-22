"""REC-PLAN-1 : transmettre le plan propre que le conducteur sait déjà calculer."""

import copy
from types import SimpleNamespace

from shapely.geometry import LineString

from ..metadrive.planning import requested_path


class OwnPlanProjector:
    """Ni voisins globaux ni références : pose, mission et gabarit propres."""

    version = "REC-PLAN-1"

    def __init__(self):
        self.routes = {}

    def project(self, packet, *, wheelbase_m, max_steering_rad):
        result = copy.deepcopy(packet)
        identity = (packet["ego_id"], packet["mission"]["version"])
        if identity not in self.routes:
            line = LineString(packet["mission"]["own_route_xy_m"])
            self.routes[identity] = SimpleNamespace(
                line=line,
                position=lambda station, line=line: tuple(
                    line.interpolate(station).coords
                )[0],
            )
        own = packet["ego"]
        plan = requested_path(
            self.routes[identity],
            own["position_xy_m"],
            own["heading_rad"],
            max(0.0, own["signed_speed_mps"]),
            wheelbase_m,
            max_steering_rad,
            1.2 * own["length_m"],
        )
        result["own_controls"].update(
            planned_path_xyh_m=plan,
            wheelbase_m=wheelbase_m,
            max_steering_rad=max_steering_rad,
        )
        return result
