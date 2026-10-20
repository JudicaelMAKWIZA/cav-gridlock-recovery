"""Application de test du transport, sans import d’un moteur."""

import math
from cav_recovery.communication.v2v import V2VExchange

COORDINATE_FRAME = "test_xy_m"
SOURCE = "analytical_ego_measurement"


class ExampleExchange(V2VExchange):
    def _own_state(self, item, row, time_s):
        position = tuple(row["position"])
        if len(position) != 2 or any(not math.isfinite(value) for value in position):
            raise ValueError("Position propre V2V invalide.")
        distance = row["distance"]
        if not math.isfinite(distance) or not math.isfinite(row["speed"]):
            raise ValueError("Distance ou vitesse propre V2V invalide.")
        previous = self.previous_distance.get(item)
        progress = distance - previous[1] if previous is not None else None
        if progress is not None and progress < -1e-6:
            raise ValueError("La distance parcourue d'un CAV ne peut pas reculer.")
        window = time_s - previous[0] if previous is not None else None
        self.previous_distance[item] = (time_s, distance)
        route = row["route"]
        index = row["route_index"]
        next_edge = route[index + 1] if 0 <= index < len(route) - 1 else None
        return {
            "vehicle_id": item,
            "observed_at_s": time_s,
            "position_xy_m": list(position),
            "coordinate_frame": COORDINATE_FRAME,
            "speed_mps": row["speed"],
            "lane_id": row["lane"],
            "intended_next_edge": next_edge,
            "progress_delta_m": progress,
            "progress_window_s": window,
            "source": SOURCE,
        }
