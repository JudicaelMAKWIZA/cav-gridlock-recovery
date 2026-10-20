"""État propre MetaDrive et transport applicatif idéal V2V-01."""

import math

from ..communication.v2v import V2VExchange


class MetaDriveExchange(V2VExchange):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.previous_positions = {}

    def _own_state(self, item, row, time_s):
        xy = list(row["position_xy_m"])
        if len(xy) != 2 or not all(
            math.isfinite(value)
            for value in xy + [row["heading_rad"], row["signed_speed_mps"]]
        ):
            raise ValueError("État propre V2V non fini.")
        previous = self.previous_positions.get(item)
        self.previous_positions[item] = (time_s, xy)
        return dict(
            vehicle_id=item,
            observed_at_s=time_s,
            position_xy_m=xy,
            state_schema_version="cgr-metadrive-state-1",
            coordinate_frame="metadrive_local_xy_m",
            heading_rad=row["heading_rad"],
            signed_speed_mps=row["signed_speed_mps"],
            length_m=row["length_m"],
            width_m=row["width_m"],
            intended_destination=row["mission"]["destination"],
            progress_delta_m=math.dist(previous[1], xy) if previous else None,
            progress_kind="unsigned_distance_not_mission_progress",
            progress_window_s=time_s - previous[0] if previous else None,
            source="metadrive_ego_emulation_not_radio",
            confidence="simulated_exact_ego",
        )

    def observe(self, time_s, readings):
        for item in set(self.previous_positions) - set(readings):
            self.previous_positions.pop(item)
        return super().observe(time_s, readings)
