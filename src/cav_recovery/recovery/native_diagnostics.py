"""Acquisition native séparée : chaque observateur reçoit son seul paquet local."""

from ..d01.native import local_packet
from ..metadrive.cooperation import MetaDriveExchange
from ..dataset.native import local_drivable_regions
from .diagnostics import RecoveryDiagnostics


class NativeRecoveryDiagnostics:
    """Instrument facultatif, indépendant des choix de la baseline.

    Les 16 rayons historiques sont acquis séparément des 360 rayons du garde.
    La racine utilise les positions propres pour simuler le transport borné ;
    aucune liste globale n'est passée à D01 ou au C-RDG d'un véhicule.
    """

    def __init__(self, measurement_contract="raw"):
        self.service = RecoveryDiagnostics(measurement_contract=measurement_contract)
        self.telemetry = MetaDriveExchange(
            range_m=30, interval_s=0.4, lifetime_s=1.2, step_s=0.2
        )
        self.contexts = {}
        self.previous_velocity = {}
        self.map_context = None

    def observe(self, instance):
        at = instance.at_s
        if self.map_context is None:
            self.map_context = dict(
                observed_at_s=0.0,
                available_at_s=0.0,
                version="recovery-own-static-map-1",
                coordinate_frame="metadrive_local_xy_m",
                drivable_polygons_xy_m=[
                    [[float(v) for v in point] for point in region.exterior.coords]
                    for region in local_drivable_regions(instance.surface)
                ],
                lane_width_m=instance.spec["topology"]["lane_width_m"],
                rule_context="not_modelled",
            )
        readings = {}
        for ego, view in instance.views.items():
            own = view.own
            readings[ego] = dict(
                position_xy_m=own["position"],
                heading_rad=own["heading_rad"],
                signed_speed_mps=own["speed_mps"],
                length_m=own["length_m"],
                width_m=own["width_m"],
                mission=dict(destination=instance.missions[ego].destination),
            )
        if at:
            self.telemetry.observe(at, readings)
        packets = {}
        for ego, view in instance.views.items():
            own = view.own
            vehicle = instance.native.agents[ego]
            if ego not in self.contexts:
                agent = instance.agent_specs[ego]
                self.contexts[ego] = dict(
                    observed_at_s=at,
                    available_at_s=at,
                    version="initial-own-plan",
                    origin=str(agent["origin"]),
                    destination=str(instance.missions[ego].destination),
                    own_route_xy_m=[
                        [float(v) for v in p] for p in view.route.line.coords
                    ],
                    movement_id=ego + ":" + agent["movement_id"],
                    demand_forward=True,
                )
            velocity = list(map(float, vehicle.velocity))
            previous = self.previous_velocity.get(ego)
            measurement = dict(
                position_xy_m=own["position"],
                heading_rad=own["heading_rad"],
                signed_speed_mps=own["speed_mps"],
                velocity_xy_mps=velocity,
                acceleration_xy_mps2=None
                if previous is None
                else [(velocity[i] - previous[i]) / 0.2 for i in (0, 1)],
                mission_progress_m=own["progress_m"],
                length_m=own["length_m"],
                width_m=own["width_m"],
            )
            self.previous_velocity[ego] = velocity
            scan = instance.native.engine.get_sensor("lidar").perceive(
                vehicle,
                instance.native.engine.physics_world.dynamic_world,
                num_lasers=16,
                distance=30,
                show=False,
            )[0]
            command = instance.last_commands.get(ego)
            controls = dict(
                requested_speed_mps=command["proposed"]["speed_mps"]
                if command
                else 0.0,
                accepted_speed_mps=command["applied"]["speed_mps"] if command else 0.0,
                native_action=command["native_action"] if command else [0.0, 0.0],
            )
            packets[ego] = local_packet(
                ego,
                at,
                measurement,
                self.contexts[ego],
                self.map_context,
                list(map(float, scan)),
                controls,
                self.telemetry.knowledge(ego),
            )
        result = self.service.observe(packets, at)
        for ego, output in result.items():
            instance.views[ego].diagnostic = output["diagnostic"]
            instance.views[ego].graph = output["graph"]
            instance.views[ego].measurement_contract = output["measurement_contract"]
        return packets, result
