"""Inventorie les sources sans importer ni réattribuer les droits de données externes."""

from pathlib import Path
import importlib


def inventory(probe=False):
    from ..metadrive import METADRIVE_COMMIT

    native = []
    classes = (
        (
            "intersection",
            "metadrive.envs.marl_envs.marl_intersection",
            "MultiAgentIntersectionEnv",
        ),
        (
            "bottleneck",
            "metadrive.envs.marl_envs.marl_bottleneck",
            "MultiAgentBottleneckEnv",
        ),
        (
            "roundabout",
            "metadrive.envs.marl_envs.marl_inout_roundabout",
            "MultiAgentRoundaboutEnv",
        ),
        (
            "parking",
            "metadrive.envs.marl_envs.marl_parking_lot",
            "MultiAgentParkingLotEnv",
        ),
    )
    for name, module, cls in classes:
        item = dict(
            name=name,
            module=module,
            environment=cls,
            qualification="unlabeled_until_C1_C7",
        )
        if probe:
            env = None
            try:
                env = getattr(importlib.import_module(module), cls)(
                    dict(
                        num_agents=2,
                        num_scenarios=10,
                        use_render=False,
                        allow_respawn=False,
                        traffic_density=0,
                        log_level=40,
                        vehicle_config=dict(
                            lidar=dict(num_lasers=0, num_others=0),
                            side_detector=dict(num_lasers=0),
                            lane_line_detector=dict(num_lasers=0),
                        ),
                    )
                )
                env.reset(seed=1)
                item.update(
                    native_reset=True,
                    vehicles=[
                        dict(
                            length_m=float(v.LENGTH),
                            width_m=float(v.WIDTH),
                            destination=v.config["destination"],
                        )
                        for v in env.agents.values()
                    ],
                    roads=sum(
                        len(r) for r in env.current_map.road_network.graph.values()
                    ),
                )
            except Exception as error:
                item.update(native_reset=False, detail=str(error))
            finally:
                if env is not None:
                    env.close()
        native.append(item)
    return dict(
        native_commit=METADRIVE_COMMIT,
        native=native,
        production="native StraightLane/NodeRoadNetwork/Bullet blocks; all agents are controlled CAV; no replay traffic",
        external=dict(
            repository="https://github.com/metadriverse/metadrive-scenario",
            inspected_commit="38361bbe091ffa669a2703935fb62683187ffce6",
            announced_synthetic=3000,
            announced_waymo=1165,
            release="MetaDrive-Scenario-0.0.1.0",
            archive="synthetic_and_real_data.zip",
            archive_bytes=891225821,
            format="Python pickle: env_class/config/scenarios; Waymo serialized scenarios",
            code_license="Apache-2.0",
            data_license="Waymo/source data terms not verified; no permission inferred from code license",
            compatibility="legacy adapter imports removed real_data_envs.waymo_idm_env and gym; not directly compatible with pinned runtime",
            replay="Waymo trajectories replayed by default; legacy --idm_traffic requests reactive traffic",
            downloaded=False,
            production_eligible=False,
            vehicle_dimensions="record-specific, not verified without data; never assumed equal to native cars",
            decision="native generation retained; importer requires schema migration, vehicle/mission/reactivity and data-rights verification",
        ),
    )
