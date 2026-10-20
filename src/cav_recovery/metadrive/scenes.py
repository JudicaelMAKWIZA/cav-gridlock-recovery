"""Essais physiques déclarés dont les métadonnées restent lisibles sans moteur.

Ces essais de faisabilité utilisent une instrumentation globale distincte
de D01. Les destinations précèdent la création ; les placements sont déclarés."""

PHYSICAL_SCENES = {
    "intersection_approaches": dict(
        geometry="native_unsignaled_intersection",
        formation="upstream_prepared_spawns_then_driving",
        hypothesis="Concurrent approach paths may constrain progress; no symmetric courtesy rule.",
        agents={
            "agent0": dict(
                spawn_lane_index=(">>", ">>>", 1),
                spawn_longitude=35.0,
                spawn_lateral=0,
                destination="1X1_1_",
            ),
            "agent1": dict(
                spawn_lane_index=("-1X2_1_", "-1X2_0_", 0),
                spawn_longitude=35.0,
                spawn_lateral=0,
                destination="1X0_1_",
            ),
            "agent2": dict(
                spawn_lane_index=("-1X1_1_", "-1X1_0_", 0),
                spawn_longitude=35.0,
                spawn_lateral=0,
                destination="->>",
            ),
        },
    ),
    "lane_merge": dict(
        geometry="native_two_to_one_bottleneck",
        formation="upstream_prepared_spawns_then_driving",
        hypothesis="Two approach lanes share one neck; trajectory occupation need not imply deadlock.",
        agents={
            "agent0": dict(
                spawn_lane_index=(">>", ">>>", 0),
                spawn_longitude=20.0,
                spawn_lateral=0,
                destination="2Y0_1_",
            ),
            "agent1": dict(
                spawn_lane_index=(">>", ">>>", 1),
                spawn_longitude=20.0,
                spawn_lateral=0,
                destination="2Y0_1_",
            ),
        },
    ),
    "shared_passage": dict(
        geometry="single_shared_strip_with_west_bay",
        formation="prepared_opposing_entrants_then_driving",
        hypothesis="Opposing cars cannot exchange longitudinal order inside the narrow strip; test retreat to west bay.",
        agents={
            "agent0": dict(
                spawn_lane_index=("west", "east", 0),
                spawn_longitude=28.0,
                spawn_lateral=0,
                destination="east",
            ),
            "agent1": dict(
                spawn_lane_index=("east", "west", 0),
                spawn_longitude=26.0,
                spawn_lateral=0,
                destination="west",
            ),
        },
    ),
}

PHYSICAL_POLICIES = ("road_following", "global_clearance")


def agents_for(case, legacy):
    return PHYSICAL_SCENES[case]["agents"] if case in PHYSICAL_SCENES else legacy[case]
