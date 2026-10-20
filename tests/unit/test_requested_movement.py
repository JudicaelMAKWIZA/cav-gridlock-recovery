"""Le plan explicite part du corps propre et respecte le mouvement non holonome."""

import math


from cav_recovery.d01.observer import EgoObserver
from tests.support.d01_fixtures import packet


def planned_packet():
    row = packet()
    row["ego"]["position_xy_m"][1] = 0.1
    distance = 1.2 * row["ego"]["length_m"]
    row["own_controls"]["planned_path_xyh_m"] = [
        [distance * index / 24, 0.1, 0.0] for index in range(25)
    ]
    row["own_controls"]["wheelbase_m"] = 2.7
    row["own_controls"]["max_steering_rad"] = math.radians(45)
    return row


def test_explicit_own_plan_does_not_teleport_an_offset_vehicle_to_the_lane_axis():
    row = planned_packet()
    result = EgoObserver().update(row)
    assert result["status"] == "SUSPECTED" and result["current_CAV_occupancy_proven"]
    assert result["own_state"]["position_xy_m"][1] == 0.1
    assert row["mission"]["own_route_xy_m"][0][1] == 0.0


def test_plan_cannot_begin_at_another_position_or_slide_sideways():
    row = planned_packet()
    row["own_controls"]["planned_path_xyh_m"][0][1] = 0.0
    assert EgoObserver().update(row)["reason"] == "OWN_PLAN_ORIGIN_MISMATCH"
    row = planned_packet()
    for index, pose in enumerate(row["own_controls"]["planned_path_xyh_m"]):
        pose[1] += index * 0.1
    assert EgoObserver().update(row)["reason"] == "OWN_PLAN_NONHOLONOMIC_MISMATCH"


def test_incomplete_or_nondrivable_plan_does_not_supply_occupancy_proof():
    row = planned_packet()
    row["own_controls"]["planned_path_xyh_m"] = row["own_controls"][
        "planned_path_xyh_m"
    ][:5]
    assert EgoObserver().update(row)["reason"] == "INCOMPLETE_OWN_MOVEMENT"
    row = planned_packet()
    row["ego"]["position_xy_m"][1] = 2.0
    for pose in row["own_controls"]["planned_path_xyh_m"]:
        pose[1] = 2.0
    assert EgoObserver().update(row)["reason"] == "MOVEMENT_ROAD_ADMISSIBILITY_UNKNOWN"
