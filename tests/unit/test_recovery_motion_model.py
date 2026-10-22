"""Le point prédit est bien le centre, pour les deux sens de marche."""

import math
import pytest
from cav_recovery.recovery.motion_model import center_pose, center_sweep
from cav_recovery.recovery.cooperative import CooperativePolicy, endpoint_pose
from cav_recovery.recovery.contracts import Action
from cav_recovery.recovery.feasibility import search
from cav_recovery.recovery.protocol import Endpoint
from cav_recovery.recovery.safety import trajectory
from cav_recovery.recovery.contracts import RecoveryContract
from tests.unit.test_recovery_prerequisites import view


@pytest.mark.parametrize("speed", [-0.6, 0.4, 0.8])
def test_straight_motion_keeps_the_historical_prediction(speed):
    own = view().own
    assert center_pose(
        own, Action(speed), 2, rear_axle_offset_m=1.4166
    ) == pytest.approx(endpoint_pose(own, Action(speed), 2))


def test_turning_center_includes_the_measured_rear_axle_offset():
    own = view().own
    action = Action(-0.6, 0.6)
    rear = 1.4166
    old = endpoint_pose(own, action, 4)
    actual = center_pose(own, action, 4, rear_axle_offset_m=rear)
    assert actual[0] - old[0] == pytest.approx(rear * (math.cos(actual[2]) - 1))
    assert actual[1] - old[1] == pytest.approx(rear * math.sin(actual[2]))
    assert math.dist(actual[:2], old[:2]) > 0.5


def test_sweep_contains_every_sampled_body_center_and_its_initial_footprint():
    from shapely.geometry import Point

    own = view().own
    action = Action(-0.6, -0.3)
    sweep = center_sweep(own, action, 2, rear_axle_offset_m=1.4166)
    for distance in [0, 0.1, 0.7, 1.0, 2.0]:
        pose = center_pose(own, action, distance, rear_axle_offset_m=1.4166)
        assert sweep.covers(Point(pose[:2]))


def test_historical_forecasts_remain_the_default_outside_explicit_audit_variants():
    policy = CooperativePolicy("V0", Endpoint("V0"), RecoveryContract())
    assert policy.pose_predictor is endpoint_pose
    assert policy.sweep_predictor is trajectory
    assert search.__kwdefaults__["pose_predictor"] is endpoint_pose
    assert search.__kwdefaults__["sweep_predictor"] is trajectory
