"""Scientific QC must distinguish projection ambiguity from altered evidence."""

import math
import pytest
from shapely.geometry import LineString
from scripts.audit_route_precision import verify_poses


@pytest.fixture
def corner():
    line = LineString([(0, 0), (1, 0), (1, 1)])
    poses = [
        [1.0, 0.1, round(math.atan2(0.25, 0.05), 6)],
        [1.0, 0.5, round(math.pi / 2, 6)],
        [1.0, 0.5, round(math.pi / 2, 6)],
    ]
    return line, poses


def test_nonunique_rounded_projection_is_not_a_route_deviation(corner):
    from shapely.geometry import Point

    line, poses = corner
    assert abs(line.project(Point(0.9, 0.0999999)) - 1.1) > 0.1
    result = verify_poses(line, 1.1, 0.4, poses, [0.9, 0.0999999])
    assert result["valid"]
    assert 0 < result["nearest_distance_excess_m"] < 1e-6


def test_arbitrary_recorded_progress_cannot_excuse_a_bad_projection(corner):
    line, poses = corner
    result = verify_poses(line, 1.2, 0.4, poses, [0.9, 0.1])
    assert not result["valid"]
    assert (
        "recorded_projection_not_nearest_within_quantization_bound" in result["issues"]
    )


@pytest.mark.parametrize("delta", [3e-6, -3e-6, 0.01, -0.01])
def test_altered_reference_positions_remain_invalid(corner, delta):
    line, poses = corner
    poses[1][0] += delta
    result = verify_poses(line, 1.1, 0.4, poses, [0.9, 0.1])
    assert not result["valid"]
    assert "world_position_not_reproduced" in result["issues"]


def test_altered_heading_remains_invalid(corner):
    line, poses = corner
    poses[1][2] += 0.001
    assert (
        "world_heading_not_reproduced"
        in verify_poses(line, 1.1, 0.4, poses, [0.9, 0.1])["issues"]
    )


def test_missing_pose_remains_invalid(corner):
    line, poses = corner
    assert not verify_poses(line, 1.1, 0.4, poses[:-1], [0.9, 0.1])["valid"]
