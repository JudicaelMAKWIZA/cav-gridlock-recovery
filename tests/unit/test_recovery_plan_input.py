"""Le plan propre enrichit le contrat d'entrée existant sans information voisine."""

import copy
from tests.support.d01_fixtures import packet
from cav_recovery.d01.inputs import adapt
from cav_recovery.d01.geometry import planned_poses
from cav_recovery.recovery.plan_input import OwnPlanProjector


def test_declared_plan_is_causal_detached_and_accepted_by_unchanged_k04_input():
    source = packet()
    before = copy.deepcopy(source)
    projected = OwnPlanProjector().project(
        source, wheelbase_m=2.46894, max_steering_rad=0.6981317008
    )
    assert source == before
    adapted = adapt(projected).data
    poses = planned_poses(adapted, 1.2 * source["ego"]["length_m"])
    assert poses
    assert (
        projected["own_controls"]["observed_at_s"]
        == source["own_controls"]["observed_at_s"]
    )
    assert projected["v2v"] == source["v2v"]
    assert projected["ego"] == source["ego"]
    assert projected["mission"] == source["mission"]
