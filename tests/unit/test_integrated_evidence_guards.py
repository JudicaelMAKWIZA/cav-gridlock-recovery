"""Une preuve incohérente ne devient pas actuelle ou confirmée grâce au cache."""

import copy
import math

import pytest

from cav_recovery.communication.reports import (
    DiagnosticExchange,
    ReportError,
    make_report,
    validate_report,
)
from cav_recovery.crdg import LocalCRDG
from cav_recovery.d01.geometry import GeometryError, MovementGeometry
from cav_recovery.d01.inputs import adapt
from cav_recovery.d01.observer import EgoObserver
from tests.support.d01_fixtures import fixture_a, packet


def test_cached_plan_is_rechecked_when_own_steering_limit_changes():
    row = packet()
    row["map"]["drivable_polygons_xy_m"] = [
        [[-20, -10], [40, -10], [40, 10], [-20, 10]]
    ]
    poses = [[0.0, 0.0, 0.0]]
    for _ in range(25):
        x, y, heading = poses[-1]
        rotation = 0.0125
        poses.append(
            [
                x + 0.25 * math.cos(heading + rotation / 2),
                y + 0.25 * math.sin(heading + rotation / 2),
                heading + rotation,
            ]
        )
    row["own_controls"].update(
        planned_path_xyh_m=poses, wheelbase_m=2.7, max_steering_rad=0.7
    )
    geometry = MovementGeometry()
    geometry.build(adapt(row).data, 1.2)
    row["own_controls"]["max_steering_rad"] = 0.01
    with pytest.raises(GeometryError, match="OWN_PLAN_STEERING_LIMIT"):
        geometry.build(adapt(row).data, 1.2)


@pytest.mark.parametrize(
    "mutation",
    ["missing_appointment", "invalid_effect_windows", "shortened_persistence"],
)
def test_confirmation_report_requires_the_complete_k04_evidence(mutation):
    observer = EgoObserver()
    for row in fixture_a():
        diagnostic = observer.update(row)
    report = make_report(diagnostic)
    if mutation == "missing_appointment":
        report["persistence"]["proof_appointments"] -= 1
    elif mutation == "invalid_effect_windows":
        report["persistence"]["all_effect_windows_qualify"] = False
    else:
        report["persistence"]["start_s"] += 0.2
    with pytest.raises(ReportError):
        validate_report(report, "V0", 30.4)


def test_anonymous_error_bound_cannot_be_negative():
    report = make_report(EgoObserver().update(packet()))
    report["anonymous_obstacles"] = [
        dict(
            ray=0,
            position_xy_m=[2.0, 0.0],
            observed_at_s=0.4,
            error_bound_m=-1.0,
            category="unknown",
            identity=None,
        )
    ]
    with pytest.raises(ReportError):
        validate_report(report, "V0", 0.4)


@pytest.mark.parametrize(
    "mutation", ["observation_mismatch", "invalid_position", "boolean_sequence"]
)
def test_received_report_envelope_matches_its_evidence(mutation):
    exchange = DiagnosticExchange()
    own = None
    for at_s in (0.2, 0.4):
        readings = {}
        for ego, x in (("V0", 0.0), ("V1", 10.0)):
            row = packet(at_s, sender="peer-" + ego)
            row["ego_id"] = ego
            diagnostic = EgoObserver().update(row)
            if ego == "V0":
                own = diagnostic
            readings[ego] = dict(position_xy_m=[x, 0.0], report=make_report(diagnostic))
        exchange.observe(at_s, readings)
    received = copy.deepcopy(exchange.knowledge("V0"))
    message = received["received"]["V1"]["message"]
    if mutation == "observation_mismatch":
        message["observed_at_s"] = 0.2
    elif mutation == "invalid_position":
        message["position_xy_m"] = [10.0, 0.0, 999.0]
    else:
        message["sequence_number"] = True
    with pytest.raises(ReportError):
        LocalCRDG("V0").update(own, received, 0.4)
