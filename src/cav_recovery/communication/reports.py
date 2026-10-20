"""Rapports locaux de diagnostic transportés sans perte ni latence réseau ajoutée."""

import copy
import math

from .v2v import V2VExchange

STATUSES = {"NO_ALERT", "SUSPECTED", "CONFIRMED_OBSTRUCTION", "UNKNOWN"}
REPORT_FIELDS = {
    "schema_version",
    "ego_id",
    "movement_id",
    "destination",
    "status",
    "reason",
    "evidence_through_s",
    "emitted_at_s",
    "temporal_scope",
    "own_effect",
    "persistence",
    "relations",
    "anonymous_obstacles",
}
RELATION_FIELDS = {
    "requester",
    "blocker",
    "movement_id",
    "observed_at_s",
    "level",
    "resource_wkt",
    "message_id",
    "source",
    "error_bound_m",
    "effect_qualifies",
}
EFFECT_FIELDS = {
    "start_s",
    "end_s",
    "samples",
    "progress_m",
    "low_speed_fraction",
    "qualifies",
}
PERSISTENCE_FIELDS = {
    "start_s",
    "end_s",
    "proof_appointments",
    "own_effect_windows",
    "first_effect_end_s",
    "last_effect_end_s",
    "all_effect_windows_qualify",
}


class ReportError(ValueError):
    """Rapport non conforme ou incohérent avec son émetteur."""


def finite(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def validate_report(report, sender, at_s):
    """Refuse les champs opaques et conserve la date de preuve d'origine."""
    if not isinstance(report, dict) or set(report) != REPORT_FIELDS:
        raise ReportError("invalid diagnostic report fields")
    if report["schema_version"] != "cgr-diagnostic-1" or report["ego_id"] != sender:
        raise ReportError("invalid diagnostic sender")
    if not isinstance(report["status"], str) or report["status"] not in STATUSES:
        raise ReportError("invalid diagnostic status")
    for key in ("ego_id", "movement_id", "destination", "reason"):
        if not isinstance(report[key], str) or not 0 < len(report[key]) <= 256:
            raise ReportError("invalid diagnostic identifier")
    observed, emitted = report["evidence_through_s"], report["emitted_at_s"]
    if (
        not all(finite(t) for t in (observed, emitted, at_s))
        or not observed <= emitted <= at_s + 1e-7
    ):
        raise ReportError("future or malformed diagnostic times")
    for name, fields, boolean in (
        ("own_effect", EFFECT_FIELDS, "qualifies"),
        ("persistence", PERSISTENCE_FIELDS, "all_effect_windows_qualify"),
    ):
        block = report[name]
        if block is not None:
            if (
                not isinstance(block, dict)
                or set(block) != fields
                or type(block[boolean]) is not bool
            ):
                raise ReportError("invalid diagnostic evidence fields")
            if not all(finite(block[key]) for key in fields - {boolean}):
                raise ReportError("invalid diagnostic evidence numbers")
            if block["start_s"] > block["end_s"] or block["end_s"] > observed + 1e-7:
                raise ReportError("future diagnostic evidence")
    if (
        not isinstance(report["anonymous_obstacles"], list)
        or len(report["anonymous_obstacles"]) > 16
    ):
        raise ReportError("invalid anonymous obstacles")
    for obstacle in report["anonymous_obstacles"]:
        if not isinstance(obstacle, dict) or set(obstacle) != {
            "ray",
            "position_xy_m",
            "observed_at_s",
            "error_bound_m",
            "category",
            "identity",
        }:
            raise ReportError("invalid anonymous obstacle fields")
        if obstacle["identity"] is not None or obstacle["category"] != "unknown":
            raise ReportError("anonymous evidence cannot name a vehicle")
        if type(obstacle["ray"]) is not int or not 0 <= obstacle["ray"] < 16:
            raise ReportError("invalid anonymous ray")
        if (
            not isinstance(obstacle["position_xy_m"], list)
            or len(obstacle["position_xy_m"]) != 2
            or not all(finite(v) for v in obstacle["position_xy_m"])
        ):
            raise ReportError("invalid anonymous position")
        if (
            not finite(obstacle["observed_at_s"])
            or abs(obstacle["observed_at_s"] - observed) > 1e-7
        ):
            raise ReportError("invalid anonymous observation time")
    if report["temporal_scope"] != "sampled_0.4s":
        raise ReportError("unsupported diagnostic scope")
    if not isinstance(report["relations"], list) or len(report["relations"]) > 64:
        raise ReportError("invalid relations")
    for relation in report["relations"]:
        if not isinstance(relation, dict) or set(relation) != RELATION_FIELDS:
            raise ReportError("invalid relation fields")
        if (
            relation["requester"] != sender
            or relation["movement_id"] != report["movement_id"]
        ):
            raise ReportError("relation does not belong to sender")
        if relation["blocker"] == sender or not isinstance(relation["blocker"], str):
            raise ReportError("invalid blocker")
        if (
            not finite(relation["observed_at_s"])
            or not finite(relation["error_bound_m"])
            or relation["error_bound_m"] < 0
        ):
            raise ReportError("invalid relation measurement")
        if type(relation["effect_qualifies"]) is not bool:
            raise ReportError("invalid effect flag")
        if abs(relation["observed_at_s"] - observed) > 1e-7:
            raise ReportError("relation time does not match its evidence")
        if not isinstance(relation["level"], str) or relation["level"] not in {
            "CANDIDATE",
            "EFFECT_SUPPORTED",
            "CONFIRMED",
        }:
            raise ReportError("invalid relation level")
        if (
            not isinstance(relation["resource_wkt"], str)
            or len(relation["resource_wkt"]) > 50000
        ):
            raise ReportError("invalid resource geometry")
        if relation["level"] == "CONFIRMED" and (
            report["status"] != "CONFIRMED_OBSTRUCTION"
            or report["persistence"] is None
            or not relation["effect_qualifies"]
        ):
            raise ReportError("confirmation without persistence")
    return report


def make_report(diagnostic):
    """Ne transmet que le diagnostic propre et les preuves qui lui sont accessibles."""
    effect = diagnostic["effect_window"]
    supported = bool(effect and effect["qualifies"])
    confirmed = diagnostic["status"] == "CONFIRMED_OBSTRUCTION"
    relations = [
        dict(
            requester=diagnostic["ego_id"],
            blocker=row["sender_id"],
            movement_id=diagnostic["movement_id"],
            observed_at_s=row["observed_at_s"],
            level="CONFIRMED"
            if confirmed
            else "EFFECT_SUPPORTED"
            if supported
            else "CANDIDATE",
            resource_wkt=row["resource_polygon_wkt"],
            message_id=row["message_id"],
            source=row["source"],
            error_bound_m=row["error_bound_m"],
            effect_qualifies=supported,
        )
        for row in diagnostic["current_occupants"]
    ]
    report = dict(
        schema_version="cgr-diagnostic-1",
        ego_id=diagnostic["ego_id"],
        movement_id=diagnostic["movement_id"],
        destination=diagnostic["destination"],
        status=diagnostic["status"],
        reason=diagnostic["reason"],
        evidence_through_s=diagnostic["evidence_through_s"],
        emitted_at_s=diagnostic["emitted_at_s"],
        temporal_scope="sampled_0.4s",
        own_effect=copy.deepcopy(effect),
        persistence=copy.deepcopy(diagnostic["coverage"]["persistence_window"]),
        relations=relations,
        anonymous_obstacles=copy.deepcopy(diagnostic["anonymous_obstacles"]),
    )
    return validate_report(report, report["ego_id"], report["emitted_at_s"])


class DiagnosticExchange(V2VExchange):
    """Même portée et cadence que la télémétrie ; chaque envoi vient du CAV émetteur."""

    def __init__(self, **kwargs):
        options = dict(range_m=30.0, interval_s=0.4, lifetime_s=1.2, step_s=0.2)
        options.update(kwargs)
        super().__init__(**options)

    def _own_state(self, item, row, time_s):
        report = validate_report(row["report"], item, time_s)
        if abs(report["emitted_at_s"] - time_s) > 1e-7:
            raise ReportError("sender report is not from this decision")
        return dict(
            position_xy_m=list(row["position_xy_m"]), report=copy.deepcopy(report)
        )
