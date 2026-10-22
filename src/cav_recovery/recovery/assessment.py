"""Appariement explicite des variantes d'activation, sans changer REC-EVAL-1."""

from pathlib import Path

from ..experiments.evaluation import active_events
from ..laboratory.storage import atomic_json, json_file, safe_destination
from .evaluation import physical_outcome, rows


def paired_assessment(nominal, intervention, output):
    """Autorise seulement le nom de phase et le délai local comme différences.

    Une intervention avant la confirmation physique conserve le statut
    NO_QUALIFIED_INCIDENT. Elle ne devient pas un succès de récupération.
    Les fenêtres de progression et de rechute restent celles de REC-EVAL-1.
    """
    nominal, intervention, output = map(Path, (nominal, intervention, output))
    for folder in (nominal, intervention):
        safe_destination(folder, output)
    differences = {}
    for name, allowed in [("spec.json", {"phase"}), ("contract.json", {"stall_s"})]:
        first, second = json_file(nominal / name), json_file(intervention / name)
        changed = {
            key: [first.get(key), second.get(key)]
            for key in first.keys() | second.keys()
            if first.get(key) != second.get(key)
        }
        if changed.keys() - allowed:
            raise ValueError("Conditions physiques non appariées : " + name)
        differences[name] = changed
    commands = list(rows(intervention / "commands.jsonl.gz"))
    active = [
        r["at_s"]
        for r in commands
        if r["applied"]["reason"] not in {"own_route", "nominal", "guard_hold"}
        and r["guard"]["accepted"]
    ]
    at = min(active) if active else json_file(nominal / "summary.json")["at_s"]
    reference = json_file(nominal / "physical-annotation.json")
    events = active_events(reference, at)
    affected = {event["requester"] for event in events}
    origin = min((e["start_s"] for e in events), default=None)
    witnesses = [
        (e["requester"], interval["start_s"])
        for e in reference["events"]
        for interval in e["effect_intervals"]
        if interval["start_s"] < at and interval["end_s"] >= at
    ]
    frames = {
        name: list(rows(folder / "reference.jsonl.gz"))
        for name, folder in [("nominal", nominal), ("intervention", intervention)]
    }
    prefix = {
        name: [frame for frame in values if frame["time_s"] <= at + 1e-7]
        for name, values in frames.items()
    }
    result = dict(
        protocol="REC-EVAL-1-explicit-activation-pairing",
        development_only=True,
        permitted_configuration_differences=differences,
        t_act_s=min(active) if active else None,
        t_phys_s=origin,
        t_diag_s=None,
        autonomous_K04_detection_not_claimed=True,
        physical_effects_present_before_action=witnesses,
        qualified_incident_present_before_action=bool(affected),
        identical_physical_reference_until_action=prefix["nominal"]
        == prefix["intervention"],
    )
    if not result["identical_physical_reference_until_action"]:
        raise ValueError("Les traces divergent avant la première intervention.")
    for name, folder in [("nominal", nominal), ("intervention", intervention)]:
        summary = json_file(folder / "summary.json")
        outcome = physical_outcome(
            frames[name],
            json_file(folder / "physical-annotation.json"),
            json_file(folder / "metadata.json"),
            summary["arrivals"],
            affected,
            at,
            json_file(folder / "contract.json"),
        )
        if summary["errors"]:
            outcome.update(status="INTEGRITY_FAILURE", success=False)
        confirmed = outcome["t_conf_s"]
        outcome.update(
            T_recovery_s=None,
            T_action_s=confirmed - at if confirmed is not None else None,
            T_end_to_end_s=confirmed - origin
            if confirmed is not None and origin is not None
            else None,
        )
        result[name] = outcome
    result["improvement_not_spontaneous"] = (
        result["intervention"]["success"] and not result["nominal"]["success"]
    )
    atomic_json(output, result)
    return result
