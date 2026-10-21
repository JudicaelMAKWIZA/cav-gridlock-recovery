"""Mesures hors ligne : jamais un graphe vide comme preuve de récupération."""

from pathlib import Path
import gzip
import json

from ..experiments.evaluation import active_events
from ..laboratory.storage import atomic_json, json_file, safe_destination


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def physical_outcome(frames, annotation, metadata, arrivals, affected, at_s, contract):
    """Population fixée hors acteur, progression nette et suivi des rechutes."""
    if not affected:
        return dict(
            status="NO_QUALIFIED_INCIDENT", success=False, affected=[], t_conf_s=None
        )
    times = [f["time_s"] for f in frames]
    origin = next((f for f in frames if f["time_s"] >= at_s - 1e-7), None)
    if origin is None or any(k not in origin["states"] for k in affected):
        return dict(
            status="INSUFFICIENT_INITIAL_MEASUREMENT",
            success=False,
            affected=sorted(affected),
            t_conf_s=None,
        )
    initial = {k: origin["states"][k]["progress_m"] for k in affected}
    last_positions = dict(initial)
    candidates = []
    progress = []
    effects = [
        (e["requester"], i["start_s"], i["end_s"])
        for e in annotation["events"]
        for i in e["effect_intervals"]
    ]
    window = contract["recovery_window_s"]
    followup = contract["relapse_followup_s"]
    effect_s = contract["effect_window_s"]
    for index, frame in enumerate(frames):
        t = frame["time_s"]
        for ego, state in frame["states"].items():
            if ego in affected:
                last_positions[ego] = state["progress_m"]
        if t < at_s:
            continue
        released = {
            ego
            for ego in affected
            if (ego in arrivals and arrivals[ego] <= t)
            or last_positions[ego] - initial[ego]
            >= metadata["vehicles"][ego]["length_m"]
        }
        progress.append(
            dict(
                at_s=t,
                released=sorted(released),
                fraction=len(released) / len(affected),
            )
        )
        if t < at_s + window - 1e-7 or len(released) < len(affected):
            continue
        begin = t - window
        earlier = next((f for f in frames if abs(f["time_s"] - begin) < 1e-7), None)
        if earlier is None:
            continue
        enough = all(
            ego in arrivals
            and arrivals[ego] <= t
            or ego in frame["states"]
            and ego in earlier["states"]
            and frame["states"][ego]["progress_m"]
            - earlier["states"][ego]["progress_m"]
            >= metadata["vehicles"][ego]["length_m"]
            for ego in affected
        )
        impeded = any(
            ego in affected and min(end, t) - max(start, begin) >= effect_s - 1e-7
            for ego, start, end in effects
        )
        if enough and not impeded:
            candidates.append(t)
    t_conf = candidates[0] if candidates else None
    relapse = t_conf is not None and any(
        ego in affected
        and min(end, t_conf + followup) - max(start, t_conf) >= effect_s - 1e-7
        for ego, start, end in effects
    )
    observed = t_conf is not None and (
        times[-1] >= t_conf + followup - 1e-7
        or all(ego in arrivals for ego in affected)
    )
    success = t_conf is not None and observed and not relapse
    validation_at = None
    if observed:
        validation_at = t_conf + followup
        if all(ego in arrivals for ego in affected):
            validation_at = min(
                validation_at, max(t_conf, max(arrivals[k] for k in affected))
            )
    return dict(
        status="DURABLE_RECOVERY"
        if success
        else "RELAPSE"
        if relapse
        else "RIGHT_CENSORED_FOLLOWUP"
        if t_conf is not None and not observed
        else "NOT_RECOVERED",
        success=success,
        affected=sorted(affected),
        t_conf_s=t_conf,
        t_final_validation_s=validation_at,
        relapse=relapse,
        best_released_fraction=max((p["fraction"] for p in progress), default=0.0),
        timeline=progress,
    )


def compare(nominal, intervention, output):
    """Contrefactuel de développement apparié, pas une évaluation indépendante."""
    nominal, intervention, output = Path(nominal), Path(intervention), Path(output)
    safe_destination(nominal, output)
    safe_destination(intervention, output)
    if json_file(nominal / "spec.json") != json_file(
        intervention / "spec.json"
    ) or json_file(nominal / "contract.json") != json_file(
        intervention / "contract.json"
    ):
        raise ValueError("Configurations non appariées.")
    n = json_file(nominal / "physical-annotation.json")
    commands = list(rows(intervention / "commands.jsonl.gz"))
    proposed = [
        r["at_s"]
        for r in commands
        if r["proposed"]["reason"] not in {"own_route", "nominal"}
    ]
    applied = [
        r["at_s"]
        for r in commands
        if r["applied"]["reason"] not in {"own_route", "nominal", "guard_hold"}
        and r["guard"]["accepted"]
    ]
    t_act = min(applied) if applied else None
    # Si la baseline n'agit pas, l'évaluation reste négative, sans choisir un
    # instant oracle qui aurait favorisé artificiellement une tentative.
    at = t_act if t_act is not None else json_file(nominal / "summary.json")["at_s"]
    relevant = active_events(n, at)
    affected = {e["requester"] for e in relevant}
    origin = min((e["start_s"] for e in relevant), default=None)
    result = dict(
        protocol="REC-EVAL-1",
        development_only=True,
        t_proposition_s=min(proposed) if proposed else None,
        t_act_s=t_act,
        t_diag_s=None,
        t_phys_s=origin,
        associated_population_source="independent_nominal_reference_at_first_applied_recovery_command",
        autonomous_K04_detection_not_claimed=True,
    )
    for name, folder in [("nominal", nominal), ("intervention", intervention)]:
        summary = json_file(folder / "summary.json")
        outcome = physical_outcome(
            list(rows(folder / "reference.jsonl.gz")),
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
            T_action_s=confirmed - t_act
            if confirmed is not None and t_act is not None
            else None,
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
