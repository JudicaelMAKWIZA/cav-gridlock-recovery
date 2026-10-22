"""Audit hors ligne des contributions ; aucune commande n'est envoyée au moteur."""

from collections import Counter, defaultdict
from pathlib import Path
import math

from ..experiments.evaluation import active_events
from ..laboratory.storage import json_file
from .evaluation import rows


def detector_audit(folder):
    """Rendez-vous corrélés et événements, sans les assimiler à des essais."""
    folder = Path(folder)
    annotation = json_file(folder / "physical-annotation.json")
    observations = list(rows(folder / "causal-observations.jsonl.gz"))
    counters = {mode: Counter() for mode in ("raw", "metro", "tcn")}
    reasons = {mode: Counter() for mode in ("raw", "metro")}
    first = {mode: {} for mode in counters}
    intervals = defaultdict(list)
    for event in annotation["events"]:
        for interval in event["effect_intervals"]:
            if interval["confirmed"]:
                intervals[event["requester"]].append(
                    (interval["confirmation_at_s"], interval["end_s"])
                )
    costs = []
    for row in observations:
        at, ego = row["at_s"], row["ego"]
        truth = any(start - 1e-7 <= at <= end + 1e-7 for start, end in intervals[ego])
        for mode in counters:
            if mode == "tcn":
                status = row["temporal"]["status"] if row["temporal"] else "UNAVAILABLE"
                positive = status == "STATISTICAL_ALERT"
                unknown = status in {"UNAVAILABLE", "ABSTAIN_INSUFFICIENT_HISTORY"}
            else:
                status = row[mode + "_status"]
                positive = status == "CONFIRMED_OBSTRUCTION"
                unknown = status == "UNKNOWN"
                reasons[mode][row[mode + "_reason"]] += 1
            counters[mode]["observations"] += 1
            counters[mode]["unknown"] += int(unknown)
            counters[mode]["truth_positive"] += int(truth)
            counters[mode][
                "tp"
                if truth and positive
                else "fn"
                if truth
                else "fp"
                if positive
                else "tn"
            ] += 1
            if positive:
                first[mode].setdefault(ego, at)
        if row["temporal_wall_s"] is not None:
            costs.append(row["temporal_wall_s"])
    events = []
    for ego, spans in intervals.items():
        for start, end in spans:
            event = dict(ego=ego, physical_confirmation_s=start, end_s=end)
            for mode in counters:
                aligned = [
                    row["at_s"]
                    for row in observations
                    if row["ego"] == ego
                    and start - 1e-7 <= row["at_s"] <= end + 1e-7
                    and (
                        row["temporal"] is not None
                        and row["temporal"]["status"] == "STATISTICAL_ALERT"
                        if mode == "tcn"
                        else row[mode + "_status"] == "CONFIRMED_OBSTRUCTION"
                    )
                ]
                event[mode + "_first_aligned_s"] = min(aligned) if aligned else None
            events.append(event)
    return dict(
        counts={mode: dict(counter) for mode, counter in counters.items()},
        reasons={mode: dict(counter) for mode, counter in reasons.items()},
        first_alerts=first,
        physical_events=events,
        temporal_inference=dict(
            calls=len(costs),
            wall_s=sum(costs),
            mean_ms=1000 * sum(costs) / len(costs) if costs else None,
            maximum_ms=1000 * max(costs) if costs else None,
        ),
        population="correlated_local_appointments_not_independent_trials",
        target="physical_persistence_confirmation_available_now",
    )


def graph_audit(folder):
    """Arêtes confirmées présentes/reçues versus référence physique indépendante."""
    folder = Path(folder)
    annotation = json_file(folder / "physical-annotation.json")
    counters = Counter()
    per_ego = defaultdict(Counter)
    frames = list(rows(folder / "diagnostics.jsonl.gz"))
    eligible = {
        (row["ego"], row["at_s"])
        for row in rows(folder / "causal-observations.jsonl.gz")
    }
    truth_by_time = {}
    for frame in frames:
        if (frame["ego"], frame["at_s"]) not in eligible:
            continue
        at = frame["at_s"]
        if at not in truth_by_time:
            truth_by_time[at] = {
                (event["requester"], blocker)
                for event in active_events(annotation, at)
                for interval in event["occupant_intervals"]
                if interval["start_s"] - 1e-7 <= at <= interval["end_s"] + 1e-7
                for blocker in [interval["blocker"]]
            }
        truth = truth_by_time[at]
        valid = {
            (e["requester"], e["blocker"])
            for e in frame["graph"]["edges"]
            if e["level"] == "CONFIRMED"
            and e["observed_at_s"] <= at + 1e-7
            and e["valid_until_s"] >= at - 1e-7
        }
        current = {
            (e["requester"], e["blocker"])
            for e in frame["graph"]["edges"]
            if e["level"] == "CONFIRMED" and e["state"] == "CURRENT"
        }
        own_missing = sum(
            e["requester"] == frame["ego"] for e in active_events(annotation, at)
        )
        record = Counter(
            valid_tp=len(valid & truth),
            valid_fp=len(valid - truth),
            valid_fn=len(truth - valid),
            current_tp=len(current & truth),
            current_fn=len(truth - current),
            appointments=1,
            own_qualifying_incidents=own_missing,
            exact_nonempty_pair_set=int(bool(truth) and valid == truth),
            global_incident_appointments=int(bool(truth)),
        )
        counters.update(record)
        per_ego[frame["ego"]].update(record)
    return dict(
        counts=dict(counters),
        per_ego={k: dict(v) for k, v in per_ego.items()},
        scope="pair_matching_not_exact_resource_geometry_or_causal_manoeuvre_feasibility",
        note="L'absence d'arêtes distantes peut être légitime pour une vue locale ; "
        "ces mesures décrivent la couverture globale, pas une obligation d'omniscience.",
    )


def communication_audit(folder):
    """Trois transports distincts : états diagnostic, rapports, négociation physique."""
    folder = Path(folder)
    counts = Counter()
    ages, far = [], 0
    for packet in rows(folder / "local.jsonl.gz"):
        for message in packet["messages"]:
            counts[message["kind"]] += 1
            ages.append(packet["at_s"] - message["sent_s"])
            if (
                message["kind"] == "INTENT"
                and math.dist(message["position"], packet["own"]["position"]) > 10
            ):
                far += 1
    return dict(
        local_mailbox_occurrences=dict(counts),
        distant_intents=far,
        max_live_age_s=max(ages, default=None),
        recovery_transport=dict(
            range_m=30, lifetime_s=0.6, simulated_delivery_delay_s=0.2
        ),
        diagnostic_transport=dict(
            range_m=30,
            interval_s=0.4,
            lifetime_s=1.2,
            simulated_additional_network_delay_s=0,
        ),
        diagnostic_report_transport="distinct mailbox; no multi-hop or infrastructure coordinator",
    )


def result_table(root):
    root = Path(root)
    records = []
    for path in sorted((root / "runs").glob("*/*/summary.json")):
        summary = json_file(path)
        case, condition = path.parent.parent.name, path.parent.name
        comparison = root / "comparisons" / (case + "-" + condition + ".json")
        outcome = json_file(comparison) if comparison.exists() else None
        records.append(
            dict(
                case=case,
                condition=condition,
                introduced=summary["introduced"],
                arrivals=len(summary["arrivals"]),
                active=len(summary["active"]),
                errors=summary["errors"],
                reverse_max_m=max(summary["reverse_m"].values(), default=0),
                minimum_gap_m=summary["minimum_gap_m"],
                observed_s=summary["at_s"],
                cpu_s=summary["cpu_s"],
                wall_s=summary["wall_s"],
                physical_label=summary["physical_label"],
                t_act_s=outcome["t_act_s"] if outcome else None,
                outcome=outcome["intervention"]["status"]
                if outcome
                else "NOMINAL_REFERENCE",
                affected=outcome["intervention"]["affected"] if outcome else [],
                recovery_success=outcome["intervention"]["success"]
                if outcome
                else None,
                nominal_spontaneous_recovery=outcome["nominal"]["success"]
                if outcome
                else None,
                incremental_gain=outcome["improvement_not_spontaneous"]
                if outcome
                else None,
                paired_prefix=outcome["identical_physical_reference_until_action"]
                if outcome
                else None,
            )
        )
    return records
