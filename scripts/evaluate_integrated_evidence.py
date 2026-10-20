"""Complète les mesures hors ligne sans transmettre la référence aux CAV."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import networkx as nx

from cav_recovery.dataset.pipeline import read_rows
from cav_recovery.dataset.contract import Contract
from cav_recovery.experiments.evaluation import active_events, metrics, json_file
from cav_recovery.experiments.temporal import causal_windows
from cav_recovery.experiments.session import percentiles
from cav_recovery.communication.reports import make_report, validate_report


def indirect_pairs(pairs):
    """Réserve les dépendances indirectes aux destinations sans lien direct établi."""
    graph = nx.DiGraph(pairs)
    result = set()
    for source in graph:
        for target, distance in nx.single_source_shortest_path_length(
            graph, source
        ).items():
            if distance >= 2 and not graph.has_edge(source, target):
                result.add((source, target))
    return result


def audit_episode(folder):
    """Compare les mêmes instants et garde les preuves globales hors des entrées."""
    result = json.loads((folder / "evaluation/results.json").read_text("utf-8"))
    reference = json.loads(
        (folder / "evaluation/physical-reference.json").read_text("utf-8")
    )["annotation"]
    frames = {
        row["time_s"]: row
        for row in read_rows(folder / "evaluation/physical-reference-frames.jsonl.gz")
    }
    inputs = {
        row["at_s"]: row["measurements"]
        for row in read_rows(folder / "diagnostic/inputs.jsonl.gz")
    }
    decisions = {
        row["at_s"]: row["vehicles"]
        for row in read_rows(folder / "diagnostic/decisions.jsonl.gz")
    }
    reports = {}
    deliveries_checked = 0
    reports_checked = 0
    errors = []
    physical_parameters = {}
    for measurements in inputs.values():
        for ego, local in measurements.items():
            parameters = tuple(
                local.get("own_controls", {}).get(key)
                for key in ("wheelbase_m", "max_steering_rad")
            )
            if ego in physical_parameters and physical_parameters[ego] != parameters:
                errors.append("vehicle physical parameters changed within an episode")
            physical_parameters[ego] = parameters
    for event in read_rows(folder / "diagnostic/communications.jsonl.gz"):
        at_s = event["time_s"]
        message = event["message"]
        sender = message["sender_id"]
        position = inputs[at_s][sender]["ego"]["position_xy_m"]
        expected = {
            key
            for key, local in inputs[at_s].items()
            if key != sender
            and math.dist(position, local["ego"]["position_xy_m"]) <= 30.0
        }
        delivered = set(event["delivered_to"])
        # Le stockage des poses est arrondi au micromètre ; la portée physique ne l'est pas.
        disputed = expected ^ delivered
        if any(
            abs(math.dist(position, inputs[at_s][key]["ego"]["position_xy_m"]) - 30.0)
            > 2e-6
            for key in disputed
        ):
            errors.append("incorrect direct delivery range")
        deliveries_checked += len(delivered)
        if event["channel"] == "diagnostic":
            try:
                validate_report(message["report"], sender, at_s)
                if message["report"] != make_report(decisions[at_s][sender]["B"]):
                    errors.append("recorded report differs from its own diagnostic")
            except ValueError as error:
                errors.append("invalid recorded diagnostic report: " + str(error))
            reports_checked += 1
            reports.setdefault(at_s, {})[sender] = delivered
    counts = {name: Counter() for name in ("indirect_global", "indirect_observable")}
    distributions = {
        name: []
        for name in (
            "active",
            "central",
            "approach",
            "affected",
            "affected_central",
            "affected_approach",
        )
    }
    first_groups = {name: None for name in ("reference", "B", "C")}
    premature = Counter()
    false_confirmations = []
    timeline = {
        row["at_s"]: row["groups"]
        for row in json.loads(
            (folder / "evaluation/collective-reference.json").read_text("utf-8")
        )
    }
    for at_s, vehicles in decisions.items():
        if abs(at_s / 0.4 - round(at_s / 0.4)) > 1e-7:
            continue
        active = {event["requester"] for event in active_events(reference, at_s)}
        states = frames[at_s]["states"]
        for name, values in (
            ("active", set(states)),
            (
                "central",
                {
                    key
                    for key, state in states.items()
                    if state["resource_id"].endswith(":central")
                },
            ),
            (
                "approach",
                {
                    key
                    for key, state in states.items()
                    if state["resource_id"].endswith(":approach")
                },
            ),
            ("affected", active),
            (
                "affected_central",
                {
                    key
                    for key in active
                    if states[key]["resource_id"].endswith(":central")
                },
            ),
            (
                "affected_approach",
                {
                    key
                    for key in active
                    if states[key]["resource_id"].endswith(":approach")
                },
            ),
        ):
            distributions[name].append(len(values))
        edges = {
            (row["requester"], row["blocker"])
            for row in frames[at_s]["witnesses"]
            if row["requester"] in active and row["road_admissible"]
        }
        global_indirect = indirect_pairs(edges)
        groups = timeline.get(at_s, [])
        if groups and first_groups["reference"] is None:
            first_groups["reference"] = at_s
        for ego, row in vehicles.items():
            observable = {
                edge
                for edge in edges
                if (edge[0] == ego or ego in reports.get(at_s, {}).get(edge[0], []))
                and any(
                    entry["message"]["sender_id"] == edge[1]
                    and abs(entry["message"]["observed_at_s"] - at_s) <= 1e-7
                    for entry in (inputs[at_s].get(edge[0], {}).get("v2v") or {}).get(
                        "received", []
                    )
                )
            }
            predicted = {
                (path["requester"], path["blocker"])
                for path in row["C"]["indirect_dependencies"]
            }
            for name, target in (
                ("indirect_global", global_indirect),
                ("indirect_observable", indirect_pairs(observable)),
            ):
                counts[name]["tp"] += len(predicted & target)
                counts[name]["fp"] += len(predicted - target)
                counts[name]["fn"] += len(target - predicted)
            if row["B"]["status"] == "CONFIRMED_OBSTRUCTION" and ego not in active:
                future = [
                    event["confirmation_at_s"]
                    for event in reference["events"]
                    if event["confirmed"]
                    and event["requester"] == ego
                    and event["start_s"] <= at_s < event["confirmation_at_s"]
                ]
                premature[
                    "before_reference_persistence"
                    if future
                    else "other_false_confirmation"
                ] += 1
                areas = [
                    proof["intersection_area_m2"]
                    for proof in row["B"]["current_occupants"]
                ]
                false_confirmations.append(
                    dict(
                        ego_id=ego,
                        at_s=at_s,
                        areas_m2=areas,
                        all_areas_below_reference_cutoff=bool(areas)
                        and max(areas) <= Contract().minimum_intersection_area_m2,
                        interpretation="D01_requires_positive_robust_area; M02_reference_requires_area_above_its_frozen_cutoff",
                    )
                )
            for mode, predicted_groups in (
                (
                    "B",
                    [row["C"]["simple_count_requesters"]]
                    if row["C"]["simple_count_baseline"]
                    else [],
                ),
                ("C", [group["affected_requesters"] for group in row["C"]["groups"]]),
            ):
                if first_groups[mode] is None and any(
                    set(group) <= set(target)
                    for group in predicted_groups
                    for target in groups
                ):
                    first_groups[mode] = at_s
    feature_count = 0
    stream = read_rows(folder / "diagnostic/features.jsonl.gz")
    for window in causal_windows(stream):
        row = window["samples"][-1]
        local = inputs[row["observed_at_s"]][row["ego_id"]]
        if dict(row["own_state"]) != local["ego"] or tuple(row["lidar"]) != tuple(
            local["lidar"]["normalized_ranges"]
        ):
            # Les listes sont gelées en tuples par l'interface causale.
            from cav_recovery.d01.inputs import plain

            if (
                plain(row["own_state"]) != local["ego"]
                or plain(row["lidar"]) != local["lidar"]["normalized_ranges"]
            ):
                errors.append("feature differs from its own recorded measurement")
        if row["available_at_s"] > row["observed_at_s"] + 1e-7:
            errors.append("future feature")
        feature_count += 1
    origin = first_groups["reference"]
    recovered_effects = []
    for release in result["observed_releases_with_resumed_progress"]:
        event = next(
            event
            for event in reference["events"]
            if event["requester"] == release["requester"]
            and abs(event["end_s"] - release["released_at_s"]) <= 1e-7
        )
        recovered_effects.append(
            dict(
                release,
                physical_effect_s=event["movement_effect_duration_s"],
                effect_established_5s=bool(event["effect_intervals"]),
            )
        )
    return dict(
        name=folder.name,
        topology=result["topology"],
        reference_label=result["reference_label"],
        distributions_on_proof_appointments={
            name: percentiles(values) for name, values in distributions.items()
        },
        distribution_histograms={
            name: dict(Counter(values)) for name, values in distributions.items()
        },
        individual_episode_metrics=result["metrics"]["B"],
        releases_with_resumed_progress=recovered_effects,
        indirect_metrics={name: metrics(value) for name, value in counts.items()},
        first_collective_at_s=first_groups,
        collective_delay_s={
            mode: first_groups[mode] - origin
            if origin is not None and first_groups[mode] is not None
            else None
            for mode in ("B", "C")
        },
        premature_confirmations=dict(premature),
        false_confirmation_details=false_confirmations,
        checked_feature_rows=feature_count,
        checked_actual_deliveries=deliveries_checked,
        checked_diagnostic_reports=reports_checked,
        checked_vehicle_physical_parameters=len(physical_parameters),
        causal_export_and_range_errors=errors[:20],
        error_count=len(errors),
    )


def aggregate_episodes(episodes):
    """Regroupe les audits déjà vérifiés, sans compter les images comme des épisodes."""
    totals = {name: Counter() for name in ("indirect_global", "indirect_observable")}
    for episode in episodes:
        for name, values in episode["indirect_metrics"].items():
            totals[name].update(
                {
                    key: value
                    for key, value in values.items()
                    if key in {"tp", "fp", "fn"}
                }
            )
    summary = dict(
        episodes=episodes,
        indirect_metrics={name: metrics(values) for name, values in totals.items()},
        error_count=sum(row["error_count"] for row in episodes),
        algorithm_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="offline_only_never_used_as_CAV_inputs",
    )
    strata = defaultdict(list)
    for episode in episodes:
        strata["all"].append(episode)
        strata[episode["reference_label"] + "/" + episode["topology"]].append(episode)
    summary["by_class_and_topology"] = {}
    for label, rows in strata.items():
        distributions = {}
        for name in rows[0]["distribution_histograms"]:
            histogram = Counter()
            for row in rows:
                histogram.update(
                    {
                        int(key): value
                        for key, value in row["distribution_histograms"][name].items()
                    }
                )
            distributions[name] = dict(
                histogram=dict(sorted(histogram.items())),
                **percentiles(
                    [value for value, count in histogram.items() for _ in range(count)]
                ),
            )
        episode_metrics = {}
        for name in ("precision", "recall", "f1"):
            values = [
                row["individual_episode_metrics"][name]
                for row in rows
                if row["individual_episode_metrics"][name] is not None
            ]
            episode_metrics[name] = dict(
                mean=sum(values) / len(values) if values else None,
                defined_episodes=len(values),
                total_episodes=len(rows),
            )
        summary["by_class_and_topology"][label] = dict(
            episodes=len(rows),
            distributions=distributions,
            individual_episode_metrics=episode_metrics,
            statistical_unit="episode_for_macro_scores; proof_appointments_for_density_distributions",
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protected = Path(r"D:\cav-recovery-data\metadrive-dense-20261018").resolve()
    if args.output_dir.resolve().is_relative_to(protected):
        raise ValueError("the frozen M02 corpus cannot receive experimental audits")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    for root in args.experiments:
        for manifest in sorted(root.glob("*/complete.json")):
            result = audit_episode(manifest.parent)
            episodes.append(result)
            json_file(
                args.output_dir / "progress.json",
                dict(completed=len(episodes), last=result["name"]),
            )
    summary = aggregate_episodes(episodes)
    json_file(args.output_dir / "additional-metrics.json", summary)
    return int(bool(summary["error_count"]))


if __name__ == "__main__":
    raise SystemExit(main())
