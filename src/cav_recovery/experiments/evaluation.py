"""Évaluation hors ligne à partir des traces physiques, sans retour vers les CAV."""

from collections import Counter, defaultdict, deque
import gzip
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import shapely
from shapely import wkt

from ..dataset.contract import Contract
from ..dataset.pipeline import read_rows, file_sha256
from ..dataset.qualify import qualify


def json_file(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )


def rectangles(poses, length, width):
    """Implémentation vectorisée indépendante de la géométrie du détecteur."""
    poses = np.asarray(poses, dtype=float)
    corners = np.array(
        [
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
        ]
    )
    cosines, sines = np.cos(poses[:, 2]), np.sin(poses[:, 2])
    coordinates = np.empty((len(poses), 4, 2))
    coordinates[:, :, 0] = (
        poses[:, 0, None]
        + cosines[:, None] * corners[:, 0]
        - sines[:, None] * corners[:, 1]
    )
    coordinates[:, :, 1] = (
        poses[:, 1, None]
        + sines[:, None] * corners[:, 0]
        + cosines[:, None] * corners[:, 1]
    )
    return shapely.polygons(coordinates)


def physical_reference(folder):
    """Reconstruit les occupations depuis les corps mesurés et le plan déclaré.

    Le plan demandé appartient au conducteur, pas à D01. Les positions de tous
    les autres corps sont réservées à cette référence hors ligne. Les seuils
    C1–C7 M02 sont réutilisés sans modifier le corpus ou ses annotations.
    """
    target_dir = folder / "evaluation"
    cache_path = target_dir / "physical-reference.json"
    frames_path = target_dir / "physical-reference-frames.jsonl.gz"
    provenance = dict(
        inputs={
            name: file_sha256(folder / name)
            for name in (
                "physics.jsonl.gz",
                "reference.jsonl.gz",
                "metadata.json",
                "annotation.json",
                "diagnostic/inputs.jsonl.gz",
            )
        },
        algorithm=hashlib.sha256(
            (
                inspect.getsource(physical_reference)
                + inspect.getsource(rectangles)
                + inspect.getsource(qualify)
            ).encode()
        ).hexdigest(),
    )
    if cache_path.exists() and frames_path.exists():
        cached = json.loads(cache_path.read_text("utf-8"))
        if cached.get("provenance") == provenance:
            return (
                cached["annotation"],
                list(read_rows(frames_path)),
                json.loads((folder / "annotation.json").read_text("utf-8")),
            )
    metadata = json.loads((folder / "metadata.json").read_text("utf-8"))
    original = json.loads((folder / "annotation.json").read_text("utf-8"))
    contexts = json.loads((folder / "diagnostic/contexts.json").read_text("utf-8"))
    surface = wkt.loads(metadata["surface_wkt"]).buffer(1e-5)
    native_frames = {
        round(row["time_s"], 7): row for row in read_rows(folder / "reference.jsonl.gz")
    }
    physics = iter(read_rows(folder / "physics.jsonl.gz"))
    physical = next(physics, None)
    frames = []
    contract = Contract()
    for local in read_rows(folder / "diagnostic/inputs.jsonl.gz"):
        at_s = local["at_s"]
        while physical is not None and physical["time_s"] < at_s - 1e-7:
            physical = next(physics, None)
        if physical is None or abs(physical["time_s"] - at_s) > 1e-7:
            raise ValueError("missing independent physical sample")
        poses = physical["states"]
        bodies = {
            key: rectangles(
                [pose],
                metadata["vehicles"][key]["length_m"],
                metadata["vehicles"][key]["width_m"],
            )[0]
            for key, pose in poses.items()
        }
        witnesses = []
        for key, measurement in local["measurements"].items():
            own = metadata["vehicles"][key]
            plan = measurement["own_controls"]["planned_path_xyh_m"]
            footprints = rectangles(plan, own["length_m"], own["width_m"])
            sweep = shapely.union_all(footprints)
            required = sweep.difference(bodies[key])
            admissible = surface.covers(sweep)
            for other, body in bodies.items():
                if other == key or math.dist(poses[key][:2], poses[other][:2]) > 12.0:
                    continue
                intersection = required.intersection(body)
                if intersection.area <= contract.minimum_intersection_area_m2:
                    continue
                witnesses.append(
                    dict(
                        requester=key,
                        blocker=other,
                        c1_expected=measurement["own_controls"]["requested_speed_mps"]
                        > 0,
                        road_admissible=bool(admissible),
                        body_intersection=True,
                        mechanism="independent_commanded_movement_occupation",
                        resource=dict(
                            id=native_frames[at_s]["states"][key]["resource_id"],
                            polygon_wkt=intersection.wkt,
                        ),
                        movement=dict(id=contexts[key]["mission"]["movement_id"]),
                        alternatives="not_evaluated",
                    )
                )
        frames.append(
            dict(time_s=at_s, states=native_frames[at_s]["states"], witnesses=witnesses)
        )
    reference = qualify(frames, metadata, original["integrity"], contract)
    target_dir = folder / "evaluation"
    target_dir.mkdir(exist_ok=True)
    json_file(
        target_dir / "physical-reference.json",
        dict(
            schema="independent-commanded-movement-reference-1",
            annotation=reference,
            thresholds=contract.record(),
            source="native_body_poses_and_own_declared_driver_plans",
            detector_outputs_used=False,
            provenance=provenance,
        ),
    )
    with gzip.open(
        target_dir / "physical-reference-frames.jsonl.gz",
        "wt",
        encoding="utf-8",
        compresslevel=3,
    ) as stream:
        for frame in frames:
            stream.write(json.dumps(frame, separators=(",", ":")) + "\n")
    return reference, frames, original


def reference_groups(edges):
    """Union de contraintes physiques contemporaines, sans hypothèse de cycle."""
    parent = {}

    def root(item):
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left, right):
        left, right = root(left), root(right)
        if left != right:
            parent[left] = right

    for edge in edges:
        union(edge["requester"], edge["blocker"])
    polygons = [wkt.loads(edge["resource"]["polygon_wkt"]) for edge in edges]
    for index, left in enumerate(edges):
        for offset, right in enumerate(edges[index + 1 :], index + 1):
            if polygons[index].intersection(polygons[offset]).area > 0:
                union(left["requester"], right["requester"])
    groups = defaultdict(set)
    for edge in edges:
        groups[root(edge["requester"])].add(edge["requester"])
    return sorted((sorted(values) for values in groups.values() if len(values) >= 2))


def active_events(reference, at_s):
    return [
        event
        for event in reference["events"]
        if event["confirmed"]
        and any(
            interval["confirmed"]
            and interval["confirmation_at_s"] <= at_s + 1e-7
            and interval["start_s"] - 1e-7 <= at_s <= interval["end_s"] + 1e-7
            for interval in event["effect_intervals"]
        )
    ]


def metrics(counts):
    tp, fp, fn = (counts.get(name, 0) for name in ("tp", "fp", "fn"))
    return dict(
        **dict(counts),
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    )


def update_binary(counts, prediction, target):
    counts[
        "tp"
        if prediction and target
        else "fp"
        if prediction
        else "fn"
        if target
        else "tn"
    ] += 1


def match_groups(predicted, target, counts):
    unmatched = {tuple(group) for group in target}
    for group in predicted:
        key = tuple(sorted(group))
        if key in unmatched:
            counts["tp"] += 1
            unmatched.remove(key)
        else:
            counts["fp"] += 1
    counts["fn"] += len(unmatched)


def evaluate_episode(folder):
    folder = Path(folder)
    complete = folder / "complete.json"
    if complete.exists():
        manifest = json.loads(complete.read_text("utf-8"))
        if any(
            file_sha256(folder / name) != expected
            for name, expected in manifest.get("files", {}).items()
        ):
            raise ValueError(
                "immutable experimental evidence changed before evaluation"
            )
    reference, frames, native = physical_reference(folder)
    by_time = {frame["time_s"]: frame for frame in frames}
    inputs = {
        row["at_s"]: row for row in read_rows(folder / "diagnostic/inputs.jsonl.gz")
    }
    deliveries = defaultdict(dict)
    for event in read_rows(folder / "diagnostic/reports.jsonl.gz"):
        deliveries[event["time_s"]][event["message"]["sender_id"]] = event[
            "delivered_to"
        ]
    counts = {
        name: Counter()
        for name in (
            "A",
            "B",
            "B_groups",
            "C_groups",
            "links_global",
            "links_observable",
            "A_collective_global",
            "B_collective_global",
            "C_collective_global",
            "affected_B",
            "affected_C",
        )
    }
    abstentions = {name: Counter() for name in ("A", "B")}
    spec = json.loads((folder / "spec.json").read_text("utf-8"))
    topology = spec["family"]
    first_predictions = defaultdict(list)
    group_timeline = []
    fully_observable = Counter()
    validity = {name: Counter() for name in ("B", "C")}
    localization = []
    raw_coverage = defaultdict(lambda: deque(maxlen=76))
    for decision in read_rows(folder / "diagnostic/decisions.jsonl.gz"):
        at_s = decision["at_s"]
        if abs(at_s / 0.4 - round(at_s / 0.4)) > 1e-7:
            continue
        frame = by_time[at_s]
        events = active_events(reference, at_s)
        affected = {event["requester"] for event in events}
        edges = [
            edge
            for edge in frame["witnesses"]
            if edge["requester"] in affected and edge["road_admissible"]
        ]
        true_pairs = {(edge["requester"], edge["blocker"]) for edge in edges}
        groups = reference_groups(edges)
        current_inputs = inputs[at_s]["measurements"]
        for ego, local_input in current_inputs.items():
            current_blockers = {
                edge["blocker"]
                for edge in frame["witnesses"]
                if edge["requester"] == ego and edge["road_admissible"]
            }
            received_current = {
                row["message"]["sender_id"]
                for row in (local_input.get("v2v") or {}).get("received", [])
                if abs(row["message"]["observed_at_s"] - at_s) <= 1e-7
            }
            raw_coverage[ego].append((at_s, bool(current_blockers & received_current)))
            if ego in affected:
                history = raw_coverage[ego]
                fully_observable["positive_ego_appointments"] += 1
                fully_observable["positive_ego_full_30s_received_coverage"] += bool(
                    len(history) == 76
                    and abs(history[0][0] - (at_s - 30.0)) <= 1e-7
                    and all(occupied for _, occupied in history)
                )
        if groups:
            group_timeline.append(dict(at_s=at_s, groups=groups))
        for ego, prediction in decision["vehicles"].items():
            if not native["integrity"]["complete_measurements"]:
                continue
            for mode in ("A", "B"):
                result = prediction[mode]
                if result is None:
                    continue
                positive = result["status"] == "CONFIRMED_OBSTRUCTION"
                update_binary(counts[mode], positive, ego in affected)
                abstentions[mode]["total"] += 1
                abstentions[mode]["unknown"] += result["status"] == "UNKNOWN"
                if result["status"] == "UNKNOWN":
                    abstentions[mode][result["reason"]] += 1
                if positive:
                    first_predictions[mode + ":" + ego].append(at_s)
            current_inputs = inputs[at_s]["measurements"]
            possible_reporters = {ego} | {
                sender
                for sender, recipients in deliveries[at_s].items()
                if ego in recipients
            }
            observable_edges = []
            for edge in edges:
                requester, blocker = edge["requester"], edge["blocker"]
                mailbox = current_inputs.get(requester, {}).get("v2v")
                seen = any(
                    row["message"]["sender_id"] == blocker
                    and abs(row["message"]["observed_at_s"] - at_s) <= 1e-7
                    for row in (mailbox or {}).get("received", [])
                )
                if requester in possible_reporters and seen:
                    observable_edges.append(edge)
            target_groups = reference_groups(observable_edges)
            simple = (
                [prediction["C"]["simple_count_requesters"]]
                if prediction["C"]["simple_count_baseline"]
                else []
            )
            predicted_groups = [
                group["affected_requesters"] for group in prediction["C"]["groups"]
            ]
            match_groups(simple, target_groups, counts["B_groups"])
            match_groups(predicted_groups, target_groups, counts["C_groups"])
            update_binary(counts["A_collective_global"], False, bool(groups))
            update_binary(counts["B_collective_global"], bool(simple), bool(groups))
            update_binary(
                counts["C_collective_global"], bool(predicted_groups), bool(groups)
            )
            for mode, predicted in (("B", simple), ("C", predicted_groups)):
                for group in predicted:
                    valid = any(set(group) <= set(target) for target in groups)
                    validity[mode]["valid" if valid else "invalid"] += 1
                reported_affected = {
                    requester for group in predicted for requester in group
                }
                target_affected = {
                    requester for group in target_groups for requester in group
                }
                counts["affected_" + mode]["tp"] += len(
                    reported_affected & target_affected
                )
                counts["affected_" + mode]["fp"] += len(
                    reported_affected - target_affected
                )
                counts["affected_" + mode]["fn"] += len(
                    target_affected - reported_affected
                )
            if prediction["B"]["status"] == "CONFIRMED_OBSTRUCTION":
                for proof in prediction["B"].get("current_occupants", []):
                    matching = next(
                        (
                            edge
                            for edge in edges
                            if edge["requester"] == ego
                            and edge["blocker"] == proof["sender_id"]
                        ),
                        None,
                    )
                    if matching is not None:
                        predicted_area = wkt.loads(proof["resource_polygon_wkt"])
                        reference_area = wkt.loads(matching["resource"]["polygon_wkt"])
                        localization.append(
                            predicted_area.intersection(reference_area).area
                            / predicted_area.union(reference_area).area
                        )
            predicted_pairs = {
                (row["requester"], row["blocker"])
                for row in prediction["C"]["edges"]
                if row["level"] == "CONFIRMED"
            }
            observable_pairs = {
                (edge["requester"], edge["blocker"]) for edge in observable_edges
            }
            for key, targets in (
                ("links_global", true_pairs),
                ("links_observable", observable_pairs),
            ):
                counts[key]["tp"] += len(predicted_pairs & targets)
                counts[key]["fp"] += len(predicted_pairs - targets)
                counts[key]["fn"] += len(targets - predicted_pairs)
            fully_observable["global_current_relations"] += len(true_pairs)
            fully_observable["one_hop_available_relations"] += len(observable_pairs)
    delays = {mode: [] for mode in ("A", "B")}
    event_results = []
    for event in reference["events"]:
        if not event["confirmed"]:
            continue
        for mode in ("A", "B"):
            start = event["confirmation_at_s"]
            candidates = [
                at
                for at in first_predictions[mode + ":" + event["requester"]]
                if start - 1e-7 <= at <= event["end_s"] + 1e-7
            ]
            delay = min(candidates) - start if candidates else None
            if delay is not None:
                delays[mode].append(delay)
            event_results.append(
                dict(
                    mode=mode,
                    requester=event["requester"],
                    reference_confirmation_s=start,
                    first_accepted_confirmation_s=min(candidates)
                    if candidates
                    else None,
                    delay_s=delay,
                    missed=not bool(candidates),
                )
            )
    released = [
        event
        for event in reference["events"]
        if event["end_kind"] == "release_observed"
    ]
    reference_graph_cycles = []
    import networkx as nx

    for row in group_timeline:
        at_s = row["at_s"]
        matured = {event["requester"] for event in active_events(reference, at_s)}
        current_edges = [
            edge
            for edge in by_time[at_s]["witnesses"]
            if edge["requester"] in matured and edge["road_admissible"]
        ]
        for group in row["groups"]:
            graph = nx.DiGraph(
                (edge["requester"], edge["blocker"])
                for edge in current_edges
                if edge["requester"] in group
            )
            if nx.is_directed_acyclic_graph(graph):
                reference_graph_cycles.append(
                    dict(at_s=at_s, affected_requesters=group)
                )
    resumed = []
    for event in released:
        end = by_time[event["end_s"]].get("states", {}).get(event["requester"])
        if end is None:
            continue
        later = [
            frame["states"][event["requester"]]["progress_m"]
            for frame in frames
            if frame["time_s"] > event["end_s"]
            and event["requester"] in frame.get("states", {})
        ]
        gain = max(later, default=end["progress_m"]) - end["progress_m"]
        if gain >= 1.0:
            resumed.append(
                dict(
                    requester=event["requester"],
                    released_at_s=event["end_s"],
                    subsequent_progress_m=gain,
                    previously_confirmed=event["confirmed"],
                    effect_established=bool(event["effect_intervals"]),
                    physical_effect_s=event["movement_effect_duration_s"],
                )
            )
    situation = spec.get("requested_situation")
    negative = reference["label"] == "NEGATIVE_CONFIRMED"
    rules = dict(
        nominal=negative,
        dense_progressing=negative and native["traffic"].get("peak_active", 0) >= 12,
        temporary_interaction=negative and bool(resumed),
        individual_obstruction=reference["label"] == "POSITIVE_CONFIRMED",
        partial_observability=fully_observable["global_current_relations"]
        > fully_observable["one_hop_available_relations"],
        natural_release=any(release["effect_established"] for release in resumed),
        collective=bool(group_timeline),
    )
    summary = dict(
        schema="integrated-evaluation-2",
        evaluation_code_sha256=file_sha256(Path(__file__)),
        name=folder.name,
        topology=topology,
        seed=spec.get("seed"),
        spec_id=spec.get("id"),
        situation=situation,
        physical_configuration=dict(
            topology=spec.get("topology"),
            movement_profile=spec.get("movement_profile"),
            active_limit=spec.get("active_limit"),
            demand=spec.get("demand"),
        ),
        scenario_acceptance=dict(
            expected_situation=situation,
            physical_rule_satisfied=rules.get(situation),
            physics_valid=native["integrity"]["complete_measurements"],
            label_inferred_from_scenario=False,
        ),
        reference_label=reference["label"],
        native_reference_label=native["label"],
        valid_physics=native["integrity"]["complete_measurements"],
        anomalies=native["integrity"]["errors"],
        metrics={name: metrics(values) for name, values in counts.items()},
        abstentions={name: dict(values) for name, values in abstentions.items()},
        event_results=event_results,
        delays_s=delays,
        confirmed_reference_events=sum(
            event["confirmed"] for event in reference["events"]
        ),
        released_constraints=len(released),
        observed_releases_with_resumed_progress=resumed,
        peak_collective_affected=max(
            (len(group) for row in group_timeline for group in row["groups"]), default=0
        ),
        collective_reference_appointments=len(group_timeline),
        acyclic_collective_appointments=reference_graph_cycles,
        traffic=native["traffic"],
        information_coverage=dict(fully_observable),
        group_validity={
            name: dict(
                **dict(value),
                precision=value["valid"] / sum(value.values())
                if sum(value.values())
                else None,
            )
            for name, value in validity.items()
        },
        localization_iou=dict(
            samples=len(localization),
            mean=sum(localization) / len(localization) if localization else None,
            minimum=min(localization) if localization else None,
        ),
        group_metric_scope="B_groups/C_groups require exact observable membership; group_validity accepts true partial groups",
        scope="development_demonstrations_not_reserved_M02_test",
        statistical_unit="episode; frame counts are descriptive, not independent samples",
    )
    result_path = folder / "evaluation/results.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text("utf-8"))
        if previous.get("evaluation_code_sha256") != summary["evaluation_code_sha256"]:
            previous_path = (
                folder
                / "evaluation"
                / ("results-previous-" + file_sha256(result_path)[:12] + ".json")
            )
            if not previous_path.exists():
                previous_path.write_bytes(result_path.read_bytes())
    json_file(result_path, summary)
    json_file(folder / "evaluation/collective-reference.json", group_timeline)
    return summary


def evaluate_campaign(root):
    root = Path(root)
    summaries = []
    for complete in sorted(root.glob("*/complete.json")):
        result_path = complete.parent / "evaluation/results.json"
        cached = (
            json.loads(result_path.read_text("utf-8")) if result_path.exists() else None
        )
        summary = (
            cached
            if cached
            and cached.get("evaluation_code_sha256") == file_sha256(Path(__file__))
            else evaluate_episode(complete.parent)
        )
        summaries.append(summary)
        json_file(
            root / "evaluation-progress.json",
            dict(completed=len(summaries), last=summary["name"]),
        )
    totals = defaultdict(Counter)
    by_topology = defaultdict(lambda: defaultdict(Counter))
    for summary in summaries:
        for name, values in summary["metrics"].items():
            raw = {
                key: value
                for key, value in values.items()
                if key in {"tp", "fp", "fn", "tn"}
            }
            totals[name].update(raw)
            by_topology[summary["topology"]][name].update(raw)
    result = dict(
        episodes=summaries,
        episode_count=len(summaries),
        metrics={name: metrics(values) for name, values in totals.items()},
        by_topology={
            topology: {name: metrics(values) for name, values in modes.items()}
            for topology, modes in by_topology.items()
        },
        future_information_used=False,
        reserved_test_used_for_calibration=False,
        training_performed=False,
        communication_model="ideal_30m_0.4s_TTL1.2s",
    )
    json_file(root / "evaluation.json", result)
    return result
