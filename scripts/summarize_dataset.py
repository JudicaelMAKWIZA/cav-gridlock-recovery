"""Analyse les preuves, mouvements et biais de campagne, sans détecteur appris."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import networkx as nx
from cav_recovery.dataset.pipeline import records, read_rows, write_json
from cav_recovery.dataset.qualify import POSITIVE, NEGATIVE, UNKNOWN


def summarize(root, exported=None):
    root = Path(root)
    labels = Counter()
    cells = defaultdict(Counter)
    parents = defaultdict(set)
    mechanisms = Counter()
    peaks = defaultdict(list)
    parameters = defaultdict(lambda: defaultdict(list))
    dimensions = set()
    lane_widths = set()
    lanes = set()
    observed_cells = defaultdict(Counter)
    pair_members = defaultdict(list)
    issues = []
    summaries = []
    source_hashes = {}
    simultaneous = Counter()
    acyclic = 0
    resolved = 0
    changing = 0
    moving_unaffected = 0
    succession_only = 0
    whole_acyclic = 0
    with gzip.open(
        root / "offline-motion-reference.jsonl.gz", "wt", compresslevel=6
    ) as offline:
        for folder, row in records(root):
            label = row["label"]
            labels[label] += 1
            spec = json.loads((folder / "spec.json").read_text())
            meta = json.loads((folder / "metadata.json").read_text())
            source_hashes[row["id"]] = json.loads(
                (folder / "manifest.json").read_text()
            )["files"]
            cells[spec["family"], spec["density_level"]][label] += 1
            parents[row["parent_id"]].add(label)
            observed_cells[spec["family"], row["traffic"]["peak_active"]][label] += 1
            mix = {
                kind: sum(a["movement_kind"] == kind for a in spec["agents"])
                / len(spec["agents"])
                for kind in ("left", "right", "straight")
            }
            pair_members[row["parent_id"], row["traffic"]["peak_active"]].append(
                dict(id=row["id"], label=label, movement_fractions=mix)
            )
            peaks[label].append(row["traffic"]["peak_active"])
            lane_widths.add(spec["topology"]["lane_width_m"])
            lanes.add(spec["topology"]["lane_num"])
            for v in meta["vehicles"].values():
                dimensions.add((v["length_m"], v["width_m"]))
            for name, value in [
                (
                    "right_turn_fraction",
                    sum(a["movement_kind"] == "right" for a in spec["agents"])
                    / len(spec["agents"]),
                ),
                (
                    "nominal_speed_mps",
                    statistics.mean(a["target_speed_mps"] for a in spec["agents"]),
                ),
                ("arrival_headway_s", spec["demand"]["headway_s"]),
                ("lane_width_m", spec["topology"]["lane_width_m"]),
            ]:
                parameters[label][name].append(value)
            events = row["events"]
            qualified = [e for e in events if e["confirmed"]]
            changing += any(
                len(e["blockers"]) > 1
                and len({(o["start_s"], o["end_s"]) for o in e["occupant_intervals"]})
                > 1
                for e in qualified
            )
            succession_only += any(
                len(e["blockers"]) > 1
                and max(o["end_s"] - o["start_s"] for o in e["occupant_intervals"]) < 30
                for e in qualified
            )
            resolved += any(not e["right_censored"] for e in events)
            commands = {
                f["time_s"]: f["commands"]
                for f in read_rows(folder / "commands.jsonl.gz")
            }
            deps = [
                d
                for d in row["dependencies"]
                if d["relevance"] == "confirmed_for_requested_movement"
            ]
            union_graph = nx.DiGraph()
            union_graph.add_edges_from((d["requester"], d["blocker"]) for d in deps)
            entire_acyclic = bool(deps) and nx.is_directed_acyclic_graph(union_graph)
            whole_acyclic += entire_acyclic
            active = {}
            intervals = []
            previous = {}
            resumptions = []
            peak_group = 0
            has_acyclic = False
            coexist = set()
            unaffected = 0
            episode_mechanisms = set()
            start_progress = {}
            before_constraint = {}
            last_t = 0.0
            for frame in read_rows(folder / "reference.jsonl.gz"):
                t = frame["time_s"]
                last_t = t
                present = set()
                physical = {
                    w["requester"] for w in frame["witnesses"] if w["road_admissible"]
                }
                for key, state in frame["states"].items():
                    start_progress.setdefault(key, state["progress_m"])
                    if key in physical:
                        before_constraint.setdefault(
                            key, state["progress_m"] - start_progress[key]
                        )
                    low = abs(state["speed_mps"]) < 0.15
                    if previous.get(key, False) and not low:
                        resumptions.append(
                            dict(vehicle=key, time_s=t, progress_m=state["progress_m"])
                        )
                    previous[key] = low
                    if low:
                        category = (
                            "physical_occupancy_candidate"
                            if key in physical
                            else "safety_wait_without_body_witness"
                            if commands[t][key]["accepted_speed_mps"] == 0
                            else "controller_response_or_startup"
                        )
                        item = (key, category)
                        present.add(item)
                        interval = active.setdefault(
                            item, dict(vehicle=key, kind=category, start_s=t, end_s=t)
                        )
                        interval["end_s"] = t
                for key in list(active):
                    if key not in present:
                        intervals.append(active.pop(key))
                current = [
                    d
                    for d in deps
                    if d["start_s"] <= t <= d["end_s"] and d["confirmation_at_s"] <= t
                ]
                graph = nx.DiGraph()
                graph.add_edges_from((d["requester"], d["blocker"]) for d in current)
                affected = {d["requester"] for d in current}
                if current:
                    unaffected = max(
                        unaffected,
                        sum(
                            abs(s["speed_mps"]) > 0.5
                            and k not in affected
                            and k not in physical
                            and commands[t][k]["reason"] == "free_native_route"
                            for k, s in frame["states"].items()
                        ),
                    )
                    for nodes in nx.weakly_connected_components(graph):
                        peak_group = max(peak_group, len(nodes))
                        if len(nodes) > 1 and nx.is_directed_acyclic_graph(
                            graph.subgraph(nodes)
                        ):
                            has_acyclic = True
                    pairs = {(d["requester"], d["blocker"]) for d in current}
                    kinds = {
                        w["mechanism"]
                        for w in frame["witnesses"]
                        if (w["requester"], w["blocker"]) in pairs
                    }
                    episode_mechanisms.update(kinds)
                    if len(kinds) > len(coexist):
                        coexist = kinds
            for item in active.values():
                item["right_censored"] = True
                intervals.append(item)
            for item in intervals:
                item.setdefault("right_censored", False)
            offline.write(
                json.dumps(
                    dict(
                        episode_id=row["id"],
                        immobilization_intervals=intervals,
                        motion_resumptions=resumptions,
                        low_speed_threshold_mps=0.15,
                        time_resolution_s=0.2,
                        label_source="global_native_offline_reference_only",
                        absence_of_body_witness_does_not_establish_controller_failure=True,
                    ),
                    separators=(",", ":"),
                )
                + "\n"
            )
            acyclic += has_acyclic
            moving_unaffected += unaffected > 0
            simultaneous[len(coexist)] += 1
            mechanisms.update(episode_mechanisms)
            summary = dict(
                id=row["id"],
                label=label,
                family=row["family"],
                density=spec["density_level"],
                peak_active=row["traffic"]["peak_active"],
                peak_group_size=peak_group,
                acyclic_confirmed_component=has_acyclic,
                simultaneous_confirmed_mechanisms=sorted(coexist),
                acyclic_union_of_confirmed_dependencies=entire_acyclic,
                peak_unaffected_moving_while_confirmed=unaffected,
                introduced=row["traffic"]["total_introduced"],
                progressed_before_first_constraint_m=before_constraint,
                technical_errors=row["integrity"]["errors"],
            )
            if row["traffic"]["peak_active"] < 8:
                issues.append(
                    dict(id=row["id"], kind="fewer_than_eight_simultaneously_active")
                )
            if row["integrity"]["errors"]:
                issues.append(dict(id=row["id"], kind="physical_or_technical_failure"))
            summaries.append(summary)
    comparable = [
        dict(
            topology=f,
            density=d,
            labels=dict(counts),
            both_classes=counts[POSITIVE] > 0 and counts[NEGATIVE] > 0,
        )
        for (f, d), counts in sorted(cells.items())
    ]
    matched = []
    for (parent, peak), members in pair_members.items():
        for positive in (m for m in members if m["label"] == POSITIVE):
            for negative in (m for m in members if m["label"] == NEGATIVE):
                delta = max(
                    abs(
                        positive["movement_fractions"][k]
                        - negative["movement_fractions"][k]
                    )
                    for k in ("left", "right", "straight")
                )
                matched.append(
                    dict(
                        parent_id=parent,
                        peak_active=peak,
                        positive_id=positive["id"],
                        negative_id=negative["id"],
                        maximum_movement_fraction_difference=delta,
                        movement_mix_close=delta <= 0.15,
                        matching="same parent geometry and actual peak active count; diagnostic cohort, labels not inferred from parameters",
                    )
                )
    write_json(root / "matched-class-pairs.json", matched)
    report = dict(
        episodes=sum(labels.values()),
        labels=dict(labels),
        quota_reached=labels[POSITIVE] >= 1000 and labels[NEGATIVE] >= 1000,
        dimensions_m=sorted(dimensions),
        lane_widths_m=sorted(lane_widths),
        lanes_per_direction=sorted(lanes),
        peak_active_by_class={
            k: dict(minimum=min(v), maximum=max(v), mean=statistics.mean(v))
            for k, v in peaks.items()
        },
        comparison_cells=comparable,
        cells_with_both_classes=sum(c["both_classes"] for c in comparable),
        observed_peak_cells=[
            dict(topology=f, peak_active=p, labels=dict(c))
            for (f, p), c in sorted(observed_cells.items())
        ],
        geometry_and_observed_density_matched_pairs=len(matched),
        matched_pairs_with_close_movement_mix=sum(
            p["movement_mix_close"] for p in matched
        ),
        parent_groups_with_both_classes=sum(
            {POSITIVE, NEGATIVE} <= values for values in parents.values()
        ),
        parameter_means_by_class={
            label: {name: statistics.mean(v) for name, v in values.items()}
            for label, values in parameters.items()
        },
        episodes_by_confirmed_mechanism_overlapping=dict(mechanisms),
        episodes_with_changing_occupants=changing,
        episodes_with_confirmation_requiring_successive_blockers=succession_only,
        episodes_with_observed_constraint_release=resolved,
        episodes_with_acyclic_confirmed_component=acyclic,
        episodes_with_entire_confirmed_graph_acyclic=whole_acyclic,
        episodes_with_moving_unaffected_vehicles=moving_unaffected,
        simultaneous_mechanism_count_histogram=dict(simultaneous),
        episode_evidence=summaries,
        issues=issues,
        source_episode_files_sha256=source_hashes,
        offline_reference_not_model_input=True,
        scientific_acceptance=False,
        limitations=[
            "ideal local sensing without occlusion/noise",
            "operational prolonged low-progress subset, not all transient obstructions",
            "spatial relevance is not a unique dynamic causal root",
            "indirect references limited to verified overlapping two-link chains",
            "no traffic-signal or regulatory-priority model",
            "human scientific acceptance still required",
        ],
    )
    write_json(root / "scientific-summary.json", report)
    if exported is not None:
        exported = Path(exported)
        partition = json.loads((root / "splits.json").read_text())["episodes"]
        from contextlib import ExitStack

        with ExitStack() as stack:
            streams = {
                group: stack.enter_context(
                    gzip.open(
                        exported / group / "mission_context.jsonl.gz",
                        "wt",
                        compresslevel=6,
                    )
                )
                for group in set(partition.values()) | {"unknown"}
            }
            controls = {
                group: stack.enter_context(
                    gzip.open(
                        exported / group / "ego_controls.jsonl.gz",
                        "wt",
                        compresslevel=6,
                    )
                )
                for group in set(partition.values()) | {"unknown"}
            }
            maps = {
                group: stack.enter_context(
                    gzip.open(
                        exported / group / "map_context.jsonl.gz", "wt", compresslevel=6
                    )
                )
                for group in set(partition.values()) | {"unknown"}
            }
            for folder, row in records(root):
                group = "unknown" if row["label"] == UNKNOWN else partition[row["id"]]
                meta = json.loads((folder / "metadata.json").read_text())
                spec = json.loads((folder / "spec.json").read_text())
                assignments = {a["id"]: a for a in spec["agents"]}
                maps[group].write(
                    json.dumps(
                        dict(
                            episode_id=row["id"],
                            available_at_s=0.0,
                            lanes=[
                                {
                                    k: r[k]
                                    for k in (
                                        "start",
                                        "end",
                                        "lane",
                                        "points",
                                        "width_m",
                                        "kind",
                                    )
                                }
                                for r in spec["roads"]
                            ],
                            surface_wkt=meta.get("surface_wkt"),
                            source="static_road_geometry_known_to_native_navigation_no_actor_states",
                        ),
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                for key, vehicle in meta["vehicles"].items():
                    context = dict(
                        episode_id=row["id"],
                        vehicle_id=key,
                        available_at_s=vehicle["introduced_at_s"],
                        mission=vehicle["mission"],
                        requested_movement=dict(
                            id=assignments[key]["movement_id"],
                            kind=assignments[key]["movement_kind"],
                        ),
                        assigned_route_xy_m=[
                            p
                            for lane in meta["route_lanes"][key]
                            for p in lane["points"]
                        ],
                        length_m=vehicle["length_m"],
                        width_m=vehicle["width_m"],
                        source="own_fixed_navigation_plan_known_at_admission",
                    )
                    streams[group].write(
                        json.dumps(context, separators=(",", ":")) + "\n"
                    )
                for frame in read_rows(folder / "commands.jsonl.gz"):
                    for key, command in frame["commands"].items():
                        controls[group].write(
                            json.dumps(
                                dict(
                                    episode_id=row["id"],
                                    vehicle_id=key,
                                    time_s=frame["time_s"],
                                    requested_speed_mps=command["requested_speed_mps"],
                                    accepted_speed_mps=command["accepted_speed_mps"],
                                    action=command["action"],
                                    source="own_command_known_at_decision_time",
                                ),
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
        manifest = json.loads((exported / "manifest.json").read_text())
        for path in (
            list(exported.glob("*/mission_context.jsonl.gz"))
            + list(exported.glob("*/ego_controls.jsonl.gz"))
            + list(exported.glob("*/map_context.jsonl.gz"))
        ):
            manifest["files"][path.relative_to(exported).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        manifest["causal_static_context"] = dict(
            file="mission_context.jsonl.gz",
            join=["episode_id", "vehicle_id"],
            availability="known at native admission; use only at times >= available_at_s",
            contents="own mission, requested movement and planned road route; no future vehicle states or global labels",
        )
        manifest["derived_annotation_script_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        manifest["causal_ego_controls"] = dict(
            file="ego_controls.jsonl.gz",
            join=["episode_id", "vehicle_id", "time_s"],
            availability="own command at this decision time; only past/present commands may enter a window",
            excluded=["guard_blockers", "observed_neighbors", "global_reference"],
        )
        manifest["causal_static_map"] = dict(
            file="map_context.jsonl.gz",
            join=["episode_id"],
            assumption="native navigation has the static road map; dynamic global actor states remain excluded",
        )
        manifest["non_predictive_join_columns"] = [
            "episode_id",
            "vehicle_id",
            "sample_index",
            "window_id",
            "time_s",
        ]
        manifest["time_coordinate_policy"] = (
            "absolute timestamps align and audit streams; models should use causal lags relative to the window end"
        )
        manifest["offline_motion_reference_sha256"] = hashlib.sha256(
            (root / "offline-motion-reference.jsonl.gz").read_bytes()
        ).hexdigest()
        write_json(exported / "manifest.json", manifest)
    return {
        k: v
        for k, v in report.items()
        if k not in ("episode_evidence", "source_episode_files_sha256")
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.dataset, args.export), indent=2))
