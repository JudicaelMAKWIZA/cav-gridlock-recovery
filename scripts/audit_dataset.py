"""Vérifie les missions, poses initiales et fenêtres causales depuis les fichiers."""

import argparse
from collections import Counter
import hashlib
import json
import math
from itertools import zip_longest
from pathlib import Path

from cav_recovery.dataset.pipeline import records, read_rows, write_json, file_sha256


def audit(dataset, exported=None, report_path=None):
    dataset = Path(dataset)
    issues = []
    labels = {}
    parents = {}
    episodes = 0
    context_truth = {}
    map_truth = {}
    for folder, row in records(dataset):
        episodes += 1
        labels[row["id"]] = row["label"]
        parents[row["id"]] = row["parent_id"]
        spec = json.loads((folder / "spec.json").read_text())
        meta = json.loads((folder / "metadata.json").read_text())
        map_truth[row["id"]] = (
            hashlib.sha256(
                json.dumps(
                    [
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
                    ]
                ).encode()
            ).hexdigest(),
            hashlib.sha256(str(meta.get("surface_wkt")).encode()).hexdigest(),
        )
        for key, v in meta.get("vehicles", {}).items():
            context_truth[row["id"], key] = (
                v["introduced_at_s"],
                v["mission"]["destination"],
                hashlib.sha256(
                    json.dumps(
                        [p for lane in meta["route_lanes"][key] for p in lane["points"]]
                    ).encode()
                ).hexdigest(),
            )
        trace = read_rows(folder / "physics.jsonl.gz")
        try:
            first = next(trace, None)
        finally:
            trace.close()
        expected = set(row["integrity"].get("introduced", {}))
        if first is None:
            if row["label"] != "UNKNOWN_OR_CENSORED":
                issues.append(dict(id=row["id"], kind="missing_physical_trace"))
            continue
        if (
            set(first["states"]) != {"agent0"}
            or set(meta.get("vehicles", {})) != expected
        ):
            if row["label"] != "UNKNOWN_OR_CENSORED":
                issues.append(dict(id=row["id"], kind="initial_identity_set"))
            continue
        first_states = {}
        arrivals = {}
        bad = set()
        for event in read_rows(folder / "native_events.jsonl.gz"):
            if event.get("event") == "introduced":
                if (
                    not event["occupancy_checked"]
                    or event["time_s"] < event["scheduled_at_s"]
                ):
                    issues.append(dict(id=row["id"], kind="invalid_admission"))
                first_states[event["vehicle_id"]] = event["initial_pose"]
                continue
            flags = event["flags"]
            if flags["arrive_dest"]:
                arrivals[event["vehicle_id"]] = event["time_s"]
            if any(value for flag, value in flags.items() if flag != "arrive_dest"):
                bad.add(event["vehicle_id"])
        for agent in spec["agents"]:
            key = agent["id"]
            if key not in expected:
                continue
            road = next(
                r
                for r in spec["roads"]
                if r["kind"] == "in"
                and r["arm"] == agent["origin"]
                and r["lane"] == agent["lane"]
            )
            mission = meta["vehicles"][key]["mission"]
            if (
                mission["origin_lane"] != [road["start"], road["end"], agent["lane"]]
                or mission["destination"] != f"a{agent['destination']}_sink"
                or mission["checkpoints"]
                != [
                    road["start"],
                    road["end"],
                    f"a{agent['destination']}_exit",
                    f"a{agent['destination']}_sink",
                ]
            ):
                issues.append(
                    dict(
                        id=row["id"],
                        vehicle=key,
                        kind="programmed_mission_not_native_mission",
                    )
                )
            start, end = road["points"][0], road["points"][-1]
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            s = agent["longitude"]
            position = [start[0] + s * dx / length, start[1] + s * dy / length]
            # Native creation pose, before actuation; full 50Hz checks are cached separately.
            actual = first_states[key]
            heading = math.atan2(dy, dx)
            delta = math.atan2(
                math.sin(heading - actual[2]), math.cos(heading - actual[2])
            )
            if math.dist(position, actual[:2]) > 1e-5 or abs(delta) > 1e-5:
                issues.append(
                    dict(
                        id=row["id"],
                        vehicle=key,
                        kind="programmed_initial_pose_mismatch",
                    )
                )
        if (arrivals != row["integrity"]["arrivals"] or bad) and row[
            "label"
        ] != "UNKNOWN_OR_CENSORED":
            issues.append(
                dict(id=row["id"], kind="native_events_inconsistent_with_qualification")
            )
    windows = 0
    targets = 0
    partition_parents = {}
    if exported:
        exported = Path(exported)
        manifest = json.loads((exported / "manifest.json").read_text())
        for filename, expected in manifest["files"].items():
            path = exported / filename
            if not path.is_file() or file_sha256(path) != expected:
                issues.append(dict(kind="export_file_hash", file=filename))
        for path in exported.glob("*/features.jsonl.gz"):
            group = path.parent.name
            contexts = {}
            if path.with_name("map_context.jsonl.gz").exists():
                for context in read_rows(path.with_name("map_context.jsonl.gz")):
                    actual = (
                        hashlib.sha256(
                            json.dumps(context["lanes"]).encode()
                        ).hexdigest(),
                        hashlib.sha256(
                            str(context["surface_wkt"]).encode()
                        ).hexdigest(),
                    )
                    if (
                        actual != map_truth.get(context["episode_id"])
                        or context["available_at_s"] != 0
                        or set(context)
                        != {
                            "episode_id",
                            "available_at_s",
                            "lanes",
                            "surface_wkt",
                            "source",
                        }
                    ):
                        issues.append(
                            dict(
                                kind="invalid_or_privileged_static_map",
                                episode=context["episode_id"],
                            )
                        )
            if path.with_name("mission_context.jsonl.gz").exists():
                for context in read_rows(path.with_name("mission_context.jsonl.gz")):
                    key = (context["episode_id"], context["vehicle_id"])
                    actual = (
                        context["available_at_s"],
                        context["mission"]["destination"],
                        hashlib.sha256(
                            json.dumps(context["assigned_route_xy_m"]).encode()
                        ).hexdigest(),
                    )
                    if actual != context_truth.get(key) or any(
                        k in context
                        for k in ("label", "family", "density", "blocker", "events")
                    ):
                        issues.append(
                            dict(
                                kind="invalid_or_privileged_mission_context",
                                episode=key[0],
                            )
                        )
                    contexts[key] = context["available_at_s"]
            if path.with_name("ego_controls.jsonl.gz").exists():
                control_episode = None
                control_truth = {}
                for command in read_rows(path.with_name("ego_controls.jsonl.gz")):
                    if command["episode_id"] != control_episode:
                        control_episode = command["episode_id"]
                        control_truth = {
                            (key, frame["time_s"]): (
                                c["requested_speed_mps"],
                                c["accepted_speed_mps"],
                                c["action"],
                            )
                            for frame in read_rows(
                                dataset
                                / "episodes"
                                / control_episode
                                / "commands.jsonl.gz"
                            )
                            for key, c in frame["commands"].items()
                        }
                    actual = (
                        command["requested_speed_mps"],
                        command["accepted_speed_mps"],
                        command["action"],
                    )
                    if actual != control_truth.get(
                        (command["vehicle_id"], command["time_s"])
                    ) or any(
                        k in command
                        for k in ("guard_blockers", "observed_neighbors", "label")
                    ):
                        issues.append(
                            dict(
                                kind="invalid_or_privileged_ego_control",
                                episode=command["episode_id"],
                            )
                        )
            for sample in read_rows(path):
                if any(
                    name in sample
                    for name in (
                        "label",
                        "family",
                        "blocker",
                        "witnesses",
                        "guard_blockers",
                        "seed",
                    )
                ):
                    issues.append(
                        dict(kind="privileged_feature", episode=sample["episode_id"])
                    )
            feature_rows = iter(read_rows(path))
            lookahead = next(feature_rows, None)
            sample_episode = None
            samples = {}
            temporal = (
                iter(read_rows(path.with_name("tcn_windows.jsonl.gz")))
                if path.with_name("tcn_windows.jsonl.gz").exists()
                else None
            )
            temporal_truth = (
                iter(read_rows(path.with_name("temporal_targets.jsonl.gz")))
                if temporal is not None
                else None
            )
            for window, target in zip_longest(
                read_rows(path.with_name("windows.jsonl.gz")),
                read_rows(path.with_name("targets.jsonl.gz")),
            ):
                if (
                    window is None
                    or target is None
                    or window["window_id"] != target["window_id"]
                ):
                    issues.append(dict(kind="window_target_alignment", partition=group))
                    continue
                windows += 1
                targets += 1
                episode = window["episode_id"]
                if episode != sample_episode:
                    samples = {}
                    sample_episode = episode
                    while lookahead is not None and lookahead["episode_id"] <= episode:
                        if lookahead["episode_id"] == episode:
                            samples[lookahead["sample_index"]] = (
                                lookahead["time_s"],
                                lookahead["vehicle_id"],
                            )
                        lookahead = next(feature_rows, None)
                if (
                    contexts
                    and contexts.get((episode, window["vehicle_id"]), float("inf"))
                    > window["start_s"] + 1e-7
                ):
                    issues.append(
                        dict(
                            kind="mission_context_not_available_at_window_start",
                            window=window["window_id"],
                        )
                    )
                if episode not in labels:
                    issues.append(dict(kind="unknown_export_episode", id=episode))
                    continue
                if group != "unknown":
                    partition_parents.setdefault(parents[episode], set()).add(group)
                if group != "unknown" and labels[episode] == "UNKNOWN_OR_CENSORED":
                    issues.append(
                        dict(kind="unknown_in_supervised_features", id=episode)
                    )
                for index in window["sample_indices"]:
                    observed_at, vehicle = samples.get(index, (float("inf"), None))
                    if (
                        observed_at > window["end_s"] + 1e-7
                        or observed_at < window["start_s"] - 1e-7
                    ):
                        issues.append(
                            dict(kind="future_feature", window=window["window_id"])
                        )
                    if vehicle != window["vehicle_id"]:
                        issues.append(
                            dict(
                                kind="window_vehicle_mismatch",
                                window=window["window_id"],
                            )
                        )
                if temporal is not None:
                    long_window = next(temporal, None)
                    long_target = next(temporal_truth, None)
                    if (
                        long_window is None
                        or long_target is None
                        or long_window["window_id"] != window["window_id"]
                        or long_target["window_id"] != window["window_id"]
                    ):
                        issues.append(
                            dict(
                                kind="temporal_window_target_alignment",
                                window=window["window_id"],
                            )
                        )
                    else:
                        if (
                            long_window["episode_id"] != episode
                            or long_window["vehicle_id"] != window["vehicle_id"]
                            or long_window["end_s"] != window["end_s"]
                            or len(long_window["sample_indices"])
                            + long_window["left_padding_samples"]
                            != long_window["expected_samples"]
                        ):
                            issues.append(
                                dict(
                                    kind="temporal_identity_or_prefix_mask",
                                    window=window["window_id"],
                                )
                            )
                        for index in long_window["sample_indices"]:
                            observed_at, vehicle = samples.get(
                                index, (float("inf"), None)
                            )
                            if (
                                not long_window["end_s"]
                                - long_window["horizon_s"]
                                - 1e-7
                                <= observed_at
                                <= long_window["end_s"] + 1e-7
                                or vehicle != window["vehicle_id"]
                            ):
                                issues.append(
                                    dict(
                                        kind="noncausal_temporal_input",
                                        window=window["window_id"],
                                    )
                                )
                        if (
                            long_target["retrospective_persistent_event"]
                            != target["event_obstructed"]
                            or long_target["supervision_available"]
                            != target["supervision_available"]
                        ):
                            issues.append(
                                dict(
                                    kind="temporal_reference_changed",
                                    window=window["window_id"],
                                )
                            )
                if (
                    not target["supervision_available"]
                    and target["event_obstructed"] is not None
                ):
                    issues.append(
                        dict(
                            kind="unknown_target_used_as_negative",
                            window=target["window_id"],
                        )
                    )
            feature_rows.close()
            if temporal is not None:
                if (
                    next(temporal, None) is not None
                    or next(temporal_truth, None) is not None
                ):
                    issues.append(dict(kind="extra_temporal_windows", partition=group))
                temporal.close()
                temporal_truth.close()
        if any(len(groups) > 1 for groups in partition_parents.values()):
            issues.append(dict(kind="export_parent_leakage"))
    result = dict(
        valid=not issues,
        episodes=episodes,
        windows=windows,
        targets=targets,
        issues=issues,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="programmed_native_missions_and_export_audit_complements_dataset_validate_deep",
    )
    write_json(
        Path(report_path)
        if report_path is not None
        else dataset / "audit-programmed-missions.json",
        result,
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument(
        "--report", type=Path, help="Write the report outside an immutable dataset."
    )
    args = parser.parse_args(argv)
    result = audit(args.dataset, args.export, args.report)
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
