"""Complète le contrôle strict v4 pour les projections ambiguës de coordonnées arrondies.

Les épisodes, labels, contrats et rapports stricts ne sont jamais réécrits.
Seul un avertissement curviligne peut être résolu en reconstruisant chaque
pose enregistrée et en prouvant la borne de quantification en espace métrique."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from shapely.geometry import LineString, Point
from cav_recovery.dataset.pipeline import read_rows, records, write_json, code_hashes

POSITION_ERROR_M = math.sqrt(2) * 0.5e-6


def verify_poses(line, station, distance, poses, rounded_position):
    """Reconstruit les poses dans le repère métrique sans relâcher la tolérance curviligne."""
    grid = np.arange(0, line.length + 0.49, 0.5)
    first = max(0, math.ceil(station * 2))
    last = min(len(grid), math.floor((station + distance) * 2) + 1)
    stations = [
        station,
        *map(float, grid[first:last]),
        min(line.length, station + distance),
    ]
    issues = []
    if not 0 <= station <= line.length or len(stations) != len(poses):
        return dict(valid=False, issues=["invalid_station_or_pose_count"])
    point = Point(rounded_position)
    nearest = line.distance(point)
    recorded = line.interpolate(station)
    # Both distances are 1-Lipschitz in the rounded input position. Their
    # difference can be at most twice the known Euclidean quantization error.
    nearest_excess = point.distance(recorded) - nearest
    if nearest_excess > 2 * POSITION_ERROR_M + 1e-9:
        issues.append("recorded_projection_not_nearest_within_quantization_bound")
    max_xy = 0.0
    max_angle = 0.0
    for pose, s in zip(poses, stations):
        expected = line.interpolate(max(0, min(s, line.length)))
        delta = expected.distance(Point(pose[:2]))
        max_xy = max(max_xy, delta)
        a = line.interpolate(max(0, min(s - 0.15, line.length)))
        b = line.interpolate(max(0, min(s + 0.15, line.length)))
        heading = math.atan2(b.y - a.y, b.x - a.x)
        angle = abs(
            math.atan2(math.sin(pose[2] - heading), math.cos(pose[2] - heading))
        )
        max_angle = max(max_angle, angle)
        if delta > POSITION_ERROR_M + 1e-9:
            issues.append("world_position_not_reproduced")
        if angle > 0.5e-6 + 1e-9:
            issues.append("world_heading_not_reproduced")
    return dict(
        valid=not issues,
        issues=sorted(set(issues)),
        maximum_position_error_m=max_xy,
        maximum_heading_error_rad=max_angle,
        nearest_distance_excess_m=nearest_excess,
        coordinate_quantization_bound_m=POSITION_ERROR_M,
    )


def audit(root, output=None):
    root = Path(root)
    campaign = json.loads((root / "campaign.json").read_text())
    unresolved = []
    resolved = []
    cache_hashes = {}
    count = 0
    strict_valid = 0
    frozen = code_hashes() == campaign["code"]
    if not frozen:
        unresolved.append(dict(kind="source_changed"))
    for folder, row in records(root):
        count += 1
        cache = root / "qc-reports" / (row["id"] + ".json")
        if not cache.exists():
            unresolved.append(dict(id=row["id"], kind="missing_strict_qc"))
            continue
        q = json.loads(cache.read_text())
        cache_hashes[row["id"]] = hashlib.sha256(cache.read_bytes()).hexdigest()
        sha = hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest()
        if q["manifest_sha256"] != sha or q["verification_code"] != campaign["code"]:
            unresolved.append(dict(id=row["id"], kind="strict_qc_provenance"))
            continue
        if q["valid"]:
            strict_valid += 1
            continue
        foreign = [
            i
            for i in q["issues"]
            if i["kind"] != "reference_not_on_assigned_native_route"
        ]
        if foreign:
            unresolved.append(dict(id=row["id"], issues=foreign))
            continue
        moments = {i["time_s"] for i in q["issues"]}
        meta = json.loads((folder / "metadata.json").read_text())
        # These are the exact native centerline points used by Route. Drop
        # only duplicate joins with the producer's fixed 1e-7 criterion.
        lines = {}
        for key, lanes in meta["route_lanes"].items():
            points = []
            for lane in lanes:
                for point in lane["points"]:
                    if not points or math.dist(points[-1], point) > 1e-7:
                        points.append(point)
            lines[key] = LineString(points)
        physics = {
            f["time_s"]: f
            for f in read_rows(folder / "physics.jsonl.gz")
            if f["time_s"] in moments
        }
        proofs = []
        for frame in read_rows(folder / "reference.jsonl.gz"):
            t = frame["time_s"]
            if t not in moments:
                continue
            for w in frame["witnesses"]:
                key = w["requester"]
                station = (
                    frame["states"][key]["progress_m"] + meta["initial_progress_m"][key]
                )
                result = verify_poses(
                    lines[key],
                    station,
                    w["movement"]["distance_m"],
                    w["movement"]["reference_poses"],
                    physics[t]["states"][key][:2],
                )
                proofs.append(
                    dict(time_s=t, requester=key, blocker=w["blocker"], **result)
                )
        if proofs and all(p["valid"] for p in proofs):
            resolved.append(
                dict(
                    id=row["id"],
                    original_issues=q["issues"],
                    proofs=proofs,
                    manifest_sha256=sha,
                    resolution="native_progress_world_pose_reconstruction_and_quantization_bounded_nearest_projection",
                )
            )
        else:
            unresolved.append(
                dict(id=row["id"], kind="route_precision_not_resolved", proofs=proofs)
            )
    strict = root / "validation-deep.json"
    if not strict.exists():
        unresolved.append(dict(kind="strict_campaign_qc_not_complete"))
    else:
        global_qc = json.loads(strict.read_text())
        if global_qc["episodes"] != count:
            unresolved.append(dict(kind="strict_coverage_incomplete"))
        # Per-episode findings can be resolved above; global identity/hash/
        # split findings are never waived by this numerical audit.
        for issue in global_qc["issues"]:
            if not ("id" in issue and "issues" in issue):
                unresolved.append(
                    dict(kind="unresolved_global_strict_issue", issue=issue)
                )
    result = dict(
        valid=not unresolved,
        episodes=count,
        strict_valid_episodes=strict_valid,
        resolved_projection_episodes=len(resolved),
        resolved=resolved,
        unresolved=unresolved,
        strict_report_sha256=hashlib.sha256(strict.read_bytes()).hexdigest()
        if strict.exists()
        else None,
        strict_episode_reports_sha256=cache_hashes,
        source_code=campaign["code"],
        contract_sha256=campaign["contract_sha256"],
        source_code_unchanged=frozen,
        raw_episodes_and_labels_unchanged=True,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation="composite audit; original strict failures retained, never rewritten",
    )
    write_json(Path(output) if output else root / "validation-combined.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.dataset, args.output)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("strict_episode_reports_sha256", "source_code", "resolved")
            },
            indent=2,
        )
    )
    raise SystemExit(0 if result["valid"] else 1)
