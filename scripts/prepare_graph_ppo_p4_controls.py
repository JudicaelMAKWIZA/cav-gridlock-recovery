"""Présélectionne deux contrôles P4 récupérables sans modifier les preuves."""

import argparse
from pathlib import Path
import shutil

from cav_recovery.laboratory.storage import atomic_json, file_hash, json_file


CASES = (("oblique-4", 4, "P4-300"),
         ("oblique-8", 8, "P4-guard-envelope"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p4-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    args = parser.parse_args()
    args.staged_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, vehicles, intervention_name in CASES:
        nominal = args.p4_root / "runs" / name / "N-300"
        intervention = args.p4_root / "runs" / name / intervention_name
        manifest = json_file(nominal / "manifest.json")["files"]
        for filename in ("spec.json", "physical-annotation.json", "summary.json"):
            if manifest[filename] != file_hash(nominal / filename):
                raise ValueError(f"Preuve P4 historique modifiée : {name}/{filename}")
        spec = json_file(nominal / "spec.json")
        annotation = json_file(nominal / "physical-annotation.json")
        nominal_summary = json_file(nominal / "summary.json")
        intervention_manifest = json_file(intervention / "manifest.json")["files"]
        for filename in ("spec.json", "physical-annotation.json", "summary.json"):
            if intervention_manifest[filename] != file_hash(intervention / filename):
                raise ValueError(f"Preuve d'intervention P4 modifiée : {name}/{filename}")
        p4_summary = json_file(intervention / "summary.json")
        if (spec["active_limit"] != vehicles or len(spec["agents"]) != vehicles
                or annotation["label"] != "POSITIVE_CONFIRMED"
                or nominal_summary["physical_label"] != "POSITIVE_CONFIRMED"
                or len(p4_summary["arrivals"]) != vehicles
                or p4_summary["errors"]):
            raise ValueError(f"Contrôle P4 non qualifié : {name}")
        identity = f"p4-{name}-{spec['seed']}"
        target = args.staged_root / f"{identity}.json"
        if target.exists():
            if file_hash(target) != manifest["spec.json"]:
                raise ValueError(f"Spécification P4 staged différente : {identity}")
        else:
            shutil.copy2(nominal / "spec.json", target)
        confirmed = [event for event in annotation["events"]
                     if event.get("confirmed")]
        rows.append(dict(
            name=identity, source_id=identity, source_spec_id=spec["id"],
            source_campaign="P4_development_control",
            source_parent="p4:integrated:collective-3",
            source_family=spec["family"],
            source_label=annotation["label"],
            source_confirmed_events=len(confirmed),
            source_affected=len({event["requester"] for event in confirmed}),
            source_spec_sha256=manifest["spec.json"],
            source_annotation_sha256=manifest["physical-annotation.json"],
            source_spec_path=str(target),
            seed=spec["seed"], vehicles=vehicles, horizon_s=spec["horizon_s"],
            historical_nominal=str(nominal), historical_p4=str(intervention),
        ))
    atomic_json(args.output, dict(
        source="P4 historical development controls, not independent test",
        status="candidate_only_requires_native_requalification",
        splits={"train": rows, "validation": []},
    ))
    print({row["name"]: row["vehicles"] for row in rows})


if __name__ == "__main__":
    main()
