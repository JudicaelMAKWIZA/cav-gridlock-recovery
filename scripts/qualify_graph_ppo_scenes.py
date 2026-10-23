"""Requalifie physiquement les candidats Graph-PPO, sans entraînement."""

import argparse
import json
from pathlib import Path
import time

import torch  # Windows : charger Torch avant MetaDrive.

from cav_recovery.laboratory.storage import atomic_json, file_hash, json_file
from cav_recovery.recovery.environment import run
from cav_recovery.recovery.graph_marl_core import scenario, status_of_episode


def qualification(folder, summary):
    annotation = json_file(folder / "physical-annotation.json")
    confirmed = [event for event in annotation["events"] if event.get("confirmed")]
    return dict(
        summary=summary,
        status=status_of_episode(summary),
        qualified_label=annotation["label"],
        confirmed_events=len(confirmed),
        affected=sorted({event["requester"] for event in confirmed}),
        resources=sorted({event["resource_id"] for event in confirmed}),
        intervals=[dict(requester=event["requester"],
                        start_s=event["start_s"], end_s=event["end_s"])
                   for event in confirmed],
        spec_sha256=file_hash(folder / "spec.json"),
        annotation_sha256=file_hash(folder / "physical-annotation.json"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    shortlist = json_file(args.shortlist)["splits"]
    report = args.output / "qualification.json"
    results = json_file(report) if report.exists() else {"scenes": []}
    done = {(row["split"], row["source_id"]) for row in results["scenes"]}
    count = 0
    for split, rows in shortlist.items():
        for row in rows:
            key = (split, row["source_id"])
            if key in done:
                continue
            if args.limit and count >= args.limit:
                return
            folder = args.output / "nominal" / split / row["source_id"]
            if folder.exists():
                if not (folder / "summary.json").exists():
                    raise RuntimeError(f"Trace incomplète à inspecter : {folder}")
                summary = json_file(folder / "summary.json")
            else:
                start = time.perf_counter()
                summary = run(scenario(row, split), folder, method="nominal",
                              measurement_contract="REC-METRO-1")
                print(split, row["name"], summary["physical_label"],
                      round(time.perf_counter() - start, 1), flush=True)
            result = qualification(folder, summary)
            result.update(split=split, source_id=row["source_id"],
                          source_label=row["source_label"],
                          source_parent=row["source_parent"],
                          source_family=row["source_family"],
                          scenario=row)
            results["scenes"].append(result)
            atomic_json(report, results)
            count += 1


if __name__ == "__main__":
    main()
