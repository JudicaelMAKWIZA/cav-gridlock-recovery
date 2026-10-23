"""Mesure l'effet causal observable de deux politiques sur une scène appariée."""

import argparse
import gzip
import json
from pathlib import Path

from cav_recovery.laboratory.storage import atomic_json, file_hash, json_file
from cav_recovery.recovery.evaluation import compare


def commands(folder):
    with gzip.open(folder / "commands.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            yield (row["ego"], round(row["at_s"], 6)), row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--p4", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    p4 = dict(commands(args.p4))
    graph = dict(commands(args.graph))
    shared = sorted(p4.keys() & graph.keys(), key=lambda item: (item[1], item[0]))
    physical = [key for key in shared if any(
        abs(a - b) > 1e-7 for a, b in zip(p4[key]["native_action"],
                                           graph[key]["native_action"]))]
    longitudinal = [key for key in shared if abs(
        p4[key]["native_action"][1] - graph[key]["native_action"][1]) > 1e-7]
    applied = [key for key in shared if p4[key]["applied"] != graph[key]["applied"]]
    proposed = [key for key in shared if p4[key]["proposed"] != graph[key]["proposed"]]
    first = longitudinal[0] if longitudinal else None
    result = dict(
        scenario_identical=json_file(args.p4 / "spec.json") == json_file(args.graph / "spec.json"),
        shared_vehicle_steps=len(shared),
        proposed_command_differences=len(proposed),
        applied_command_differences=len(applied),
        native_action_differences=len(physical),
        longitudinal_action_differences=len(longitudinal),
        first_applied_difference=None if first is None else dict(
            ego=first[0], at_s=first[1],
            p4=p4[first]["applied"], graph=graph[first]["applied"],
            p4_native_action=p4[first]["native_action"],
            graph_native_action=graph[first]["native_action"],
        ),
        graph_rejections=sum(not row["guard"]["accepted"] for row in graph.values()),
        p4_rejections=sum(not row["guard"]["accepted"] for row in p4.values()),
        graph_summary=json_file(args.graph / "summary.json"),
        p4_summary=json_file(args.p4 / "summary.json"),
        source_hashes={name: file_hash(folder / "graph-marl-manifest.json")
                       for name, folder in (("p4", args.p4), ("graph", args.graph))},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result["paired_graph"] = compare(args.nominal, args.graph,
                                     args.output.parent / "paired-graph.json")
    result["paired_p4"] = compare(args.nominal, args.p4,
                                  args.output.parent / "paired-p4.json")
    atomic_json(args.output, result)
    print(dict(scenario_identical=result["scenario_identical"],
               applied_differences=len(physical), first=first,
               graph_gain=result["paired_graph"]["improvement_not_spontaneous"],
               p4_gain=result["paired_p4"]["improvement_not_spontaneous"]))


if __name__ == "__main__":
    main()
