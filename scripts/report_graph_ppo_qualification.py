"""Produit la table privée des scènes qualifiées, sans accéder au test réservé."""

import argparse
import csv
import json
from pathlib import Path

from cav_recovery.laboratory.storage import json_file


FIELDS = (
    "split", "source_id", "seed", "parent", "topology", "horizon_s",
    "vehicles_configured", "vehicles_introduced", "peak_active",
    "affected_count", "affected_ids", "resources", "constraint_intervals",
    "source_label", "qualified_label", "status", "arrivals_nominal",
    "errors", "wall_s", "cpu_s", "graph_decisions", "graph_useful_choices",
    "p4_recovery_current", "recoverability_status",
)


def probe_counts(root, identifier):
    folder = root / "probes" / identifier
    summary_path = folder / "probe-summary.json"
    graph = json_file(summary_path) if summary_path.exists() else None
    if graph and graph["decisions"] == 0:
        graph = None
    if graph:
        return graph["decisions"], graph["useful_choices"]
    for path in sorted(folder.glob("random-graph*/neural-decisions.jsonl"),
                       reverse=True):
        with path.open(encoding="utf-8") as stream:
            choices = [json.loads(line) for line in stream]
        return len(choices), sum(row["admissible_count"] > 1 for row in choices)
    return "", ""


def recoverability(root, identifier):
    if identifier not in {"p4-oblique-4-21021", "p4-oblique-8-21021"}:
        return "", "NON_DEMONTREE_SUR_CODE_ACTUEL"
    vehicle_count = 4 if identifier == "p4-oblique-4-21021" else 8
    path = root / f"p4-control-{vehicle_count}-paired.json"
    if not path.exists():
        return "", "NON_DEMONTREE_SUR_CODE_ACTUEL"
    paired = json_file(path)
    proven = paired["improvement_not_spontaneous"]
    return proven, ("RECUPERATION_DURABLE_DEMONTREE" if proven
                    else "RECUPERATION_NON_DEMONTREE")


def scene_row(root, item):
    scenario = item["scenario"]
    summary = item["summary"]
    identifier = item["source_id"]
    graph_decisions, graph_useful = probe_counts(root, identifier)
    p4_recovery, recovery_status = recoverability(root, identifier)
    if item["qualified_label"] != "POSITIVE_CONFIRMED":
        recovery_status = "SANS_OBSTRUCTION_QUALIFIANTE"
    return dict(
        split=item["split"], source_id=identifier, seed=scenario["seed"],
        parent=item["source_parent"], topology=item["source_family"],
        horizon_s=scenario["horizon_s"],
        vehicles_configured=scenario["vehicles"],
        vehicles_introduced=summary["introduced"],
        peak_active=summary["peak_active"],
        affected_count=len(item["affected"]),
        affected_ids=" ".join(item["affected"]),
        resources=" ".join(item["resources"]),
        constraint_intervals="; ".join(
            f"{row['requester']}:{row['start_s']:.1f}-{row['end_s']:.1f}"
            for row in item["intervals"]),
        source_label=item["source_label"],
        qualified_label=item["qualified_label"],
        status="VALID" if item["status"]["valid"] else "INVALID",
        arrivals_nominal=len(summary["arrivals"]),
        errors="; ".join(map(str, summary["errors"])),
        wall_s=round(summary["wall_s"], 3),
        cpu_s=round(summary["cpu_s"], 3),
        graph_decisions=graph_decisions,
        graph_useful_choices=graph_useful,
        p4_recovery_current=p4_recovery,
        recoverability_status=recovery_status,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    args = parser.parse_args()
    root = args.qualification.resolve()
    rows = [scene_row(root, item) for item in
            json_file(root / "qualification.json")["scenes"]]
    output = root / "SCENES_QUALIFIEES.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(output, len(rows))


if __name__ == "__main__":
    main()
