"""Résume les scènes de développement sans confondre obstruction et récupérabilité."""

import argparse
import csv
from pathlib import Path

import torch  # Garde l'ordre d'import Windows du projet.

from cav_recovery.laboratory.storage import atomic_json, json_file


def has_cycle(edges):
    """Détecte seulement un cycle parmi les dépendances confirmées."""
    neighbors = {}
    for requester, blocker in edges:
        neighbors.setdefault(requester, set()).add(blocker)
    visiting, visited = set(), set()

    def walk(node):
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        found = any(walk(peer) for peer in neighbors.get(node, ()))
        visiting.remove(node)
        visited.add(node)
        return found

    return any(walk(node) for node in neighbors)


def temporal_cycle(links):
    """N'affirme un cycle que si les liens sont confirmés simultanément."""
    boundaries = sorted({time for link in links
                         for time in (link["confirmation_at_s"], link["end_s"])
                         if time is not None})
    return any(has_cycle((link["requester"], link["blocker"])
                         for link in links
                         if link["confirmation_at_s"] is not None
                         and link["confirmation_at_s"] <= (left + right) / 2
                         < (link["end_s"] if link["end_s"] is not None else float("inf")))
               for left, right in zip(boundaries, boundaries[1:]))


def report(config_path, qualification_root, output):
    config = json_file(config_path)
    qualification = json_file(qualification_root / "qualification.json")["scenes"]
    indexed = {(row["split"], row["source_id"]): row for row in qualification}
    rows = []
    for split in ("train", "validation"):
        unique = {row["source_id"]: row for row in config["scenes"][split]}
        for identifier, selection in unique.items():
            entry = indexed[split, identifier]
            annotation = json_file(qualification_root / "nominal" / split /
                                   identifier / "physical-annotation.json")
            links = [link for link in annotation["dependencies"]
                     if link["relevance"] == "confirmed_for_requested_movement"]
            edges = [(link["requester"], link["blocker"]) for link in links]
            affected = set(entry["affected"])
            collective_evidence = (len(affected) >= 2 and
                                   any(source in affected and target in affected
                                       for source, target in edges))
            controls = {"p4-oblique-4-21021": "p4-control-4-paired.json",
                        "p4-oblique-8-21021": "p4-control-8-paired.json"}
            control_path = qualification_root / controls.get(identifier, "missing")
            p4_proven = (split == "train" and control_path.exists()
                         and json_file(control_path)["improvement_not_spontaneous"]
                         and json_file(control_path)["intervention"]["status"]
                         == "DURABLE_RECOVERY")
            failed_path = output.parent / f"{identifier}-p4-paired.json"
            p4_failed = (split == "train" and failed_path.exists()
                         and json_file(failed_path)["intervention"]["status"]
                         == "NOT_RECOVERED")
            graph_path = output.parent / f"{identifier}-graph-paired.json"
            graph_status = (json_file(graph_path)["intervention"]["status"]
                            if split == "train" and graph_path.exists() else
                            "NOT_TESTED")
            rows.append(dict(
                split=split, source_id=identifier,
                label=entry["qualified_label"],
                family=entry["source_family"], vehicles=selection["vehicles"],
                affected=len(affected), confirmed_events=entry["confirmed_events"],
                confirmed_direct_links=len(edges),
                confirmed_cycle=temporal_cycle(links),
                collective_dependency_evidence=collective_evidence,
                recovered_under_current_P4=("YES" if p4_proven else
                                            "NO_IN_OBSERVED_PROBE" if p4_failed
                                            else "UNPROVEN"),
                untrained_graph_probe_status=graph_status,
                recovery_interpretation=(
                    "DURABLE_RECOVERY_DEMONSTRATED" if p4_proven else
                    "NOT_RECOVERED_IN_EXECUTED_P4_PROBE" if p4_failed else
                    "RECOVERABILITY_NOT_DEMONSTRATED" if affected else
                    "NEGATIVE_NO_QUALIFYING_OBSTRUCTION"),
            ))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "SCENE_OPPORTUNITIES.csv").open("w", newline="",
                                                   encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(output / "SCENE_OPPORTUNITIES.json", rows)
    positive = [row for row in rows if row["label"] == "POSITIVE_CONFIRMED"]
    collective = [row for row in positive if row["collective_dependency_evidence"]]
    proven = sum(row["recovery_interpretation"] ==
                 "DURABLE_RECOVERY_DEMONSTRATED" for row in positive)
    failed = sum(row["recovery_interpretation"] ==
                 "NOT_RECOVERED_IN_EXECUTED_P4_PROBE" for row in positive)
    graph_failed = sum(row["untrained_graph_probe_status"] == "NOT_RECOVERED"
                       for row in positive)
    lines = ["# Opportunités physiques des scènes sélectionnées", "",
             f"{len(rows)} scènes uniques ; {len(positive)} positives ; "
             f"{len(collective)} positives avec au moins un lien confirmé "
             "entre véhicules affectés.",
             f"Cycles confirmés : {sum(row['confirmed_cycle'] for row in positive)}. "
             "Un cycle n'est pas requis pour qualifier l'obstruction.",
             f"Récupération P4 démontrée : {proven} contrôles de développement. "
             f"{failed} autres positifs sondés restent "
             "non récupérés dans ces essais ; cela ne prouve pas leur impossibilité. "
             "Les autres positifs restent non démontrés.",
             f"Réseau Graph non entraîné : {graph_failed} positif(s) sondé(s) "
             "sans récupération durable ; aucune conclusion d'apprentissage.",
             "Les annotations nominales documentent la contrainte ; elles ne "
             "prouvent ni qu'une action sûre existe ni que P4 réussira.",
             "Aucune scène du test réservé n'a été lue."]
    (output / "SCENE_OPPORTUNITIES.md").write_text(
        "\n\n".join(lines) + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(len(report(args.config, args.qualification, args.output)))
