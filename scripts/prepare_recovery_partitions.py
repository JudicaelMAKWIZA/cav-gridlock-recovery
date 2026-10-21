"""Réserve des parents disjoints, sans simulation ni lecture du test final."""

import argparse
import json
from pathlib import Path

from cav_recovery.dataset.contract import digest
from cav_recovery.dataset.generation import candidate
from cav_recovery.laboratory.storage import atomic_json


def reservation():
    """Garde ensemble les quatre variantes d'une géométrie et de son flux parent.

    Les indices élevés constituent de nouvelles recettes de récupération.
    L'asymétrique reste réservé au test de récupération, pas nécessairement
    inconnu des détecteurs historiques déjà entraînés sur M02.
    """
    groups, used = [], set()
    parent = 25000
    for partition, count in [("train", 8), ("validation", 4), ("test", 4)]:
        selected = 0
        while selected < count:
            first = candidate(parent * 4)
            parent += 1
            allowed = (first["family"] == "asymmetric") == (partition == "test")
            if not allowed or first["lineage_group"] in used:
                continue
            used.add(first["lineage_group"])
            recipes = []
            for variant in range(4):
                index = (parent - 1) * 4 + variant
                spec = candidate(index)
                recipes.append(
                    dict(
                        index=index,
                        phase="production",
                        seed=spec["seed"],
                        configuration_sha256=digest(spec),
                    )
                )
            groups.append(
                dict(
                    partition=partition,
                    parent_id=first["parent_id"],
                    lineage_group=first["lineage_group"],
                    family=first["family"],
                    active_limit=first["active_limit"],
                    recipes=recipes,
                )
            )
            selected += 1
    return dict(
        version="REC-SPLIT-1",
        groups=groups,
        simulated=False,
        evaluation_performed=False,
        unit="whole_parent_all_four_variants",
        limitation="topology_held_out_for_recovery_not_for_historical_detection",
        geometry_and_flow_grouping="generator_parent_plus_conservative_lineage_bucket",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        if json.loads(args.output.read_text(encoding="utf-8")) != reservation():
            raise ValueError("La réservation existante diffère ; aucune réécriture.")
        print("Réservation identique, conservée.")
        return
    atomic_json(args.output, reservation())


if __name__ == "__main__":
    main()
