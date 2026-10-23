"""Présélectionne des scènes M02 sans lire le test réservé.

Les labels M02 servent seulement à choisir des candidats. Chaque candidat
doit ensuite être requalifié dans l'environnement de récupération natif.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

from cav_recovery.laboratory.storage import atomic_json, file_hash


TARGETS = {
    "train": {"POSITIVE_CONFIRMED": 15, "NEGATIVE_CONFIRMED": 5},
    "validation": {"POSITIVE_CONFIRMED": 6, "NEGATIVE_CONFIRMED": 4},
}


def candidates(m02_root):
    """Parcourt uniquement les identités de développement et de validation M02."""
    split_map = json.loads((m02_root / "splits.json").read_text(encoding="utf-8"))["episodes"]
    rows = defaultdict(list)
    for identity, split in sorted(split_map.items()):
        if split not in TARGETS:
            continue
        folder = m02_root / "episodes" / identity
        spec_path = folder / "spec.json"
        annotation_path = folder / "annotation.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec["active_limit"] != 8:
            continue
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        label = annotation["label"]
        if label not in TARGETS[split]:
            continue
        index = spec["seed"] - 1000
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["files"]["spec.json"] != file_hash(spec_path)
                or manifest["files"]["annotation.json"] != file_hash(annotation_path)):
            raise ValueError(f"Empreinte M02 incohérente : {identity}")
        confirmed = [event for event in annotation["events"] if event.get("confirmed")]
        rows[split, label].append(dict(
            name=f"m02-{index}", m02_index=index, source_id=identity,
            seed=spec["seed"], vehicles=8, horizon_s=210.0,
            source_parent=spec["parent_id"], source_family=spec["family"],
            source_label=label, source_confirmed_events=len(confirmed),
            source_affected=len({event["requester"] for event in confirmed}),
            source_spec_sha256=file_hash(spec_path),
            source_annotation_sha256=file_hash(annotation_path),
        ))
    return rows


def select(rows, targets=TARGETS):
    """Choisit des parents distincts et alterne les familles sans score de P4."""
    selected = {}
    for split, split_targets in targets.items():
        selected[split] = []
        used_parents = set()
        for label, target in split_targets.items():
            pool = rows[split, label]
            families = sorted({row["source_family"] for row in pool})
            for family in families:
                family_rows = sorted(
                    (row for row in pool if row["source_family"] == family),
                    key=lambda row: (row["source_parent"], row["seed"]),
                )
                for row in family_rows:
                    if len([item for item in selected[split]
                            if item["source_label"] == label]) >= target:
                        break
                    if row["source_parent"] in used_parents:
                        continue
                    selected[split].append(row)
                    used_parents.add(row["source_parent"])
                    break
            for row in sorted(pool, key=lambda item: (item["source_parent"], item["seed"])):
                if len([item for item in selected[split]
                        if item["source_label"] == label]) >= target:
                    break
                if row["source_parent"] not in used_parents:
                    selected[split].append(row)
                    used_parents.add(row["source_parent"])
            found = sum(row["source_label"] == label for row in selected[split])
            if found < target:
                raise ValueError(f"Parents insuffisants : {split}/{label} {found}/{target}")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m02-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--train-positive", type=int, default=15)
    args = parser.parse_args()
    targets = {split: dict(labels) for split, labels in TARGETS.items()}
    targets["train"]["POSITIVE_CONFIRMED"] = args.train_positive
    result = select(candidates(args.m02_root), targets)
    args.staged_root.mkdir(parents=True, exist_ok=True)
    for entries in result.values():
        for row in entries:
            source = args.m02_root / "episodes" / row["source_id"] / "spec.json"
            target = args.staged_root / f"{row['source_id']}.json"
            if target.exists():
                if file_hash(target) != row["source_spec_sha256"]:
                    raise ValueError(f"Copie de scène déjà différente : {target}")
            else:
                shutil.copy2(source, target)
            row["source_spec_path"] = str(target)
    atomic_json(args.output, dict(
        source="M02 development/validation only; no reserved test opened",
        status="candidate_only_requires_native_requalification",
        splits=result,
    ))
    print({split: len(items) for split, items in result.items()})


if __name__ == "__main__":
    main()
