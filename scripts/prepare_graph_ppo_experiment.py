"""Fige une campagne privée à partir de scènes déjà qualifiées physiquement."""

import argparse
import copy
from pathlib import Path
import random
import subprocess
import sys

import torch  # Sous Windows, Torch doit précéder les imports MetaDrive.

from cav_recovery.laboratory.storage import atomic_json, file_hash, json_file
from cav_recovery.recovery.graph_marl_core import load_config, scene_plan


PROJECT = Path(__file__).resolve().parents[1]


def qualified_rows(source):
    """Retient des références nominales intactes et des parents distincts."""
    qualification = json_file(source / "qualification.json")["scenes"]
    shortlist = json_file(source / "candidate-shortlist.json")["splits"]
    indexed = {(row["split"], row["source_id"]): row for row in qualification}
    selected = {}
    for split in ("train", "validation"):
        groups = {"POSITIVE_CONFIRMED": [], "NEGATIVE_CONFIRMED": []}
        for candidate in shortlist[split]:
            result = indexed.get((split, candidate["source_id"]))
            if result is None or not result["status"]["valid"]:
                continue
            label = result["qualified_label"]
            if label not in groups:
                continue
            reference = source / "nominal" / split / candidate["source_id"]
            if (file_hash(reference / "physical-annotation.json")
                    != result["annotation_sha256"]
                    or file_hash(candidate["source_spec_path"])
                    != candidate["source_spec_sha256"]):
                raise ValueError(f"Empreinte source invalide : {candidate['source_id']}")
            row = copy.deepcopy(candidate)
            row["qualified_label"] = label
            row["qualified_affected"] = result["affected"]
            groups[label].append(row)
        selected[split] = groups
    return selected


def interleave(positives, negatives, seed):
    generator = random.Random(seed)
    positive, negative = list(positives), list(negatives)
    generator.shuffle(positive)
    generator.shuffle(negative)
    order = []
    while positive or negative:
        order.extend(positive[:2])
        del positive[:2]
        if negative:
            order.append(negative.pop(0))
    return order


def build_config(source, *, repetitions=5):
    groups = qualified_rows(source)
    positive_pool = groups["train"]["POSITIVE_CONFIRMED"]
    positive_pool.sort(key=lambda row: row.get("source_campaign") !=
                       "P4_development_control")
    train_positive = positive_pool[:20]
    train_negative = groups["train"]["NEGATIVE_CONFIRMED"][:10]
    validation_positive = groups["validation"]["POSITIVE_CONFIRMED"][:6]
    validation_negative = groups["validation"]["NEGATIVE_CONFIRMED"][:5]
    if (len(train_positive) < 12 or len(validation_positive) < 4
            or len(train_negative) < 4 or len(validation_negative) < 3):
        raise RuntimeError("Diversité physique insuffisante : TRAIN_NOT_READY.")
    train_parents = {row["source_parent"] for row in train_positive + train_negative}
    validation_parents = {row["source_parent"] for row in
                          validation_positive + validation_negative}
    if train_parents & validation_parents:
        raise RuntimeError("Fuite de parent entre entraînement et validation.")
    nominal_cost = {(entry["split"], entry["source_id"]): entry["summary"]["wall_s"]
                    for entry in json_file(source / "qualification.json")["scenes"]}
    periodic_positive = min(validation_positive,
                            key=lambda row: (nominal_cost["validation", row["source_id"]],
                                             row["source_id"]))
    periodic_negative = min(validation_negative,
                            key=lambda row: (nominal_cost["validation", row["source_id"]],
                                             row["source_id"]))
    config = json_file(PROJECT / "configs/recovery/graph-marl-p4-v1.json")
    config.update(
        version="REC-GRAPH-PPO-P4-3",
        scope="development_qualified_no_reserved_test",
        algorithm="shared_local_graph_option_PPO_P4",
        qualification_root=str(source),
        maximum_wall_minutes=360,
        ppo_batch_size=12,
        minibatch_size=6,
        gae_lambda=0.995,
        validation_every_episodes=12,
        minimum_available_gib_eight_cav=2.0,
        light_training_trace=True,
        detailed_trace_rule="preflight_and_training_episode_zero_only",
        periodic_validation_ids=[periodic_positive["source_id"],
                                 periodic_negative["source_id"]],
        priority_validation_ids=[
            "ad1ba7cbf83eea435c26cf7b",  # T positif, validation périodique.
            "4b44e8aedd659cece99a84e5",  # Oblique positif.
            "e87ea0d593e625c709c8e4b7",  # Oblique négatif, périodique.
            "635fa94e0f506e80137f345b",  # Quatre approches négatif.
        ],
        model_selection_rule="periodic_gains_then_arrivals_then_earliest",
    )
    config["reward"] = dict(
        progress_per_vehicle_length=0.2,
        confirmed_passage=1.0,
        arrival=2.0,
        collective_recovery=10.0,
        per_step_cost=0.002,
        reverse_per_m=0.2,
        idle_after_10s_per_step=0.001,
        invalid_event=-100.0,
    )
    train = []
    for repeat in range(repetitions):
        ordered = interleave(train_positive, train_negative, 231023 + repeat)
        if repeat == 0:
            # Les deux témoins récupérables apparaissent avant l'épuisement
            # possible du budget, sans supprimer les autres configurations.
            controls = [row for key in ("p4-oblique-4-21021", "p4-oblique-8-21021")
                        for row in ordered if row["source_id"] == key]
            ordered = controls + [row for row in ordered if row not in controls]
        train.extend(copy.deepcopy(ordered))
    preflight_row = next(
        (row for row in train_positive
         if row["source_id"] == "p4-oblique-4-21021"), None,
    )
    if preflight_row is None:
        raise RuntimeError("Contrôle préliminaire P4 à quatre CAV absent.")
    config["scenes"] = dict(
        preflight=[copy.deepcopy(preflight_row)],
        train=train,
        validation=copy.deepcopy(validation_positive + validation_negative),
    )
    return config, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, root = args.qualification.resolve(), args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Campagne déjà présente : {root}")
    config, groups = build_config(source)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "experiment-config.json", config)
    load_config(root / "experiment-config.json")
    atomic_json(root / "scene-plan.json", scene_plan(config))
    files = sorted((PROJECT / "src/cav_recovery").rglob("*.py"))
    files += sorted((PROJECT / "scripts").glob("*.py"))
    files += sorted((PROJECT / "tests").rglob("*.py"))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=PROJECT, text=True).strip()
    atomic_json(root / "source-manifest.json", dict(
        git_commit=commit, python=sys.version, torch=torch.__version__,
        files={path.relative_to(PROJECT).as_posix(): file_hash(path)
               for path in files},
    ))
    atomic_json(root / "qualification-selection.json", dict(
        qualification_sha256=file_hash(source / "qualification.json"),
        shortlist_sha256=file_hash(source / "candidate-shortlist.json"),
        unique_train_positive=len(groups["train"]["POSITIVE_CONFIRMED"][:20]),
        unique_train_negative=len(groups["train"]["NEGATIVE_CONFIRMED"][:10]),
        unique_validation_positive=len(groups["validation"]["POSITIVE_CONFIRMED"][:6]),
        unique_validation_negative=len(groups["validation"]["NEGATIVE_CONFIRMED"][:5]),
        training_episodes=len(config["scenes"]["train"]),
        selected={split: {label: sorted({row["source_id"] for row in
                                        config["scenes"][split]
                                        if row["qualified_label"] == label})
                          for label in ("POSITIVE_CONFIRMED", "NEGATIVE_CONFIRMED")}
                  for split in ("train", "validation")},
    ))
    print(root)


if __name__ == "__main__":
    main()
