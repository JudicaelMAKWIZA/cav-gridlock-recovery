"""Sous-étude exploratoire appariée sur les traces intégrées déjà enregistrées.

Les variantes d'une même recette restent groupées. Cette collection choisie
pour des démonstrations ne devient pas un test indépendant de généralisation.
"""

from collections import defaultdict
from pathlib import Path
import re

import numpy as np

from .data import rows, atomic_npz
from .features import (
    BASE_FEATURES,
    DIAGNOSTIC_FEATURES,
    GRAPH_FEATURES,
    vectorize,
    augmentation,
)
from .storage import atomic_json, file_hash, json_file, identity, single_writer


def target_at(annotation, ego, at_s):
    current = [
        e
        for e in annotation["events"]
        if e["requester"] == ego and e["start_s"] - 1e-7 <= at_s <= e["end_s"] + 1e-7
    ]
    positive = any(
        e["confirmed"]
        and any(
            p["confirmed"]
            and p["start_s"] - 1e-7 <= at_s <= p["end_s"] + 1e-7
            and p["confirmation_at_s"] <= at_s + 1e-7
            for p in e["effect_intervals"]
        )
        for e in current
    )
    ambiguous = not positive and any(e["status"] == "open_candidate" for e in current)
    return int(positive), not ambiguous


def prepare_integrated(source, output, seed=1741):
    source, output = Path(source), Path(output)
    with single_writer(output):
        if (output / "complete.json").exists():
            return json_file(output / "complete.json")
        selection = json_file(source / "demonstration-selection.json")
        names = selection["all_evaluated_names"]
        parents = {name: re.match(r"(.+-\d+)", name)[1] for name in names}
        groups = sorted(set(parents.values()))
        # Stratification des parents sur les classes physiques déjà établies,
        # jamais sur les prédictions d'un modèle. Une variante positive suffit
        # à placer le parent dans la strate mixte/positive ; il reste indivisible.
        strata = {group: False for group in groups}
        for name in names:
            folder = next(
                root / name
                for root in [
                    source / "campaign-validated",
                    source / "supplementary-demonstrations",
                ]
                if (root / name / "complete.json").exists()
            )
            strata[parents[name]] |= (
                json_file(folder / "evaluation/results.json")["reference_label"]
                == "POSITIVE_CONFIRMED"
            )
        generator = np.random.default_rng(seed)
        assignments = {}
        for value in [False, True]:
            candidates = generator.permutation(
                sorted(g for g in groups if strata[g] == value)
            ).tolist()
            held = max(1, round(len(candidates) * 0.175))
            for i, group in enumerate(candidates):
                assignments[group] = (
                    "validation"
                    if i < held
                    else "test_ood"
                    if i < 2 * held
                    else "train"
                )
        assert len(groups) == 23
        atomic_json(
            output / "split.json",
            dict(
                seed=seed,
                groups=assignments,
                stratification="frozen physical reference classes at parent level; no model predictions",
                grouping="parent recipe; all suffix variants indivisible",
                test_ood_name_is_storage_only=True,
                test_meaning="held-out exploratory demonstration groups, not topology OOD nor reserved M02 test",
            ),
        )
        feature_names = [*BASE_FEATURES, *DIAGNOSTIC_FEATURES, *GRAPH_FEATURES]
        buffers = {
            p: dict(
                x=[],
                start=[],
                end=[],
                y=[],
                available=[],
                time_s=[],
                episode=[],
                d01_A=[],
                d01_B=[],
                abstain_A=[],
                abstain_B=[],
            )
            for p in ["train", "validation", "test_ood"]
        }
        offsets = defaultdict(int)
        members = defaultdict(list)
        source_files = {}
        for name in names:
            folder = next(
                root / name
                for root in [
                    source / "campaign-validated",
                    source / "supplementary-demonstrations",
                ]
                if (root / name / "complete.json").exists()
            )
            complete = json_file(folder / "complete.json")
            feature_file = folder / "diagnostic/features.jsonl.gz"
            actual = file_hash(feature_file)
            assert actual == complete["files"]["diagnostic/features.jsonl.gz"]
            reference = folder / "evaluation/physical-reference.json"
            annotation = json_file(reference)["annotation"]
            source_files[name] = dict(
                features_sha256=actual,
                reference_sha256=file_hash(reference),
                complete_sha256=file_hash(folder / "complete.json"),
            )
            sampled_diagnostics = {}
            decision_path = folder / "diagnostic/decisions.jsonl.gz"
            assert (
                file_hash(decision_path)
                == complete["files"]["diagnostic/decisions.jsonl.gz"]
            )
            for frame in rows(decision_path):
                at = frame["at_s"]
                if abs(at / 2 - round(at / 2)) > 1e-7:
                    continue
                for ego, record in frame["vehicles"].items():
                    sampled_diagnostics[(ego, at)] = tuple(
                        record[mode]["status"] for mode in ["A", "B"]
                    )
            partition = assignments[parents[name]]
            data = buffers[partition]
            episode_number = len(members[partition])
            traces = defaultdict(list)
            previous = {}
            start_time = {}
            for row in rows(feature_file):
                ego = row["ego_id"]
                at = row["observed_at_s"]
                if row["available_at_s"] > at + 1e-7:
                    raise ValueError("Export causal futur.")
                state = row["own_state"]
                controls = dict(
                    requested_speed_mps=row["requested_speed_mps"],
                    accepted_speed_mps=row["accepted_speed_mps"],
                    native_action=row["own_native_action"],
                )
                features = vectorize(
                    state,
                    controls,
                    row["lidar"],
                    row["received_message_ages_s"],
                    previous.get(ego),
                    lidar_available=row["lidar_available"],
                    v2v_available=row["v2v_box_available"],
                )
                traces[ego].append((at, np.concatenate([features, augmentation(row)])))
                previous[ego] = state
                start_time.setdefault(ego, at)
            for ego, sequence in sorted(traces.items()):
                base = offsets[partition]
                data["x"].append(np.stack([v for _, v in sequence]))
                offsets[partition] += len(sequence)
                for i, (at, _) in enumerate(sequence):
                    # Tous les comparateurs sont évalués aux mêmes rendez-vous
                    # de preuve (grille globale de 2 s), avec au moins 5 s propres.
                    if (
                        at - start_time[ego] < 5.0 - 1e-7
                        or abs(at / 2 - round(at / 2)) > 1e-7
                    ):
                        continue
                    begin = max(0, i - 175)
                    if abs(at - sequence[begin][0] - (i - begin) * 0.2) > 1e-6:
                        raise ValueError("Lacune temporelle non masquée.")
                    target, available = target_at(annotation, ego, at)
                    data["start"].append(base + begin)
                    data["end"].append(base + i)
                    data["y"].append(target)
                    data["available"].append(available)
                    data["time_s"].append(at)
                    data["episode"].append(episode_number)
                    a, b = sampled_diagnostics[(ego, at)]
                    data["d01_A"].append(a == "CONFIRMED_OBSTRUCTION")
                    data["d01_B"].append(b == "CONFIRMED_OBSTRUCTION")
                    data["abstain_A"].append(a == "UNKNOWN")
                    data["abstain_B"].append(b == "UNKNOWN")
            members[partition].append(
                dict(
                    episode=name,
                    parent=parents[name],
                    partition=partition,
                    topology=json_file(folder / "spec.json")["family"],
                    seed=complete["seed"],
                )
            )
        for partition, data in buffers.items():
            x = np.concatenate(data.pop("x")).astype(np.float32)
            np.save(output / f"{partition}-x.npy", x)
            atomic_npz(
                output / f"{partition}-windows.npz",
                **{
                    k: np.asarray(
                        v,
                        dtype=bool
                        if k == "available"
                        else np.float64
                        if k == "time_s"
                        else np.int64,
                    )
                    for k, v in data.items()
                },
            )
            atomic_json(output / f"{partition}-episodes.json", members[partition])
        x = np.load(output / "train-x.npy", mmap_mode="r")
        mean = np.mean(x, axis=0, dtype=np.float64)
        std = np.std(x, axis=0, dtype=np.float64)
        atomic_json(
            output / "normalization.json",
            dict(
                mean=mean.tolist(),
                std=np.where(std < 1e-6, 1.0, std).tolist(),
                features=feature_names,
                fitted_partition="train",
                observations=len(x),
            ),
        )
        result = dict(
            source_id=identity(source_files),
            source_files=source_files,
            features=feature_names,
            files={p.name: file_hash(p) for p in output.glob("*.np*")},
            target="persistent_confirmation_available_at_end",
            oracle_as_input=False,
            scope="exploratory_paired_demonstrations_not_independent_generalization",
            code_sha256=file_hash(Path(__file__)),
            grouped_partitions=assignments,
        )
        atomic_json(output / "complete.json", result)
    return result
