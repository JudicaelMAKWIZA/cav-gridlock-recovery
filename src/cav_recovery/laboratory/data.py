"""Cache causal reprenable : un fragment par épisode, sans réécrire M02."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import itertools
import os
from pathlib import Path
import time

import numpy as np
import orjson

from .features import BASE_FEATURES, vectorize, received_ages
from .storage import (
    atomic_json,
    file_hash,
    identity,
    json_file,
    safe_destination,
    single_writer,
)


def rows(path):
    with gzip.open(path, "rb") as stream:
        for line in stream:
            yield orjson.loads(line)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def index_windows(exported, output):
    """Conserve les cibles et fenêtres gelées ; aucun réétiquetage n'est effectué."""
    exported, output = Path(exported), Path(output)
    (output / "indices").mkdir(exist_ok=True)
    for partition in ["train", "validation", "test_ood"]:
        source = exported / partition
        sentinel = output / f"indices-{partition}.json"
        fingerprints = {
            name: file_hash(source / (name + ".jsonl.gz"))
            for name in ["tcn_windows", "temporal_targets"]
        }
        expected = json_file(exported / "manifest.json")["files"]
        for name, actual in fingerprints.items():
            assert actual == expected[f"{partition}/{name}.jsonl.gz"]
        if sentinel.exists():
            assert json_file(sentinel)["source_hashes"] == fingerprints
            continue
        windows = rows(source / "tcn_windows.jsonl.gz")
        targets = rows(source / "temporal_targets.jsonl.gz")
        episode = None
        records = []
        count = 0

        def flush():
            if episode is None:
                return
            atomic_npz(
                output / "indices" / (episode + ".npz"),
                end_sample=np.array([r[0] for r in records], dtype=np.int64),
                first_sample=np.array([r[1] for r in records], dtype=np.int64),
                sample_count=np.array([r[2] for r in records], dtype=np.int16),
                y=np.array([r[3] for r in records], dtype=np.int8),
                available=np.array([r[4] for r in records], dtype=bool),
                time_s=np.array([r[5] for r in records], dtype=np.float64),
                vehicle=np.array([r[6] for r in records]),
                window_id=np.array([r[7] for r in records]),
            )

        for window, target in itertools.zip_longest(windows, targets):
            if (
                window is None
                or target is None
                or window["window_id"] != target["window_id"]
            ):
                raise ValueError("Cibles et fenêtres désalignées.")
            if window["episode_id"] != episode:
                flush()
                episode = window["episode_id"]
                records = []
            indices = window["sample_indices"]
            if (
                not window["causal"]
                or window["expected_samples"] != 176
                or len(indices) + window["left_padding_samples"] != 176
                or any(a >= b for a, b in zip(indices, indices[1:]))
            ):
                raise ValueError("Fenêtre causale invalide.")
            records.append(
                (
                    indices[-1],
                    indices[0],
                    len(indices),
                    bool(target["persistent_confirmation_available_at_end"]),
                    target["supervision_available"],
                    window["end_s"],
                    window["vehicle_id"],
                    window["window_id"],
                )
            )
            count += 1
        flush()
        atomic_json(sentinel, dict(source_hashes=fingerprints, windows=count))
        print(f"Index {partition} : {count} fenêtres gelées.", flush=True)


def prepare_episode(arguments):
    source, output, episode, partition, code_id = arguments
    source, output = Path(source) / "episodes" / episode, Path(output)
    target = output / "shards" / (episode + ".npz")
    receipt = target.with_suffix(".json")
    source_manifest = file_hash(source / "manifest.json")
    if receipt.exists():
        old = json_file(receipt)
        if (
            old["code_id"] == code_id
            and old["source_manifest_sha256"] == source_manifest
            and file_hash(target) == old["cache_sha256"]
        ):
            return old
        raise ValueError("Un cache antérieur incompatible doit rester séparé.")
    manifest = json_file(source / "manifest.json")
    checked = {}
    for filename in ["observations.jsonl.gz", "commands.jsonl.gz"]:
        actual = file_hash(source / filename)
        assert actual == manifest["files"][filename], (episode, filename)
        checked[filename] = actual
    indices = np.load(output / "indices" / (episode + ".npz"), allow_pickle=False)
    by_vehicle = defaultdict(list)
    lookup = {}
    previous = {}
    commands = iter(rows(source / "commands.jsonl.gz"))
    command_frame = next(commands)
    for sample_index, row in enumerate(rows(source / "observations.jsonl.gz")):
        at = row["time_s"]
        while command_frame["time_s"] < at - 1e-7:
            command_frame = next(commands)
        if abs(command_frame["time_s"] - at) > 1e-7:
            raise ValueError("La commande propre n’est pas alignée avec la mesure.")
        ego = row["vehicle_id"]
        controls = command_frame["commands"][ego]
        state = dict(
            observed_at_s=at,
            heading_rad=row["heading_rad"],
            signed_speed_mps=row["signed_speed_mps"],
            acceleration_xy_mps2=row["acceleration_xy_mps2"],
            mission_progress_m=row["mission_progress_m"],
        )
        values = vectorize(
            state,
            controls,
            row["lidar_normalized"],
            received_ages(row["received_v2v"], at),
            previous.get(ego),
            lidar_available=not row["missing_mask"]["lidar"],
            v2v_available=not row["missing_mask"]["v2v"],
        )
        lookup[sample_index] = (ego, len(by_vehicle[ego]), at)
        by_vehicle[ego].append(values)
        previous[ego] = state
    offsets = {}
    values = []
    offset = 0
    for ego in sorted(by_vehicle):
        offsets[ego] = offset
        own = np.stack(by_vehicle[ego])
        values.append(own)
        offset += len(own)
    starts, ends = [], []
    for first, last, count, ego, at in zip(
        indices["first_sample"],
        indices["end_sample"],
        indices["sample_count"],
        indices["vehicle"],
        indices["time_s"],
    ):
        first_ego, first_local, first_time = lookup[int(first)]
        last_ego, last_local, last_time = lookup[int(last)]
        if (
            first_ego != ego
            or last_ego != ego
            or abs(at - last_time) > 1e-7
            or last_local - first_local + 1 != count
        ):
            raise ValueError(
                "Les fenêtres natives ne correspondent pas au cache causal."
            )
        if abs(last_time - first_time - (count - 1) * 0.2) > 1e-6:
            raise ValueError("Une lacune temporelle doit être explicitement masquée.")
        starts.append(offsets[ego] + first_local)
        ends.append(offsets[ego] + last_local)
    atomic_npz(
        target,
        x=np.concatenate(values),
        start=np.array(starts, dtype=np.int64),
        end=np.array(ends, dtype=np.int64),
        y=indices["y"],
        available=indices["available"],
        time_s=indices["time_s"],
        vehicle=indices["vehicle"],
        window_id=indices["window_id"],
    )
    result = dict(
        episode=episode,
        partition=partition,
        code_id=code_id,
        source_manifest_sha256=source_manifest,
        source_files=checked,
        cache_sha256=file_hash(target),
        observations=offset,
        windows=len(starts),
        supervised=int(indices["available"].sum()),
        positives=int(np.sum(indices["y"] & indices["available"])),
    )
    atomic_json(receipt, result)
    return result


def consolidate(output, episodes, provenance):
    output = Path(output)
    for partition in ["train", "validation", "test_ood"]:
        members = [row for row in episodes if row["partition"] == partition]
        count = sum(row["observations"] for row in members)
        x = np.lib.format.open_memmap(
            output / f"{partition}-x.npy.tmp",
            mode="w+",
            dtype=np.float32,
            shape=(count, len(BASE_FEATURES)),
        )
        endpoint_rows = defaultdict(list)
        offset = 0
        for number, row in enumerate(members):
            with np.load(
                output / "shards" / (row["episode"] + ".npz"), allow_pickle=False
            ) as shard:
                n = len(shard["x"])
                x[offset : offset + n] = shard["x"]
                for key in ["start", "end"]:
                    endpoint_rows[key].append(shard[key] + offset)
                for key in ["y", "available", "time_s"]:
                    endpoint_rows[key].append(shard[key])
                endpoint_rows["episode"].append(
                    np.full(len(shard["y"]), number, dtype=np.int32)
                )
                offset += n
        x.flush()
        del x
        os.replace(output / f"{partition}-x.npy.tmp", output / f"{partition}-x.npy")
        atomic_npz(
            output / f"{partition}-windows.npz",
            **{k: np.concatenate(v) for k, v in endpoint_rows.items()},
        )
        atomic_json(output / f"{partition}-episodes.json", members)
    # Ajustement uniquement sur les observations propres de train, jamais sur le test.
    x = np.load(output / "train-x.npy", mmap_mode="r")
    sums = np.zeros(x.shape[1], np.float64)
    squares = sums.copy()
    for begin in range(0, len(x), 100000):
        block = np.asarray(x[begin : begin + 100000], dtype=np.float64)
        sums += block.sum(axis=0)
        squares += (block * block).sum(axis=0)
    mean = sums / len(x)
    std = np.sqrt(np.maximum(squares / len(x) - mean * mean, 0.0))
    std = np.where(std < 1e-6, 1.0, std)
    atomic_json(
        output / "normalization.json",
        dict(
            mean=mean.tolist(),
            std=std.tolist(),
            fitted_partition="train",
            observations=len(x),
            features=list(BASE_FEATURES),
        ),
    )
    files = {p.name: file_hash(p) for p in sorted(output.glob("*.np*"))}
    files["normalization.json"] = file_hash(output / "normalization.json")
    atomic_json(
        output / "complete.json",
        dict(
            **provenance,
            files=files,
            features=list(BASE_FEATURES),
            target="persistent_confirmation_available_at_end",
            supervision_masked=True,
            oracle_as_input=False,
            episodes=episodes,
        ),
    )


def prepare_m02(dataset, output, workers=2):
    dataset, output = Path(dataset).resolve(), safe_destination(dataset, output)
    exported = dataset / "export-final"
    production = dataset / "production-final"
    output.mkdir(parents=True, exist_ok=True)
    before = {
        str(p.relative_to(dataset)): file_hash(p)
        for p in [
            exported / "manifest.json",
            production / "splits.json",
            production / "campaign.json",
        ]
    }
    source_id = identity(before)
    code_id = identity(
        {
            p.name: file_hash(p)
            for p in [Path(__file__), Path(__file__).with_name("features.py")]
        }
    )
    with single_writer(output):
        if (output / "complete.json").exists():
            result = json_file(output / "complete.json")
            assert result["source_id"] == source_id and result["code_id"] == code_id
            return result
        index_windows(exported, output)
        assignments = []
        for partition in ["train", "validation", "test_ood"]:
            assignments += [
                (str(production), str(output), row["episode_id"], partition, code_id)
                for row in rows(exported / partition / "episodes.jsonl.gz")
            ]
        started = time.perf_counter()
        receipts = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for count, receipt in enumerate(
                pool.map(prepare_episode, assignments, chunksize=1), 1
            ):
                receipts.append(receipt)
                atomic_json(
                    output / "progress.json",
                    dict(
                        completed=count,
                        total=len(assignments),
                        observations=sum(r["observations"] for r in receipts),
                        wall_s=time.perf_counter() - started,
                    ),
                )
                if count % 25 == 0:
                    print(
                        f"Cache M02 : {count}/{len(assignments)} épisodes.", flush=True
                    )
        consolidate(
            output,
            receipts,
            dict(
                source_id=source_id,
                code_id=code_id,
                source_files=before,
                wall_s=time.perf_counter() - started,
            ),
        )
        assert all(file_hash(dataset / k) == v for k, v in before.items())
    return json_file(output / "complete.json")
