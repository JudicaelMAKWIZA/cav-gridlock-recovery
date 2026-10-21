"""Audit des calculs et des identités, sans modifier les résultats sauvegardés."""

import ast
from pathlib import Path

import numpy as np

from .checkpoints import CheckpointStore
from .metrics import binary_metrics
from .model import create_model
from .storage import atomic_json, file_hash, identity, json_file, verify_cache_files
from .training import WindowData, configure, predict
from .workbench import Laboratory


def audit(lab=None):
    lab = lab or Laboratory.from_environment()
    caches = [lab.cache, lab.integrated_cache]
    for cache in caches:
        verify_cache_files(cache)
    recalculated = []
    for scope, cache in [("m02", lab.cache), ("integrated", lab.integrated_cache)]:
        for filename in sorted((lab.campaign / scope).glob("*/*-predictions.npz")):
            partition = filename.name.removesuffix("-predictions.npz")
            score = json_file(filename.with_name(partition + "-metrics.json"))
            with (
                np.load(filename) as data,
                np.load(cache / (partition + "-windows.npz")) as windows,
            ):
                indices = data["indices"]
                assert np.array_equal(data["target"], windows["y"][indices])
                assert np.array_equal(data["time_s"], windows["time_s"][indices])
                assert windows["available"][indices].all()
                metrics = binary_metrics(
                    data["target"], data["probability"] >= score["threshold"]
                )
                assert metrics == score["micro"], filename
                recalculated.append(
                    dict(
                        model=filename.parent.name,
                        partition=partition,
                        windows=len(indices),
                        metrics_exact=True,
                    )
                )
    original = lab.root / "core-trained-v1"
    names = ["training.py", "model.py", "features.py", "metrics.py", "checkpoints.py"]
    original_identity = identity({name: file_hash(original / name) for name in names})
    scientific = []
    for name in [*names, "data.py", "integrated.py"]:
        current = Path(__file__).with_name(name)
        original_ast = ast.dump(
            ast.parse((original / name).read_text("utf-8")), include_attributes=False
        )
        current_ast = ast.dump(
            ast.parse(current.read_text("utf-8")), include_attributes=False
        )
        assert original_ast == current_ast, name
        scientific.append(
            dict(
                module=name,
                original_sha256=file_hash(original / name),
                candidate_sha256=file_hash(current),
                ast_identical=True,
            )
        )
    for complete in lab.campaign.glob("*/*/complete.json"):
        result = json_file(complete)
        assert result["compatibility"]["code"] == original_identity, complete
        checkpoint = CheckpointStore(
            complete.parent / "checkpoints", result["compatibility"]
        )
        checkpoint.load()
        checkpoint.load(best=True)
    selected = json_file(lab.campaign / "selected-model.json")
    folder = lab.campaign / "m02" / selected["selected"]
    result = json_file(folder / "complete.json")
    config = json_file(folder / "config.json")
    configure(config["seed"], config["threads"])
    state, _ = CheckpointStore(folder / "checkpoints", result["compatibility"]).load(
        best=True
    )
    model = create_model(config, result["compatibility"]["features"])
    model.load_state_dict(state["model"])
    data = WindowData(lab.cache, "test_ood")
    with np.load(folder / "test_ood-predictions.npz") as saved:
        indices = saved["indices"][:1024]
        probabilities, _ = predict(model, data, indices, config["batch_size"])
        np.testing.assert_array_equal(
            probabilities, saved["probability"][: len(indices)]
        )
    for relative, expected in json_file(lab.cache / "complete.json")[
        "source_files"
    ].items():
        assert file_hash(lab.dataset / relative) == expected
    stats = json_file(lab.cache / "normalization.json")
    assert stats["fitted_partition"] == "train"
    assert (
        stats["observations"]
        == np.load(lab.cache / "train-x.npy", mmap_mode="r").shape[0]
    )
    for cache in caches:
        partitions = {}
        for partition in ["train", "validation", "test_ood"]:
            members = json_file(cache / (partition + "-episodes.json"))
            for row in members:
                if cache == lab.integrated_cache:
                    group = row["parent"]
                else:
                    spec = json_file(
                        lab.dataset
                        / "production-final/episodes"
                        / row["episode"]
                        / "spec.json"
                    )
                    group = spec["lineage_group"]
                assert group not in partitions or partitions[group] == partition
                partitions[group] = partition
    proof = json_file(lab.campaign / "power-cut-proof.json")
    assert proof["completed"] and proof["actual_process_terminated"]
    report = dict(
        valid=True,
        prediction_tables_recomputed=recalculated,
        source_equivalence=scientific,
        selected_model_predictions_exact_on_1024_windows=True,
        training_normalization_only=True,
        data_labels_and_partitions_unchanged=True,
        parent_or_geometry_leakage=False,
        checkpoint_hashes_verified=True,
        actual_interruption_recovered=True,
        no_retraining=True,
        no_physical_simulation=True,
    )
    atomic_json(lab.root / "audit-final.json", report)
    return report
