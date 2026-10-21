"""Entraînement CPU reprenable à la frontière de chaque lot sauvegardé."""

import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .checkpoints import CheckpointStore, restore_random_state
from .metrics import binary_metrics, choose_threshold, grouped_metrics  # noqa: F401
from .model import create_model
from .storage import atomic_json, file_hash, identity, json_file, single_writer


class WindowData:
    """Matrice mémoire mappée ; le masque évite de confondre préfixe et mesure."""

    def __init__(self, cache, partition, feature_count=None):
        cache = Path(cache)
        self.x = np.load(cache / f"{partition}-x.npy", mmap_mode="r")
        with np.load(cache / f"{partition}-windows.npz", allow_pickle=False) as arrays:
            self.windows = {k: arrays[k] for k in arrays.files}
        self.normalization = json_file(cache / "normalization.json")
        self.features = feature_count or self.x.shape[1]
        self.mean = np.array(self.normalization["mean"][: self.features], np.float32)
        self.std = np.array(self.normalization["std"][: self.features], np.float32)

    def indices(self, stride=1):
        # L'échantillonnage est fixe et indépendant des cibles : un endpoint sur n
        # dans l'ordre gelé, identique lors de chaque reprise.
        indices = np.arange(len(self.windows["y"]), dtype=np.int64)[::stride]
        return indices[self.windows["available"][indices]]

    def batch(self, indices):
        batch = np.zeros((len(indices), 176, self.features), np.float32)
        masks = np.zeros((len(indices), 176), np.float32)
        for j, index in enumerate(indices):
            start, end = (
                int(self.windows["start"][index]),
                int(self.windows["end"][index]),
            )
            count = end - start + 1
            if not 1 <= count <= 176:
                raise ValueError("Fenêtre causale trop longue.")
            values = (self.x[start : end + 1, : self.features] - self.mean) / self.std
            batch[j, -count:] = np.clip(values, -10, 10)
            masks[j, -count:] = 1.0
        return (
            torch.from_numpy(batch.transpose(0, 2, 1).copy()),
            torch.from_numpy(masks[:, None, :]),
            torch.from_numpy(self.windows["y"][indices].astype(np.float32)),
        )


def predict(model, data, indices, batch_size=256):
    values = []
    loss_sum = 0.0
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(indices), batch_size):
            batch = indices[begin : begin + batch_size]
            x, mask, y = data.batch(batch)
            logits = model(x, mask)
            loss_sum += float(
                F.binary_cross_entropy_with_logits(logits, y, reduction="sum")
            )
            values.append(torch.sigmoid(logits).numpy())
    return np.concatenate(values) if values else np.array(
        [], np.float32
    ), loss_sum / max(1, len(indices))


def configure(seed, threads):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(
    cache,
    output,
    config,
    *,
    stop_after_steps=None,
    feature_count=None,
    abrupt_exit=False,
):
    cache, output = Path(cache), Path(output)
    configure(config["seed"], config["threads"])
    data = WindowData(cache, "train", feature_count)
    validation = WindowData(cache, "validation", feature_count)
    train_indices = data.indices(config["train_stride"])
    validation_indices = validation.indices(config["validation_selection_stride"])
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("Partition supervisée vide.")
    dataset_identity = file_hash(cache / "complete.json")
    code_identity = identity(
        {
            name: file_hash(Path(__file__).with_name(name))
            for name in [
                "training.py",
                "model.py",
                "features.py",
                "metrics.py",
                "checkpoints.py",
            ]
        }
    )
    compatibility = dict(
        dataset=dataset_identity,
        configuration=identity(config),
        code=code_identity,
        normalization=file_hash(cache / "normalization.json"),
        features=data.features,
    )
    with single_writer(output):
        if (output / "complete.json").exists():
            result = json_file(output / "complete.json")
            if result["compatibility"] != compatibility:
                raise ValueError("Résultat existant incompatible.")
            return result
        atomic_json(output / "config.json", config)
        atomic_json(output / "normalization.json", data.normalization)
        model = create_model(config, data.features)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        store = CheckpointStore(output / "checkpoints", compatibility)
        state, receipt = store.load()
        history = []
        epoch = cursor = step = 0
        best_loss = float("inf")
        bad_epochs = 0
        accumulated_loss = 0.0
        accumulated_count = 0
        if state is not None:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            restore_random_state(state["random_state"])
            epoch, cursor, step = state["epoch"], state["cursor"], state["global_step"]
            history = state["history"]
            best_loss = state["best_validation_loss"]
            bad_epochs = state["bad_epochs"]
            accumulated_loss = state["accumulated_loss"]
            accumulated_count = state["accumulated_count"]
            atomic_json(
                output / f"resume-{time.time_ns()}.json",
                dict(epoch=epoch, cursor=cursor, step=step, **receipt),
            )
        positives = int(data.windows["y"][train_indices].sum())
        if positives == 0 or positives == len(train_indices):
            raise ValueError("Les deux classes doivent être représentées dans train.")
        positive_weight = min(10.0, (len(train_indices) - positives) / positives)
        weight = torch.tensor(positive_weight, dtype=torch.float32)
        started = time.perf_counter()

        def save(best=False):
            return store.save(
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=None,
                    epoch=epoch,
                    cursor=cursor,
                    global_step=step,
                    history=history,
                    best_validation_loss=best_loss,
                    bad_epochs=bad_epochs,
                    accumulated_loss=accumulated_loss,
                    accumulated_count=accumulated_count,
                    config=config,
                    data_identity=dataset_identity,
                    normalization=data.normalization,
                    sampler=dict(
                        seed=config["seed"],
                        epoch=epoch,
                        cursor=cursor,
                        policy="permutation NumPy PCG64 seed+epoch; next unsaved batch only",
                    ),
                    extra_state=dict(
                        checkpoint_boundary="completed_batch",
                        physical_episode_state=None,
                    ),
                ),
                best=best,
            )

        for current_epoch in range(epoch, config["epochs"]):
            epoch = current_epoch
            permutation = np.random.default_rng(config["seed"] + epoch).permutation(
                train_indices
            )
            model.train()
            while cursor < len(permutation):
                indices = permutation[cursor : cursor + config["batch_size"]]
                x, mask, y = data.batch(indices)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x, mask)
                loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                cursor += len(indices)
                step += 1
                accumulated_loss += float(loss.detach()) * len(indices)
                accumulated_count += len(indices)
                if step % config["checkpoint_steps"] == 0:
                    save()
                    atomic_json(
                        output / "progress.json",
                        dict(
                            epoch=epoch,
                            cursor=cursor,
                            steps=step,
                            total_train_windows=len(train_indices),
                            wall_s=time.perf_counter() - started,
                        ),
                    )
                    print(
                        json.dumps(
                            dict(
                                output=output.name,
                                epoch=epoch,
                                step=step,
                                cursor=cursor,
                            )
                        ),
                        flush=True,
                    )
                if stop_after_steps is not None and step >= stop_after_steps:
                    save()
                    atomic_json(
                        output / "interrupted.json",
                        dict(intentional=True, epoch=epoch, cursor=cursor, step=step),
                    )
                    if abrupt_exit:
                        # Test explicite d'une coupure du processus : pas de finally,
                        # verrou abandonné, puis restauration automatique au relancement.
                        os._exit(86)
                    return dict(interrupted=True, step=step)
            _, validation_loss = predict(
                model, validation, validation_indices, config["batch_size"]
            )
            improved = validation_loss < best_loss
            if improved:
                best_loss = validation_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
            history.append(
                dict(
                    epoch=epoch + 1,
                    train_loss=accumulated_loss / accumulated_count,
                    validation_loss=validation_loss,
                    global_step=step,
                )
            )
            atomic_json(output / "history.json", history)
            print(json.dumps(dict(output=output.name, **history[-1])), flush=True)
            epoch += 1
            cursor = 0
            accumulated_loss = 0.0
            accumulated_count = 0
            save(best=improved)
            if bad_epochs >= config["patience"]:
                break
        result = dict(
            compatibility=compatibility,
            history=history,
            best_validation_loss=best_loss,
            train_windows=len(train_indices),
            validation_selection_windows=len(validation_indices),
            class_weight=positive_weight,
            parameters=sum(p.numel() for p in model.parameters()),
            wall_s=time.perf_counter() - started,
            completed_epochs=epoch,
            no_test_used=True,
            checkpoint_granularity="completed_batch",
            normalization_fitted_partition="train",
            torch_version=torch.__version__,
        )
        atomic_json(output / "complete.json", result)
        return result


def evaluate(
    cache, output, *, partitions=("validation", "test_ood"), feature_count=None
):
    """Le calibrage validation est publié avant toute lecture des cibles du test."""
    cache, output = Path(cache), Path(output)
    result = json_file(output / "complete.json")
    config = json_file(output / "config.json")
    configure(config["seed"], config["threads"])
    compatibility = result["compatibility"]
    if file_hash(cache / "complete.json") != compatibility["dataset"]:
        raise ValueError("Cache différent.")
    store = CheckpointStore(output / "checkpoints", compatibility)
    state, receipt = store.load(best=True)
    model = create_model(config, compatibility["features"])
    model.load_state_dict(state["model"])
    validation = WindowData(cache, "validation", feature_count)
    indices = validation.indices()
    probabilities, _ = predict(model, validation, indices, config["batch_size"])
    calibrated = choose_threshold(validation.windows["y"][indices], probabilities)
    atomic_json(
        output / "calibration.json", dict(**calibrated, best_checkpoint=receipt)
    )
    metrics = {}
    for partition in partitions:
        data = WindowData(cache, partition, feature_count)
        indices = data.indices()
        probabilities, loss = predict(model, data, indices, config["batch_size"])
        predictions = probabilities >= calibrated["threshold"]
        scores = grouped_metrics(
            data.windows["y"][indices], predictions, data.windows["episode"][indices]
        )
        # Les fenêtres non supervisées restent dénombrées, sans les compter négatives.
        scores["unsupervised_windows_excluded"] = int(
            (~data.windows["available"]).sum()
        )
        scores["loss"] = loss
        scores["threshold"] = calibrated["threshold"]
        from .data import atomic_npz

        atomic_npz(
            output / (partition + "-predictions.npz"),
            indices=indices,
            probability=probabilities,
            target=data.windows["y"][indices],
            episode=data.windows["episode"][indices],
            time_s=data.windows["time_s"][indices],
        )
        atomic_json(output / (partition + "-metrics.json"), scores)
        metrics[partition] = scores
    return metrics
