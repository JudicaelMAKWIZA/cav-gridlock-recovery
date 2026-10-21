"""Exécution du protocole gelé, sélection validation puis test à accès unique."""

from pathlib import Path
import subprocess
import sys
import time

from .storage import atomic_json, file_hash, json_file, single_writer


def await_cache(cache, timeout_s=3600):
    start = time.monotonic()
    while not (cache / "complete.json").exists():
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(f"Cache inachevé : {cache}")
        time.sleep(2)


def paired_scores(cache, output, models):
    import numpy as np
    from .metrics import grouped_metrics

    rows = {}
    for partition in ["validation", "test_ood"]:
        with np.load(cache / (partition + "-windows.npz")) as raw:
            mask = raw["available"]
            y = raw["y"][mask]
            groups = raw["episode"][mask]
            baseline = {
                mode: grouped_metrics(y, raw["d01_" + mode][mask], groups)
                for mode in ["A", "B"]
            }
            for mode in ["A", "B"]:
                baseline[mode]["abstentions"] = int(raw["abstain_" + mode][mask].sum())
            for name, folder in models.items():
                result = json_file(folder / (partition + "-metrics.json"))
                with np.load(folder / (partition + "-predictions.npz")) as predictions:
                    assert np.array_equal(predictions["target"], y)
                    if name == "tcn_raw":
                        threshold = result["threshold"]
                        union = raw["d01_B"][mask].astype(bool) | (
                            predictions["probability"] >= threshold
                        )
                        baseline["D01_B_or_TCN_statistical_alert"] = grouped_metrics(
                            y, union, groups
                        )
                baseline[name] = result
            rows[partition] = baseline
    atomic_json(
        output / "paired-comparisons.json",
        dict(
            scope="held-out groups inside chosen development demonstrations; not independent generalization",
            target="physical_confirmation_available_now",
            statistical_alert_is_not_K04_confirmation=True,
            configurations=rows,
        ),
    )
    return rows


def run(root, config_file, *, m02_cache, integrated_cache):
    """Un seul coordinateur ; la reprise ne réentraîne pas les résultats terminés."""
    from .training import train, evaluate
    from .features import BASE_FEATURES

    root = Path(root)
    m02_cache = Path(m02_cache)
    integrated_cache = Path(integrated_cache)
    from .storage import safe_destination, protect_frozen_dataset

    safe_destination(m02_cache, root)
    safe_destination(integrated_cache, root)
    protect_frozen_dataset(root)
    if (root / "complete.json").exists():
        return json_file(root / "complete.json")
    protocol = json_file(config_file)
    config_base = {k: v for k, v in protocol.items() if k != "trials"}
    root.mkdir(parents=True, exist_ok=True)
    with single_writer(root / "coordinator"):
        atomic_json(root / "protocol.json", protocol)
        await_cache(m02_cache)
        completed = {}
        for number, trial in enumerate(protocol["trials"]):
            name = trial["name"]
            output = root / "m02" / name
            config = dict(config_base, **trial)
            # Une coupure réelle de processus lors du premier entraînement long
            # vérifie les protections avant de poursuivre la campagne.
            proof = root / "power-cut-proof.json"
            if (
                number == 0
                and not proof.exists()
                and not (output / "complete.json").exists()
            ):
                command = [
                    sys.executable,
                    "-m",
                    "cav_recovery.cli",
                    "lab",
                    "train",
                    str(m02_cache),
                    str(output),
                    "--config",
                    str(Path(config_file).resolve()),
                    "--trial",
                    name,
                    "--simulate-power-cut-after",
                    "3",
                ]
                code = subprocess.call(command)
                if code != 86:
                    raise RuntimeError(f"Test de coupure inattendu : {code}")
                interrupted = json_file(output / "interrupted.json")
                atomic_json(
                    proof,
                    dict(
                        exit_code=code,
                        checkpoint=interrupted,
                        actual_process_terminated=True,
                        completed=False,
                    ),
                )
            result = train(m02_cache, output, config)
            evaluate(m02_cache, output, partitions=("validation",))
            completed[name] = result
            if number == 0 and proof.exists():
                record = json_file(proof)
                record.update(
                    completed=True,
                    resume_records=[p.name for p in output.glob("resume-*.json")],
                    stale_locks_preserved=[
                        p.name for p in output.glob("interrupted-lock-*.json")
                    ],
                    final_epochs=result["completed_epochs"],
                )
                assert record["resume_records"] and record["stale_locks_preserved"]
                atomic_json(proof, record)
        candidates = [
            t["name"] for t in protocol["trials"] if t["architecture"] == "tcn"
        ]
        selected = min(
            candidates, key=lambda name: completed[name]["best_validation_loss"]
        )
        selection = dict(
            selected=selected,
            criterion="minimum_validation_binary_cross_entropy",
            validation_losses={
                k: v["best_validation_loss"] for k, v in completed.items()
            },
            test_used=False,
            dataset_identity=file_hash(m02_cache / "complete.json"),
        )
        atomic_json(root / "selected-model.json", selection)
        for name in [selected, "instantaneous_baseline"]:
            evaluate(m02_cache, root / "m02" / name)
        await_cache(integrated_cache)
        models = {}
        for name, features in [
            ("tcn_raw", len(BASE_FEATURES)),
            ("tcn_d01", len(BASE_FEATURES) + 4),
            ("tcn_d01_crdg", len(BASE_FEATURES) + 6),
        ]:
            config = dict(
                config_base,
                name=name,
                hidden=8,
                architecture="tcn",
                epochs=6,
                train_stride=1,
                validation_selection_stride=1,
            )
            output = root / "integrated" / name
            train(integrated_cache, output, config, feature_count=features)
            evaluate(integrated_cache, output, feature_count=features)
            models[name] = output
        paired_scores(integrated_cache, root, models)
        # Transfert M02 → démonstrations : mêmes variables, normalisation M02,
        # seuil M02 validation ; aucune recalibration sur les démonstrations.
        transfer_evaluation(
            m02_cache,
            integrated_cache,
            root / "m02" / selected,
            root / "transfer-m02-to-integrated",
        )
        atomic_json(
            root / "complete.json",
            dict(
                selected=selection,
                power_cut_recovery=json_file(root / "power-cut-proof.json"),
                physical_simulations_started=0,
                datasets_unchanged=True,
            ),
        )
    return json_file(root / "complete.json")


def transfer_evaluation(m02_cache, integrated_cache, model_folder, output):
    import numpy as np
    from .data import atomic_npz
    from .training import WindowData, predict, configure
    from .model import create_model
    from .checkpoints import CheckpointStore
    from .metrics import grouped_metrics

    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)
    config = json_file(model_folder / "config.json")
    receipt = json_file(model_folder / "complete.json")
    configure(config["seed"], config["threads"])
    state, _ = CheckpointStore(
        model_folder / "checkpoints", receipt["compatibility"]
    ).load(best=True)
    model = create_model(config, receipt["compatibility"]["features"])
    model.load_state_dict(state["model"])
    normalization = json_file(Path(m02_cache) / "normalization.json")
    threshold = json_file(model_folder / "calibration.json")["threshold"]
    for partition in ["train", "validation", "test_ood"]:
        data = WindowData(
            integrated_cache, partition, receipt["compatibility"]["features"]
        )
        data.mean = np.asarray(normalization["mean"], np.float32)
        data.std = np.asarray(normalization["std"], np.float32)
        indices = data.indices()
        p, loss = predict(model, data, indices, config["batch_size"])
        atomic_npz(
            output / (partition + "-predictions.npz"),
            indices=indices,
            probability=p,
            target=data.windows["y"][indices],
            episode=data.windows["episode"][indices],
            time_s=data.windows["time_s"][indices],
        )
        result = grouped_metrics(
            data.windows["y"][indices], p >= threshold, data.windows["episode"][indices]
        )
        atomic_json(
            output / (partition + "-metrics.json"),
            dict(
                **result,
                loss=loss,
                threshold=threshold,
                scope="domain_transfer_descriptive_demonstrations_not_recalibrated",
            ),
        )
