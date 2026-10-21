"""Consultation commune aux notebooks et à l'export académique, sans réentraînement."""

from collections import Counter, defaultdict
import csv
import html
import os
from pathlib import Path

import numpy as np

from .data import rows, atomic_npz
from .metrics import binary_metrics
from .storage import json_file


class Laboratory:
    def __init__(self, root, dataset, historical):
        self.root = Path(root)
        self.dataset = Path(dataset)
        self.historical = Path(historical)
        self.campaign = self.root / "campaign-cpu-v1"
        self.cache = self.root / "cache-m02-v1"
        self.integrated_cache = self.root / "cache-integrated-v2"

    @classmethod
    def from_environment(cls):
        data = Path(os.environ.get("CGR_DATA_ROOT", "D:/cav-recovery-data"))
        return cls(
            os.environ.get("CGR_LAB_ROOT", data / "detection-tcn-lab-20261021"),
            os.environ.get("CGR_M02_ROOT", data / "metadrive-dense-20261018"),
            os.environ.get(
                "CGR_HISTORICAL_ROOT", data / "integrated-metadrive-20261020"
            ),
        )

    def dataset_overview(self):
        scientific = json_file(
            self.dataset / "production-final/scientific-summary.json"
        )
        splits = json_file(self.dataset / "production-final/splits.json")["episodes"]
        counts = defaultdict(Counter)
        for episode in scientific["episode_evidence"]:
            partition = (
                "unknown"
                if episode["label"] == "UNKNOWN_OR_CENSORED"
                else splits[episode["id"]]
            )
            counts[partition][episode["label"]] += 1
        labels = ["POSITIVE_CONFIRMED", "NEGATIVE_CONFIRMED", "UNKNOWN_OR_CENSORED"]
        return [
            dict(
                partition=k,
                total=sum(v.values()),
                **{label: v.get(label, 0) for label in labels},
            )
            for k, v in sorted(counts.items())
        ]

    def historical_episodes(self):
        names = json_file(self.historical / "demonstration-selection.json")[
            "all_evaluated_names"
        ]
        results = []
        for name in names:
            folder = self.historical / "campaign-validated" / name
            if not (folder / "complete.json").exists():
                folder = self.historical / "supplementary-demonstrations" / name
            results.append(json_file(folder / "evaluation/results.json"))
        return results

    def historical_metrics(self):
        from ..experiments.evaluation import metrics

        totals = defaultdict(Counter)
        for result in self.historical_episodes():
            for task, values in result["metrics"].items():
                totals[task].update(
                    {key: values.get(key, 0) for key in ["tp", "fp", "fn", "tn"]}
                )
        return [dict(task=task, **metrics(counts)) for task, counts in totals.items()]

    def historical_abstentions(self):
        counts = defaultdict(Counter)
        for result in self.historical_episodes():
            for mode, values in result["abstentions"].items():
                counts[mode].update(values)
        return [dict(configuration=mode, **values) for mode, values in counts.items()]

    def model_results(self, scope="m02", partition="test_ood"):
        if scope == "m02":
            selected = json_file(self.campaign / "selected-model.json")["selected"]
            models = [selected, "instantaneous_baseline"]
            return [
                dict(
                    model=name,
                    **json_file(
                        self.campaign / "m02" / name / (partition + "-metrics.json")
                    )["micro"],
                )
                for name in models
            ]
        comparisons = json_file(self.campaign / "paired-comparisons.json")[
            "configurations"
        ][partition]
        return [
            dict(model=name, **value["micro"]) for name, value in comparisons.items()
        ]

    def histories(self):
        return {
            p.parent.name: json_file(p)
            for p in sorted((self.campaign / "m02").glob("*/history.json"))
        }

    def transfer_results(self):
        """Transfert descriptif, sans ajuster le seuil ni la normalisation M02."""
        folder = self.campaign / "transfer-m02-to-integrated"
        return [
            dict(
                partition=partition,
                **json_file(folder / f"{partition}-metrics.json")["micro"],
            )
            for partition in ["train", "validation", "test_ood"]
        ]

    def replay_costs(self):
        """Coûts enregistrés du rejeu, acquisition native et physique exclues."""
        source = self.root / "native-model-parity/diagnostic/timings.json"
        if not source.exists():
            return []
        records = json_file(source)
        return [
            dict(
                component=key,
                frames=len(records),
                unit="ms par frame, tous CAV",
                mean=float(np.mean(values)),
                median=float(np.median(values)),
                p95=float(np.quantile(values, 0.95)),
                maximum=float(np.max(values)),
                scope="rejeu enregistré, sans moteur, rendu ni acquisition native",
            )
            for key in [
                "detector",
                "crdg",
                "temporal_inference",
                "operational_with_temporal",
            ]
            for values in [[row[key] for row in records]]
        ]

    def endpoint_metadata(self, scope, partition):
        """Les identités servent aux jointures/audits et restent hors du tenseur."""
        target = self.root / "endpoint-identities" / f"{scope}-{partition}.npz"
        if target.exists():
            with np.load(target, allow_pickle=False) as data:
                return {k: data[k] for k in data.files}
        cache = self.cache if scope == "m02" else self.integrated_cache
        members = json_file(cache / (partition + "-episodes.json"))
        with np.load(cache / (partition + "-windows.npz")) as data:
            expected_time = data["time_s"]
            expected_episode = data["episode"]
        vehicles = []
        times = []
        episode_ids = []
        for index, member in enumerate(members):
            name = member["episode"]
            if scope == "m02":
                with np.load(
                    cache / "shards" / (name + ".npz"), allow_pickle=False
                ) as shard:
                    vehicles.extend(shard["vehicle"])
                    times.extend(shard["time_s"])
                    episode_ids.extend([index] * len(shard["y"]))
            else:
                folder = self.historical / "campaign-validated" / name
                if not (folder / "complete.json").exists():
                    folder = self.historical / "supplementary-demonstrations" / name
                own = defaultdict(list)
                for row in rows(folder / "diagnostic/features.jsonl.gz"):
                    own[row["ego_id"]].append(row["observed_at_s"])
                for ego, sequence in sorted(own.items()):
                    for at in sequence:
                        if (
                            at - sequence[0] >= 5.0 - 1e-7
                            and abs(at / 2 - round(at / 2)) <= 1e-7
                        ):
                            vehicles.append(ego)
                            times.append(at)
                            episode_ids.append(index)
        assert np.array_equal(expected_episode, np.asarray(episode_ids))
        np.testing.assert_allclose(expected_time, times, atol=1e-7)
        result = dict(
            vehicle=np.asarray(vehicles),
            time_s=np.asarray(times),
            episode=np.asarray(episode_ids),
        )
        atomic_npz(target, **result)
        return result

    def confusion(self, scope="m02", partition="test_ood", name=None):
        if scope == "m02":
            name = name or json_file(self.campaign / "selected-model.json")["selected"]
        else:
            name = name or "tcn_d01_crdg"
        folder = self.campaign / scope / name
        with np.load(folder / (partition + "-predictions.npz")) as data:
            score = json_file(folder / (partition + "-metrics.json"))
            computed = binary_metrics(
                data["target"], data["probability"] >= score["threshold"]
            )
            assert all(
                computed[k] == score["micro"][k] for k in ["tp", "fp", "fn", "tn"]
            )
        return computed

    @staticmethod
    def table_html(records, limit=40):
        if not records:
            return "<p>Aucun enregistrement.</p>"
        keys = list(dict.fromkeys(key for row in records for key in row))

        def render(value):
            if value is None:
                return "non défini"
            if isinstance(value, float):
                return f"{value:.5g}"
            return html.escape(str(value))

        header = (
            "<tr>" + "".join("<th>" + html.escape(k) + "</th>" for k in keys) + "</tr>"
        )
        body = "".join(
            "<tr>"
            + "".join("<td>" + render(row.get(k)) + "</td>" for k in keys)
            + "</tr>"
            for row in records[:limit]
        )
        return (
            '<table style="border-collapse:collapse" border="1" cellpadding="6">'
            + header
            + body
            + "</table>"
        )

    def plot_dataset(self):
        import matplotlib.pyplot as plt

        summary = json_file(self.dataset / "production-final/summary.json")
        fig, ax = plt.subplots(figsize=(9, 4))
        families = list(summary["families"])
        bottom = np.zeros(len(families))
        for label, title, color in [
            ("POSITIVE_CONFIRMED", "Positifs", "#35698b"),
            ("NEGATIVE_CONFIRMED", "Négatifs", "#ce8525"),
            ("UNKNOWN_OR_CENSORED", "Inconnus", "#85919f"),
        ]:
            values = [summary["families"][family].get(label, 0) for family in families]
            ax.bar(families, values, bottom=bottom, label=title, color=color)
            bottom += values
        ax.set(ylabel="Épisodes", title="M02 gelé : quatre topologies, 2 194 épisodes")
        ax.legend()
        fig.tight_layout()
        return fig

    def plot_quality(self, scope="m02", partition="test_ood"):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter

        values = self.model_results(scope, partition)
        labels = {
            "tcn_h8": "TCN causal\n8 canaux cachés",
            "tcn_h12": "TCN causal\n12 canaux cachés",
            "instantaneous_baseline": "Baseline\ninstantanée",
            "A": "D01 A",
            "B": "D01 B",
            "D01_B_or_TCN_statistical_alert": "Alerte statistique\nD01 B ou TCN",
            "tcn_raw": "TCN brut",
            "tcn_d01": "TCN + D01",
            "tcn_d01_crdg": "TCN + D01\n+ C-RDG",
        }
        fig, ax = plt.subplots(figsize=(12, 4.8))
        x = np.arange(len(values))
        for shift, key, label, color in [
            (-0.24, "precision", "Précision", "#35698b"),
            (0, "recall", "Rappel", "#ce8525"),
            (0.24, "f1", "F1", "#347c65"),
        ]:
            bars = ax.bar(
                x + shift,
                [row[key] or 0 for row in values],
                0.23,
                label=label,
                color=color,
            )
            ax.bar_label(
                bars,
                labels=[
                    "n.d." if row[key] is None else f"{row[key] * 100:.1f}"
                    for row in values
                ],
                rotation=90,
                padding=3,
                fontsize=8,
            )
        ax.set(
            xticks=x,
            xticklabels=[labels.get(row["model"], row["model"]) for row in values],
            ylim=(0, 1.2),
            ylabel="Score (%)",
            title=(
                "Test OOD M02 : confirmation disponible maintenant"
                if scope == "m02"
                else "Groupes de démonstration tenus à part : étude exploratoire appariée"
            ),
        )
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend()
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        return fig

    def plot_losses(self):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for name, history in self.histories().items():
            for ax, key in zip(axes, ["train_loss", "validation_loss"]):
                ax.plot(
                    [r["epoch"] for r in history],
                    [r[key] for r in history],
                    marker="o",
                    label=name,
                )
        axes[0].set_title("Train : BCE pondérée")
        axes[1].set_title("Validation : BCE non pondérée")
        for ax in axes:
            ax.set(xlabel="Époque", ylabel="Perte")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        fig.tight_layout()
        return fig

    def plot_confusion(self, scope="m02"):
        import matplotlib.pyplot as plt

        score = self.confusion(scope)
        array = np.array([[score["tn"], score["fp"]], [score["fn"], score["tp"]]])
        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(array, cmap="Blues")
        for (y, x), value in np.ndenumerate(array):
            ax.text(
                x,
                y,
                str(value),
                ha="center",
                va="center",
                color="white" if value > array.max() / 2 else "#172c40",
            )
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Absente", "Présente"],
            yticklabels=["Absente", "Présente"],
            xlabel="Estimation statistique",
            ylabel="Confirmation physique de référence",
            title="Matrice de confusion : fenêtres supervisées",
        )
        fig.colorbar(image, ax=ax, label="Fenêtres corrélées")
        fig.tight_layout()
        return fig

    def plot_historical_graphs(self):
        import matplotlib.pyplot as plt

        summary = json_file(self.historical / "report-summary.json")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        partial = summary["group_validity"]
        axes[0].bar(
            ["B", "C"],
            [
                partial[k]["valid"] / (partial[k]["valid"] + partial[k]["invalid"])
                for k in ["B", "C"]
            ],
            color=["#ce8525", "#35698b"],
        )
        axes[0].set(
            title="Validité physique des sous-groupes",
            ylabel="Proportion valide",
            ylim=(0, 1.05),
        )
        axes[1].bar(
            ["B exact", "C exact"],
            [summary["metrics"][k]["f1"] for k in ["B_groups", "C_groups"]],
            color=["#ce8525", "#35698b"],
        )
        axes[1].set(title="Restitution exacte des groupes", ylabel="F1", ylim=(0, 0.1))
        fig.suptitle(
            "32 démonstrations historiques choisies, sans modification des mesures"
        )
        fig.tight_layout()
        return fig

    def event_analysis(self, scope="m02", partition="test_ood", name=None):
        """Délais sur événements éligibles ; les manqués et préalertes restent visibles."""
        cache = self.cache if scope == "m02" else self.integrated_cache
        name = name or (
            json_file(self.campaign / "selected-model.json")["selected"]
            if scope == "m02"
            else "tcn_d01_crdg"
        )
        output = self.campaign / scope / name
        predictions = np.load(output / (partition + "-predictions.npz"))
        threshold = json_file(output / (partition + "-metrics.json"))["threshold"]
        metadata = self.endpoint_metadata(scope, partition)
        indices = predictions["indices"]
        actors = metadata["vehicle"][indices]
        times = predictions["time_s"]
        episodes = predictions["episode"]
        positive = predictions["probability"] >= threshold
        lookup = defaultdict(list)
        for index, (episode, actor) in enumerate(zip(episodes, actors)):
            lookup[(int(episode), str(actor))].append(index)
        members = json_file(cache / (partition + "-episodes.json"))
        records = []
        for number, member in enumerate(members):
            if scope == "m02":
                annotation = json_file(
                    self.dataset
                    / "production-final/episodes"
                    / member["episode"]
                    / "annotation.json"
                )
            else:
                folder = self.historical / "campaign-validated" / member["episode"]
                if not folder.exists():
                    folder = (
                        self.historical
                        / "supplementary-demonstrations"
                        / member["episode"]
                    )
                annotation = json_file(folder / "evaluation/physical-reference.json")[
                    "annotation"
                ]
            for event in annotation["events"]:
                for effect in event["effect_intervals"]:
                    if not effect["confirmed"] or not event["confirmed"]:
                        continue
                    confirmation = effect["confirmation_at_s"]
                    end = effect["end_s"]
                    own = np.asarray(
                        lookup.get((number, event["requester"]), []), dtype=np.int64
                    )
                    local_times = times[own]
                    local_positive = positive[own]
                    eligible = (local_times >= confirmation - 1e-7) & (
                        local_times <= end + 1e-7
                    )
                    detected = local_times[eligible & local_positive]
                    premature = (
                        (local_times >= effect["start_s"] - 1e-7)
                        & (local_times < confirmation - 1e-7)
                        & local_positive
                    )
                    records.append(
                        dict(
                            episode=member["episode"],
                            vehicle=event["requester"],
                            confirmation_s=confirmation,
                            end_s=end,
                            eligible_endpoints=int(eligible.sum()),
                            first_statistical_alert_s=float(detected[0])
                            if len(detected)
                            else None,
                            delay_s=float(detected[0] - confirmation)
                            if len(detected)
                            else None,
                            missed=not bool(len(detected)),
                            censored_no_endpoint=not bool(eligible.any()),
                            premature_statistical_alert=bool(premature.any()),
                        )
                    )
        return records

    def export_table(self, folder, name, records):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if not records:
            return
        with (folder / (name + ".csv")).open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            keys = list(dict.fromkeys(key for row in records for key in row))
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)
        (folder / (name + ".html")).write_text(
            self.table_html(records, limit=len(records)), encoding="utf-8"
        )

    def plot_event_delays(self, records):
        import matplotlib.pyplot as plt

        eligible = [r for r in records if not r["censored_no_endpoint"]]
        delays = [r["delay_s"] for r in eligible if r["delay_s"] is not None]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(delays, bins=20, color="#35698b")
        axes[0].set(
            xlabel="Délai après confirmation (s)",
            ylabel="Événements retrouvés",
            title="Délais conditionnels aux événements retrouvés",
        )
        axes[1].bar(
            ["Retrouvés", "Manqués", "Sans endpoint"],
            [
                len(delays),
                sum(r["missed"] for r in eligible),
                sum(r["censored_no_endpoint"] for r in records),
            ],
            color=["#35698b", "#ce8525", "#85919f"],
        )
        axes[1].set(
            title="Les manqués ne disparaissent pas des effectifs",
            ylabel="Événements physiques",
        )
        fig.suptitle(
            f"Préalertes statistiques dans {sum(r['premature_statistical_alert'] for r in records)} événements "
            "(sous-ensemble non exclusif des effectifs ci-dessous)",
            fontsize=10,
        )
        fig.tight_layout()
        return fig

    def plot_motion(self, episode_name=None, ego=None, model_name="tcn_d01_crdg"):
        """Exemple réel du groupe tenu à part le moins bien restitué par le modèle."""
        import matplotlib.pyplot as plt

        members = json_file(self.integrated_cache / "test_ood-episodes.json")
        metrics = json_file(
            self.campaign / "integrated" / model_name / "test_ood-metrics.json"
        )
        if episode_name is None:
            eligible = [r for r in metrics["episodes"] if r["f1"] is not None]
            worst = min(
                eligible or metrics["episodes"],
                key=lambda r: r["f1"] if r["f1"] is not None else 1.0,
            )
            episode_name = members[worst["episode"]]["episode"]
        partition = next(
            partition
            for partition in ["train", "validation", "test_ood"]
            if any(
                m["episode"] == episode_name
                for m in json_file(
                    self.integrated_cache / (partition + "-episodes.json")
                )
            )
        )
        members = json_file(self.integrated_cache / (partition + "-episodes.json"))
        number = next(i for i, m in enumerate(members) if m["episode"] == episode_name)
        folder = self.historical / "campaign-validated" / episode_name
        if not folder.exists():
            folder = self.historical / "supplementary-demonstrations" / episode_name
        identity_rows = self.endpoint_metadata("integrated", partition)
        if ego is None:
            with np.load(
                self.campaign
                / "integrated"
                / model_name
                / (partition + "-predictions.npz")
            ) as predictions:
                actor_ids = identity_rows["vehicle"][predictions["indices"]]
                selected_episode = predictions["episode"] == number
                incorrect = (
                    predictions["probability"] >= metrics["threshold"]
                ) != predictions["target"]
                actors = np.unique(actor_ids[selected_episode])
                ego = max(
                    actors,
                    key=lambda actor: int(
                        np.sum(selected_episode & (actor_ids == actor) & incorrect)
                    ),
                )
        features = [
            row
            for row in rows(folder / "diagnostic/features.jsonl.gz")
            if row["ego_id"] == ego
        ]
        at = np.array([r["observed_at_s"] for r in features])
        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(
            at,
            [r["own_state"]["signed_speed_mps"] for r in features],
            label="Vitesse mesurée",
        )
        axes[0].plot(
            at,
            [r["requested_speed_mps"] for r in features],
            label="Consigne demandée",
            ls="--",
        )
        axes[0].set_ylabel("m/s")
        axes[0].legend(fontsize=8)
        statuses = ["NO_ALERT", "SUSPECTED", "CONFIRMED_OBSTRUCTION", "UNKNOWN"]
        axes[1].step(
            at,
            [statuses.index(r["local_status"]) for r in features],
            where="post",
            color="#35698b",
        )
        axes[1].set(
            yticks=range(4),
            yticklabels=["Sans alerte", "Soupçon", "Confirmé K04", "Abstention"],
            ylabel="D01 (pas 0,2 s)",
        )
        with np.load(
            self.campaign / "integrated" / model_name / (partition + "-predictions.npz")
        ) as data:
            chosen = (data["episode"] == number) & (
                identity_rows["vehicle"][data["indices"]] == ego
            )
            axes[2].step(
                data["time_s"][chosen],
                data["probability"][chosen],
                where="post",
                label="Probabilité TCN",
            )
            axes[2].step(
                data["time_s"][chosen],
                data["target"][chosen],
                where="post",
                label="Confirmation physique disponible",
                ls=":",
            )
        axes[2].axhline(
            metrics["threshold"],
            color="#c44c4c",
            ls="--",
            label="Seuil ajusté sur validation",
        )
        axes[2].set(
            xlabel="Temps simulé (s)", ylabel="Probabilité / cible", ylim=(-0.05, 1.1)
        )
        axes[2].legend(fontsize=8)
        spec = json_file(folder / "spec.json")
        fig.cgr_context = dict(
            scenario=episode_name,
            ego_id=str(ego),
            seed=spec["seed"],
            topology=spec["family"],
            partition=partition,
            first_time_s=float(at[0]),
            last_time_s=float(at[-1]),
            choice="lowest defined held-out episode F1, then highest error count CAV",
            source=str(folder / "diagnostic/features.jsonl.gz"),
        )
        fig.suptitle(
            f"{episode_name} · {ego.replace('agent', 'V')} · graine {spec['seed']} · {spec['family']}\nPartition {partition} des démonstrations ; sélection par plus faible F1 d’épisode, sans réajuster le modèle"
        )
        for ax in axes:
            ax.grid(alpha=0.2)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        return fig
