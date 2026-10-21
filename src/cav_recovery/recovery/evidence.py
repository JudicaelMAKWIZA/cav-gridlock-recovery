"""Consultation des essais enregistrés : aucun moteur ou apprentissage lancé."""

from pathlib import Path
import os

from ..laboratory.storage import json_file


class RecoveryLab:
    def __init__(self, root=None):
        self.root = Path(
            root
            or os.environ.get(
                "CGR_RECOVERY_ROOT",
                "D:/cav-recovery-data/recovery-prerequisites-20261021",
            )
        )

    def runs(self):
        results = []
        for path in sorted(self.root.glob("*/*/summary.json")):
            row = json_file(path)
            if not isinstance(row, dict) or "method" not in row:
                continue
            results.append(
                dict(
                    trial=path.parent.parent.name,
                    method=row["method"],
                    seed=row["seed"],
                    horizon_observed_s=row["at_s"],
                    peak_active=row.get("peak_active"),
                    introduced=row["introduced"],
                    arrived=len(row["arrivals"]),
                    integrity_errors=len(row["errors"]),
                    physical_label=row["physical_label"],
                    greatest_reverse_m=max(row["reverse_m"].values(), default=0.0),
                    cpu_s=row["cpu_s"],
                    wall_s=row["wall_s"],
                )
            )
        return results

    def comparisons(self):
        return [
            dict(trial=p.parent.name, **json_file(p))
            for p in sorted(self.root.glob("*/comparison.json"))
        ]

    def plot_comparison(self, name):
        import matplotlib.pyplot as plt

        result = json_file(self.root / name / "comparison.json")
        figure, axis = plt.subplots(figsize=(9, 4))
        for method, label in [
            ("nominal", "Nominal"),
            ("intervention", "Baseline locale"),
        ]:
            timeline = result[method].get("timeline", [])
            axis.plot(
                [r["at_s"] for r in timeline],
                [r["fraction"] for r in timeline],
                label=label,
            )
        axis.set(
            xlabel="Temps simulé (s)",
            ylabel="Fraction de A0 ayant progressé d’une longueur",
            ylim=(-0.03, 1.05),
            title=name
            + "\nPopulation physique fixée hors acteur ; ce tracé seul ne prouve pas la durabilité",
        )
        if result["t_act_s"] is not None:
            axis.axvline(
                result["t_act_s"],
                color="gray",
                ls=":",
                label="Première commande appliquée",
            )
        axis.legend()
        axis.grid(alpha=0.2)
        figure.tight_layout()
        return figure

    def metrology(self):
        return [json_file(p) for p in sorted(self.root.glob("metrology-*/report.json"))]

    def plot_metrology(self):
        import matplotlib.pyplot as plt

        reports = self.metrology()
        figure, axis = plt.subplots(figsize=(8, 4))
        if not reports:
            raise ValueError("Aucun rejeu métrologique enregistré.")
        report = reports[-1]
        for offset, (key, label) in enumerate(
            [("historical_raw_B", "Historique brut"), ("metrics", "REC-METRO-1")]
        ):
            metrics = report[key]
            axis.bar(
                [n + offset * 0.35 for n in range(3)],
                [metrics[k] for k in ["precision", "recall", "f1"]],
                width=0.35,
                label=label,
            )
        axis.set_xticks([0.175, 1.175, 2.175], ["Précision", "Rappel", "F1"])
        axis.set(
            ylim=(0, 1.08),
            ylabel="Mesure sur les mêmes rendez-vous",
            title=report["development_scene"]
            + " · scène connue de développement\nAdaptation de mesure explicite ; aucune mise à jour des scores historiques",
        )
        axis.legend()
        figure.tight_layout()
        return figure
