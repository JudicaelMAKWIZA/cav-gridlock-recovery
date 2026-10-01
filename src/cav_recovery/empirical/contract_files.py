"""Création du rapport et publication des fichiers du contrat de trafic."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

from .traffic_inputs import ALL_CATEGORIES, ContractInputError, ENTRY_GATES, EXIT_GATES


OUTPUTS = ("regime_profile.csv", "empirical_contract.json", "quality_summary.json", "regime_report.md")


def validate_output_directory(output_dir: str | Path) -> Path:
    """Accepte un dossier de sortie absent ou vide, sans écraser son contenu."""

    destination = Path(output_dir).resolve()
    if destination.exists():
        if not destination.is_dir():
            raise ContractInputError("Le chemin de sortie existe et n'est pas un dossier.")
        if any(destination.iterdir()):
            raise ContractInputError("Le dossier de sortie existe déjà et n'est pas vide.")
    return destination


def _write_profile(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "window_start_s",
        "window_end_s",
        "duration_s",
        "eligible_for_stratification",
        "passenger_in_w23183369",
        "passenger_in_w284241336",
        "passenger_in_total",
        "regime_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["eligible_for_stratification"] = str(bool(row["eligible_for_stratification"])).lower()
            output["regime_id"] = row["regime_id"] or ""
            writer.writerow(output)


def _report(
    load_levels: list[dict[str, object]],
    composition: dict[str, object],
    provenance_limitations: Iterable[str],
) -> str:
    lines = [
        "# Contrat empirique des scénarios",
        "",
        "## Méthode",
        "",
        "Les treize fenêtres complètes sont classées par entrées Car+Taxi croissantes, puis par "
        "temps de début. Les groupes LOW/MID/HIGH contiennent 4/5/4 fenêtres. La fenêtre partielle "
        "reste hors stratification.",
        "",
        "Ces groupes décrivent des niveaux relatifs de charge observée. Ils ne démontrent ni "
        "congestion, ni capacité, ni gridlock.",
        "",
        "## Contrat passenger_CAV",
        "",
        "| Niveau | Fenêtres | Entrées | Exposition par porte (s) | Référence de passages (véh/h) "
        "| Classifiables | Censurées en sortie |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for load_level in load_levels:
        lines.append(
            f"| {load_level['regime_id']} | {len(load_level['member_windows'])} | "
            f"{load_level['passenger_entry_total']} | "
            f"{load_level['exposure_s']} | {load_level['combined_reference_flow_veh_per_h']} | "
            f"{load_level['classifiable_visits']} | {load_level['censored_exit']} |"
        )
    lines.extend([
        "",
        "Les mouvements sont calculés sur les seuls mouvements Car+Taxi complets, avec les "
        "dénominateurs poolés. Les censures ne deviennent jamais des mouvements.",
        "",
        "## Mouvements passenger_CAV",
        "",
        "| Niveau | Entrée | Vers W23183369_OUT | Vers W284241336_OUT | Dénominateur |",
        "|---|---|---:|---:|---:|",
    ])
    for load_level in load_levels:
        for entry in ENTRY_GATES:
            counts = load_level["passenger_movement_counts"][entry]
            denominator = load_level["passenger_movement_denominators"][entry]
            lines.append(
                f"| {load_level['regime_id']} | {entry} | {counts[EXIT_GATES[0]]} | "
                f"{counts[EXIT_GATES[1]]} | {denominator} |"
            )
    lines.extend([
        "",
        "## Contexte empirique",
        "",
        f"Les six catégories totalisent {composition['total_observed_entries']} entrées observées : "
        f"{composition['classifiable_visits']} visites classifiables et "
        f"{composition['censored_exit']} censurées en sortie. Les "
        f"{composition['censored_entry']} censures d'entrée restent séparées.",
        "",
        "| Catégorie | Entrées observées | Classifiables | Censurées en sortie |",
        "|---|---:|---:|---:|",
    ])
    for category in ALL_CATEGORIES:
        values = composition["categories"][category]
        lines.append(
            f"| {category} | {values['observed_entries']} | {values['classifiable_visits']} | "
            f"{values['censored_exit']} |"
        )
    lines.extend([
        "",
        "Motorcycle, Bus, Medium Vehicle et Heavy Vehicle restent documentés, sans conversion en "
        "voitures. Seuls Car et Taxi alimentent le contrat principal passenger_CAV.",
        "",
        "## Statut des informations",
        "",
        "### OBSERVÉ / issu du pipeline empirique",
        "",
        "Les catégories pNEUMA, les comptages de passages, les mouvements classifiables, les censures "
        "et l'exposition proviennent du pipeline empirique validé. Les franchissements et mouvements "
        "sont extraits des trajectoires ; ils ne sont pas annotés directement sur une vidéo brute.",
        "",
        "### DÉRIVÉ / ESTIMÉ",
        "",
        "Les portes et la configuration C3, les taux en véh/h, les parts d'entrée, les probabilités "
        "de mouvements, N_passenger_in, les niveaux LOW/MID/HIGH et l'agrégation Car+Taxi sont "
        "dérivés des données validées.",
        "",
        "### SUPPOSÉ / EXPÉRIMENTAL",
        "",
        "La population simulée principale en CAV et ses paramètres restent différés : dynamique "
        "longitudinale, car-following, dimensions, accélération et décélération, processus d'arrivée, "
        "feux, capacités, incidents et actions de récupération. Aucune valeur nouvelle n'est fixée ici.",
        "",
        "## Limites",
        "",
        "Les taux sont des références empiriques de passages. Leur transformation en arrivées ou "
        "insertions SUMO n'est pas définie par ce contrat. La caractérisation porte sur un seul secteur "
        "et une seule séquence courte, avec une couverture inférée sans inspection vidéo brute.",
        "",
    ])
    lines.extend(f"- {limitation}" for limitation in provenance_limitations)
    lines.append("")
    return "\n".join(lines)


def write_contract_files(
    output_dir: str | Path,
    window_rows: list[dict[str, object]],
    contract: dict[str, object],
    quality: dict[str, object],
    load_levels: list[dict[str, object]],
    composition: dict[str, object],
    provenance_limitations: Iterable[str],
) -> None:
    """Écrit les quatre fichiers puis publie leur dossier en une seule opération.

    Tous les fichiers sont d'abord fermés dans un dossier temporaire voisin. Le
    dossier final ne peut donc pas contenir une génération partielle.
    """

    destination = validate_output_directory(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".empirical-contract-", dir=destination.parent))
    destination_was_empty = destination.exists()
    try:
        _write_profile(stage / "regime_profile.csv", window_rows)
        (stage / "empirical_contract.json").write_text(
            json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (stage / "quality_summary.json").write_text(
            json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (stage / "regime_report.md").write_text(
            _report(load_levels, composition, provenance_limitations),
            encoding="utf-8",
        )
        if destination_was_empty:
            destination.rmdir()
        try:
            os.replace(stage, destination)
        except Exception:
            if destination_was_empty and not destination.exists():
                destination.mkdir()
            raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
