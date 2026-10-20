"""Vocabulaire de la vue : les codes scientifiques restent inchangés en mémoire."""

from functools import lru_cache
import json
from pathlib import Path
import subprocess


FRENCH = {
    "NO_ALERT": "Pas d'alerte",
    "SUSPECTED": "Contrainte suspectée",
    "CONFIRMED_OBSTRUCTION": "Obstruction confirmée",
    "UNKNOWN": "Information insuffisante",
    "STOPPED_UNEXPLAINED": "Arrêt sans cause physique établie",
    "PROGRESSING": "Le véhicule progresse",
    "ORDINARY_WAIT": "Attente normale ou demande suspendue",
    "CANDIDATE_PHYSICAL_CONSTRAINT": "Occupation observée ; persistance à vérifier",
    "PERSISTENT_MOVEMENT_OBSTRUCTION": "Mouvement physiquement obstrué dans la durée",
    "NON_NOMINAL_REVERSE": "Recul mesuré : le détecteur s'abstient",
    "INSUFFICIENT_INFORMATION": "Les preuves disponibles ne suffisent pas",
    "MOVEMENT_ROAD_ADMISSIBILITY_UNKNOWN": "Trajectoire autorisée non vérifiable",
    "INCOMPLETE_OWN_MOVEMENT": "Trajectoire propre incomplète",
    "CONTRADICTORY_CURRENT_EVIDENCE": "Observations actuelles contradictoires",
    "UNKNOWN_RULE_CONTEXT": "Règles de priorité non renseignées",
    "EGO_OBSERVATION_GAP": "Interruption des observations embarquées",
    "OLD_OWN_MEASUREMENT": "Mesure propre trop ancienne",
    "INVALID_STATIC_GEOMETRY": "Géométrie de carte invalide",
    "MISSION_CHANGED": "Mission modifiée",
    "RESOURCE_CHANGED": "L'espace requis a changé",
    "KNOWN_RELATION_RELEASED": "L'occupation connue a disparu",
    "MISSING_CURRENT_CAV_OCCUPANCY": "Occupation actuelle non étayée",
    "CONSTRAINT_OBSERVED": "Début d'une contrainte observée",
    "OBSTRUCTION_CONFIRMED": "Obstruction confirmée",
    "MOVEMENT_RESUMED": "Reprise du mouvement",
    "OBSERVATION_LOST": "Preuves interrompues",
    "RELEASE_OBSERVED": "Libération observée",
    "DEMAND_WITHDRAWN": "Demande de mouvement suspendue",
    "CENSORED": "Observation terminée sans conclusion complète",
}


def french(code):
    """Un code nouveau reste accessible dans les traces, sans envahir le panneau."""
    return FRENCH.get(code, "Donnée non exploitable ; consulter la trace technique")


def label(key):
    return "V" + key.removeprefix("agent") if key.startswith("agent") else key


def wrapped(text, font, width):
    """Retourne des lignes tenant réellement dans la largeur, même pour un identifiant."""
    line = ""
    for word in str(text).split():
        candidate = (line + " " + word).strip()
        if font.size(candidate)[0] <= width:
            line = candidate
            continue
        if line:
            yield line
        line = ""
        for character in word:
            if line and font.size(line + character)[0] > width:
                yield line
                line = ""
            line += character
    if line:
        yield line


@lru_cache(maxsize=1)
def source_stamp():
    """Le SHA est lu une fois, au lancement, depuis le dépôt de ce module."""
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=3,
            ).strip()
        )
        return dict(commit=commit, dirty=dirty)
    except (OSError, subprocess.SubprocessError):
        return dict(commit="indisponible", dirty=True)


def save_capture(surface, directory, metadata, number):
    """Sauvegarde les pixels réellement affichés et leur provenance en fichier voisin."""
    import pygame

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"inspection-{metadata['at_s']:07.2f}-{number:04d}"
    path = directory / (stem + ".png")
    # Une reprise au même instant ne remplace jamais une capture déjà présente.
    while path.exists() or path.with_suffix(".json").exists():
        number += 1
        path = directory / f"inspection-{metadata['at_s']:07.2f}-{number:04d}.png"
    pygame.image.save(surface, str(path))
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path
