"""Prépare les métadonnées privées sans exécuter MetaDrive ni entraîner."""

import json
from pathlib import Path
import shutil
import subprocess
import uuid

import torch  # avant tout import possible de MetaDrive sous Windows

from cav_recovery.laboratory.storage import atomic_json, file_hash, json_file
from cav_recovery.recovery.graph_marl_notebook import environment_check


PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path("D:/cav-recovery-data/graph-marl-training-20261023")
CONFIG = PROJECT / "configs/recovery/graph-marl-p4-v1.json"
NOTEBOOK = PROJECT / "notebooks/11_ENTRAINEMENT_GRAPH_MARL_INTERACTIF.ipynb"


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    target = ROOT / "experiment-config.json"
    if target.exists():
        if file_hash(target) != file_hash(CONFIG):
            if (ROOT / "reports/preflight.json").exists() or (ROOT / "checkpoints/latest.json").exists():
                raise ValueError("Une campagne a commencé avec une autre configuration.")
            shutil.copy2(CONFIG, target)
    else:
        shutil.copy2(CONFIG, target)
    identity = ROOT / "experiment-identity.json"
    if not identity.exists():
        atomic_json(identity, dict(
            campaign_id=f"graph-marl-p4-20261023-{uuid.uuid4().hex[:12]}",
            private_root=str(ROOT), public_notebook=str(NOTEBOOK),
            source_p4_commit="e49e42eaa81ff8379f255496bfba94ef5c74d70f",
            config_sha256=file_hash(target),
        ))
    elif json_file(identity)["config_sha256"] != file_hash(target):
        content = json_file(identity)
        content["config_sha256"] = file_hash(target)
        atomic_json(identity, content)
    atomic_json(ROOT / "environment.json", environment_check(PROJECT, ROOT))
    source_paths = [CONFIG, NOTEBOOK] + sorted(
        (PROJECT / "src/cav_recovery/recovery").glob("*.py"))
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    atomic_json(ROOT / "source-manifest.json", dict(
        campaign=json_file(identity), git_commit=commit,
        files={path.relative_to(PROJECT).as_posix(): file_hash(path)
               for path in source_paths},
        frozen_p4_reference="D:/cav-recovery-data/graph-marl-prep-20261023/p4-reference",
        historical_m02_modified=False,
    ))
    readme = f"""# Expérience Graph-MARL + P4 — préparation seule

Identité : `{json_file(identity)['campaign_id']}`. Base P4 : `e49e42eaa81ff8379f255496bfba94ef5c74d70f`.

Notebook à ouvrir : `{NOTEBOOK}`. Noyau : `CGR MetaDrive (Python 3.12 sur D:)`, interpréteur `D:/cav-recovery-venvs/metadrive/Scripts/python.exe`.

Les cellules 01–02 sont vérifiées à la préparation. Les cellules 03–16, y compris tout épisode, entraînement, validation et export, appartiennent à l'étudiant. Les données M02 et le test réservé ne sont jamais sollicités.

Un processus distinct exécute l'entraînement pour garder le notebook réactif. Le PID, son heure de création et le journal figurent dans `reports/process.json`. La cellule 10 lit `reports/experiment_status.json` et `training/history.json`. La cellule 12 crée `stop-requested` pour demander une fin à la frontière d'épisode. La cellule 09 reprend le dernier checkpoint SHA-256 valide, l'optimiseur et les états aléatoires. Les fragments incomplets restent dans `training/episodes` avec un suffixe lors d'une reprise.

Récompense et arrivées physiques sont distinctes. La sélection du checkpoint vient seulement des scènes de validation. Les comparaisons REC-EVAL-1 avec nominal, B3 et P4 sont faites à horizon et graine identiques ; le test historique est exclu. Aucune performance Graph-MARL n'existe à ce stade.

Arborescence : `preflight/` pour le contrôle, `training/episodes/` pour les traces natives, `training/logs/` pour le journal du processus, `checkpoints/` pour les générations vérifiées, `validation/` pour les scènes distinctes et les comparaisons, `figures/` pour les images produites après exécution, `reports/` pour les statuts, `final/` pour le manifeste. Les sorties sont privées et ne doivent pas être publiées.

En cas de WinError 1114 sur `c10.dll`, redémarrer le noyau et exécuter 01 en premier : importer PyTorch après MetaDrive échoue dans cet environnement Windows. Ne pas ignorer un échec natif.
"""
    (ROOT / "README_EXPERIENCE.md").write_text(readme, encoding="utf-8")
    print(ROOT)


if __name__ == "__main__":
    main()
