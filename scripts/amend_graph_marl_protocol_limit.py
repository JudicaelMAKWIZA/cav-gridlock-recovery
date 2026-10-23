"""Trace le correctif de taille WKT après l'arrêt de l'épisode 8 CAV.

Ne simule rien et ne modifie ni les traces ni les checkpoints existants.
"""

import copy
from pathlib import Path
import shutil
import subprocess
import time

import torch  # avant les imports qui peuvent charger MetaDrive sous Windows

from cav_recovery.laboratory.checkpoints import CheckpointStore
from cav_recovery.laboratory.storage import atomic_json, file_hash, json_file
from cav_recovery.recovery.graph_marl_core import compatibility, load_config
from cav_recovery.recovery.graph_marl_notebook import process_status


PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path("D:/cav-recovery-data/graph-marl-training-20261023")
CONFIG = ROOT / "experiment-config.json"
ALLOWED = {
    "src/cav_recovery/recovery/protocol.py",
    "src/cav_recovery/recovery/graph_marl_notebook.py",
}


def verify_manifest(folder):
    manifest = json_file(folder / "graph-marl-manifest.json")
    for name, expected in manifest["files"].items():
        if file_hash(folder / name) != expected:
            raise ValueError(f"Preuve modifiée : {folder / name}")


def main():
    if process_status(ROOT)["alive"]:
        raise RuntimeError("Un producteur est encore actif : amendement refusé.")
    manifest_path = ROOT / "source-manifest.json"
    old = json_file(manifest_path)
    current = {name: file_hash(PROJECT / name) for name in old["files"]}
    changed = {name for name, digest in current.items()
               if digest != old["files"][name]}
    amendment_path = ROOT / "SOURCE_AMENDMENT_PROTOCOL_WKT.json"
    if amendment_path.exists():
        if changed:
            raise ValueError("Amendement déjà présent mais sources divergentes.")
        print("Amendement déjà vérifié :", amendment_path)
        return
    if changed != ALLOWED:
        raise ValueError(f"Sources changées hors du correctif prévu : {sorted(changed)}")
    if file_hash(CONFIG) != file_hash(PROJECT / "configs/recovery/graph-marl-p4-v1.json"):
        raise ValueError("Configuration scientifique différente.")
    history = json_file(ROOT / "training/history.json")
    if [row["index"] for row in history] != [0, 1, 2]:
        raise ValueError("Le correctif est borné à trois épisodes checkpointés.")
    store = CheckpointStore(ROOT / "checkpoints",
                            compatibility(CONFIG, load_config(CONFIG)))
    saved, receipt = store.load()
    if saved is None or saved["history"] != history:
        raise ValueError("Historique et checkpoint divergent.")
    preflight = json_file(ROOT / "reports/preflight.json")
    verify_manifest(ROOT / preflight["folder"])
    for row in history:
        folder = ROOT / row["folder"]
        verify_manifest(folder)
        hashes = json_file(folder / "source-hashes.json")
        if hashes["protocol.py"] != old["files"]["src/cav_recovery/recovery/protocol.py"]:
            raise ValueError(f"Source historique incohérente : {folder}")
    failures = []
    for folder in sorted((ROOT / "training/episodes").glob("episode-003*")):
        failure = folder / "failure.json"
        if not failure.exists():
            continue
        row = json_file(failure)
        if row["message"] != "Espace demandé invalide." or row["at_s"] != 96.0:
            raise ValueError(f"Échec inattendu : {folder}")
        if json_file(folder / "source-hashes.json")["protocol.py"] != old["files"][
                "src/cav_recovery/recovery/protocol.py"]:
            raise ValueError(f"Version du protocole inattendue : {folder}")
        failures.append(str(folder.relative_to(ROOT)))
    if len(failures) != 2:
        raise ValueError("Deux échecs appariés à 96,0 s sont requis.")
    status_path = ROOT / "reports/experiment_status.json"
    status = json_file(status_path)
    if status["state"] != "running" or status["completed_episodes"] != 3:
        raise ValueError("État de campagne inattendu.")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=PROJECT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"],
                               cwd=PROJECT, text=True).strip():
        raise ValueError("Publier d'abord les sources corrigées sur la branche dédiée.")
    prior_manifest = ROOT / "source-manifest-before-protocol-amendment.json"
    prior_status = ROOT / "reports/experiment-status-before-protocol-amendment.json"
    if prior_manifest.exists() or prior_status.exists():
        raise FileExistsError("Une sauvegarde d'amendement préexiste.")
    shutil.copy2(manifest_path, prior_manifest)
    shutil.copy2(status_path, prior_status)
    amended = copy.deepcopy(old)
    amended["git_commit"] = commit
    amended["files"].update(current)
    atomic_json(amendment_path, dict(
        reason="Le WKT d'une enveloppe locale valide dépasse 20 000 caractères.",
        old_git_commit=old["git_commit"], new_git_commit=commit,
        changes={name: dict(old_sha256=old["files"][name],
                            new_sha256=current[name]) for name in sorted(changed)},
        prior_completed_episodes=3, first_episode_under_new_source=3,
        checkpoint=receipt["receipt"], failed_fragments=failures,
        preflight_unchanged=True, prior_status_sha256=file_hash(prior_status),
        claim="Correction de la borne de taille et du statut ; aucune règle physique modifiée.",
    ))
    atomic_json(manifest_path, amended)
    status["state"] = "interrupted_at_episode"
    status["error_type"] = "ValueError"
    status["error"] = "Espace demandé invalide."
    status["updated_at"] = time.time()
    atomic_json(status_path, status)
    print("Amendement vérifié :", amendment_path)


if __name__ == "__main__":
    main()
