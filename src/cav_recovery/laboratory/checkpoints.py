"""Sauvegardes générationnelles vérifiées, réutilisables pour d'autres modèles.

Un état Graph-MARL pourra fournir ses politiques, critiques et compteurs dans
extra_state. Cela ne sauvegarde pas magiquement l'état d'un moteur physique :
une reprise d'interaction devra partir d'une frontière d'épisode explicite.
"""

import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .storage import atomic_json, file_hash, json_file


def random_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
    )


def restore_random_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


class CheckpointStore:
    """Ne remplace jamais l'unique génération valide ; replie sur la précédente."""

    def __init__(self, folder, compatibility):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.compatibility = compatibility

    def save(self, state, *, best=False):
        filename = f"checkpoint-{state['global_step']:010d}-{time.time_ns()}.pt"
        target = self.folder / filename
        temporary = target.with_suffix(".tmp")
        payload = dict(
            schema="cgr-checkpoint-1",
            compatibility=self.compatibility,
            random_state=random_state(),
            **state,
        )
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Validation avant remplacement et avant publication du pointeur.
        restored = torch.load(temporary, map_location="cpu", weights_only=False)
        if (
            restored["compatibility"] != self.compatibility
            or restored["global_step"] != state["global_step"]
        ):
            raise ValueError("Checkpoint temporaire invalide.")
        os.replace(temporary, target)
        receipt = dict(
            file=filename, sha256=file_hash(target), step=state["global_step"]
        )
        atomic_json(target.with_suffix(".json"), receipt)
        atomic_json(self.folder / "latest.json", receipt)
        if best:
            atomic_json(self.folder / "best.json", receipt)
        return receipt

    def load(self, *, best=False):
        receipts = []
        index = self.folder / ("best.json" if best else "latest.json")
        if index.exists():
            try:
                receipts.append(json_file(index))
            except (ValueError, OSError):
                pass
        if not best:
            for path in sorted(self.folder.glob("checkpoint-*.json"), reverse=True):
                try:
                    receipts.append(json_file(path))
                except (ValueError, OSError):
                    continue
        failures = []
        for receipt in receipts:
            path = self.folder / receipt["file"]
            try:
                if file_hash(path) != receipt["sha256"]:
                    failures.append(dict(file=path.name, reason="sha256_mismatch"))
                    continue
                state = torch.load(path, map_location="cpu", weights_only=False)
            except (OSError, RuntimeError, EOFError, ValueError):
                failures.append(dict(file=path.name, reason="unreadable"))
                continue
            if state["compatibility"] != self.compatibility:
                raise ValueError(
                    "Modèle, données ou configuration incompatibles avec la reprise."
                )
            return state, dict(receipt=receipt, rejected=failures)
        if receipts:
            raise ValueError(
                "Aucun checkpoint valide ; les générations invalides sont conservées."
            )
        return None, None
