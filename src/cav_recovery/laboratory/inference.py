"""Estimation statistique locale facultative ; ne confirme jamais K04 à sa place."""

from collections import deque
from pathlib import Path

import numpy as np
import torch

from .checkpoints import CheckpointStore
from .features import BASE_FEATURES, augmentation, vectorize
from .model import create_model
from .storage import json_file


class LocalTemporalMonitor:
    """Un historique propre par mouvement, sans lecteur de référence physique."""

    def __init__(self, artifact):
        artifact = Path(artifact)
        result = json_file(artifact / "complete.json")
        self.config = json_file(artifact / "config.json")
        torch.set_num_threads(self.config.get("threads", 4))
        self.threshold = json_file(artifact / "calibration.json")["threshold"]
        self.normalization = json_file(artifact / "normalization.json")
        self.count = result["compatibility"]["features"]
        state, receipt = CheckpointStore(
            artifact / "checkpoints", result["compatibility"]
        ).load(best=True)
        self.model = create_model(self.config, self.count)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.model_identity = receipt["receipt"]["sha256"]
        self.previous = {}
        self.histories = {}

    def observe(self, row):
        at = row["observed_at_s"]
        if row["available_at_s"] > at + 1e-7:
            raise ValueError("Estimation demandée avant la disponibilité des mesures.")
        if (
            abs(row["own_state"]["observed_at_s"] - at) > 1e-7
            or row["own_state"].get("available_at_s", at) > at + 1e-7
        ):
            raise ValueError("La mesure propre doit appartenir au même instant causal.")
        key = (row["ego_id"], row["movement_id"])
        previous = self.previous.get(key)
        if previous and at <= previous["observed_at_s"]:
            raise ValueError("Le temps du mouvement doit augmenter.")
        history = self.histories.setdefault(key, deque(maxlen=176))
        if previous and abs(at - previous["observed_at_s"] - 0.2) > 1e-6:
            history.clear()
            previous = None
        controls = dict(
            requested_speed_mps=row["requested_speed_mps"],
            accepted_speed_mps=row["accepted_speed_mps"],
            native_action=row["own_native_action"],
        )
        values = vectorize(
            row["own_state"],
            controls,
            row["lidar"],
            row["received_message_ages_s"],
            previous,
            lidar_available=row["lidar_available"],
            v2v_available=row["v2v_box_available"],
        )
        if self.count > len(BASE_FEATURES):
            values = np.concatenate([values, augmentation(row)])[: self.count]
        history.append(values)
        self.previous[key] = dict(row["own_state"])
        result = dict(
            ego_id=key[0],
            movement_id=key[1],
            available_at_s=at,
            threshold=self.threshold,
            model_sha256=self.model_identity,
            scope="statistical_confirmation_estimate_not_physical_proof",
            action=None,
            changes_K04_status=False,
            history_samples=len(history),
        )
        if len(history) < 26:
            return dict(result, probability=None, status="ABSTAIN_INSUFFICIENT_HISTORY")
        x = np.zeros((176, self.count), np.float32)
        mask = np.zeros((1, 1, 176), np.float32)
        mean = np.asarray(self.normalization["mean"][: self.count], np.float32)
        std = np.asarray(self.normalization["std"][: self.count], np.float32)
        x[-len(history) :] = np.clip((np.stack(history) - mean) / std, -10, 10)
        mask[:, :, -len(history) :] = 1
        with torch.inference_mode():
            probability = float(
                torch.sigmoid(
                    self.model(
                        torch.from_numpy(x.T[None].copy()), torch.from_numpy(mask)
                    )
                )[0]
            )
        return dict(
            result,
            probability=probability,
            status="STATISTICAL_ALERT"
            if probability >= self.threshold
            else "NO_STATISTICAL_ALERT",
        )
