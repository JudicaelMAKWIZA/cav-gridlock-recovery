"""Fenêtres causales par mouvement, utilisables ultérieurement sans entraîner de TCN."""

from collections import deque
import math

from ..d01.inputs import freeze


def causal_windows(records, history_samples=176, sample_s=0.2):
    """Expose le passé observé et son masque ; une lacune ne devient jamais une mesure.

    Les identifiants servent aux jointures et aux partitions, pas aux variables
    prédictives. Les statistiques de normalisation devront être ajustées sur
    train uniquement lors d'une future étude, séparée de cette interface.
    """
    histories = {}
    times = {}
    fields = {
        "ego_id",
        "movement_id",
        "observed_at_s",
        "available_at_s",
        "own_state",
        "lidar",
        "requested_speed_mps",
        "accepted_speed_mps",
        "own_native_action",
        "received_message_ages_s",
        "v2v_box_available",
        "lidar_available",
        "local_status",
        "current_received_relations",
        "observed_collective_groups",
        "scope",
    }
    own_fields = {
        "observed_at_s",
        "available_at_s",
        "position_xy_m",
        "heading_rad",
        "signed_speed_mps",
        "velocity_xy_mps",
        "acceleration_xy_mps2",
        "mission_progress_m",
        "length_m",
        "width_m",
    }
    if (
        type(history_samples) is not int
        or history_samples < 1
        or not math.isfinite(sample_s)
        or sample_s <= 0
    ):
        raise ValueError("invalid causal window configuration")
    for row in records:
        if not set(row) <= fields or not set(row.get("own_state", {})) <= own_fields:
            raise ValueError(
                "reference or unsupported fields cannot enter causal features"
            )
        key = (row["ego_id"], row["movement_id"])
        at_s = row["observed_at_s"]
        if not math.isfinite(at_s) or not math.isfinite(row["available_at_s"]):
            raise ValueError("finite causal feature times required")
        if row["available_at_s"] > at_s + 1e-7:
            raise ValueError("feature was not available at the requested causal time")
        history = histories.setdefault(key, deque(maxlen=history_samples))
        if key in times:
            if at_s <= times[key]:
                raise ValueError(
                    "causal features must have increasing times per movement"
                )
            if abs(at_s - times[key] - sample_s) > 1e-7:
                history.clear()
        times[key] = at_s
        history.append(freeze(row))
        yield dict(
            ego_id=key[0],
            movement_id=key[1],
            through_s=at_s,
            samples=tuple(history),
            observed_mask=(False,) * (history_samples - len(history))
            + (True,) * len(history),
            target_included=False,
        )
