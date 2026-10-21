"""Variables locales autorisées ; identités et références ne sont jamais prédictives."""

import math
import numpy as np

BASE_FEATURES = (
    "signed_speed_mps",
    "longitudinal_acceleration_mps2",
    "lateral_acceleration_mps2",
    "acceleration_available",
    "progress_delta_m",
    "previous_sample_available",
    "yaw_rate_radps",
    "requested_speed_mps",
    "accepted_speed_mps",
    "steering_action",
    "longitudinal_action",
    "controls_available",
    *(f"lidar_{index:02d}" for index in range(16)),
    "lidar_available",
    "v2v_available",
    "received_count",
    "minimum_received_age_s",
    "maximum_received_age_s",
)
DIAGNOSTIC_FEATURES = ("d01_no_alert", "d01_suspected", "d01_confirmed", "d01_unknown")
GRAPH_FEATURES = ("current_local_relations", "observed_local_collective_groups")
STATUSES = ("NO_ALERT", "SUSPECTED", "CONFIRMED_OBSTRUCTION", "UNKNOWN")


def received_ages(mailbox, at_s):
    if mailbox is None:
        return []
    values = mailbox.get("received", {})
    entries = values.values() if isinstance(values, dict) else values
    result = []
    for entry in entries:
        message = entry["message"]
        observed = message["observed_at_s"]
        received = entry["received_at_s"]
        if not 0 <= observed <= received <= at_s + 1e-7:
            raise ValueError("Un message futur ne peut pas entrer dans les variables.")
        if at_s <= message["valid_until_s"] + 1e-7:
            result.append(max(0.0, at_s - observed))
    return result


def vectorize(
    state,
    controls,
    lidar,
    ages,
    previous=None,
    *,
    lidar_available=True,
    v2v_available=True,
):
    """Représentation invariante aux identifiants, translations et rotations.

    Le passé propre sert uniquement aux différences ; aucun état global ou
    diagnostic D01/C-RDG ne peut entrer par cette fonction.
    """
    at_s = state["observed_at_s"]
    if state.get("available_at_s", at_s) > at_s + 1e-7:
        raise ValueError("Mesure propre disponible dans le futur.")
    previous_ok = (
        previous is not None and abs(at_s - previous["observed_at_s"] - 0.2) < 1e-6
    )
    acceleration = state.get("acceleration_xy_mps2")
    heading = state["heading_rad"]
    if acceleration is None:
        longitudinal = lateral = 0.0
    else:
        ax, ay = acceleration
        longitudinal = ax * math.cos(heading) + ay * math.sin(heading)
        lateral = -ax * math.sin(heading) + ay * math.cos(heading)
    progress = (
        state["mission_progress_m"] - previous["mission_progress_m"]
        if previous_ok
        else 0.0
    )
    yaw = (
        math.remainder(heading - previous["heading_rad"], 2 * math.pi) / 0.2
        if previous_ok
        else 0.0
    )
    controls = controls or {}
    action = controls.get("action", controls.get("native_action", (0.0, 0.0)))
    rays = list(lidar) if lidar is not None else [None] * 16
    if len(rays) != 16:
        raise ValueError("Le schéma requiert 16 rayons LiDAR.")
    rays = [1.0 if value is None else float(value) for value in rays]
    if any(not 0 <= r <= 1 for r in rays):
        raise ValueError("Rayon LiDAR hors du domaine normalisé.")
    result = [
        state["signed_speed_mps"],
        longitudinal,
        lateral,
        float(acceleration is not None),
        progress,
        float(previous_ok),
        yaw,
        controls.get("requested_speed_mps", 0.0),
        controls.get("accepted_speed_mps", 0.0),
        *action,
        float(bool(controls)),
        *rays,
        float(lidar_available),
        float(v2v_available),
        len(ages),
        min(ages, default=0.0),
        max(ages, default=0.0),
    ]
    array = np.asarray(result, dtype=np.float32)
    if (
        len(array) != len(BASE_FEATURES)
        or not np.isfinite(array).all()
        or any(age < 0 for age in ages)
    ):
        raise ValueError("Variables locales invalides.")
    return array


def augmentation(row):
    """Extension explicitement réservée aux ablations D01 puis D01+C-RDG."""
    status = row["local_status"]
    if status not in STATUSES:
        raise ValueError("Statut D01 inconnu.")
    return np.asarray(
        [
            *(float(status == value) for value in STATUSES),
            row["current_received_relations"],
            row["observed_collective_groups"],
        ],
        dtype=np.float32,
    )
