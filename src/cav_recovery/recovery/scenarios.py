"""Progression de développement, distincte des partitions de test réservées."""

import copy
from ..dataset.generation import candidate
from ..dataset.contract import digest


def crossing(
    count=2, *, seed=41001, movement="straight", horizon_s=130.0, synchronization_m=0.0
):
    """Deux flux arrivent en mouvement ; aucune pose de blocage n'est imposée."""
    if count not in (2, 4, 6, 8, 12, 20) or movement not in {"straight", "left"}:
        raise ValueError("Configuration de développement non prise en charge.")
    spec = candidate(48, phase="calibration")
    template = spec["agents"][0]
    agents = []
    for index in range(count):
        origin = index % 2 if movement == "straight" else 2 * (index % 2)
        destination = (origin + 2) % 4 if movement == "straight" else (origin + 3) % 4
        agent = copy.deepcopy(template)
        agent.update(
            id=f"agent{index}",
            origin=origin,
            destination=destination,
            lane=0,
            movement_id=f"a{origin}_a{destination}_l0",
            movement_kind=movement,
            departure_s=round((index // 2) * 4.0, 3),
            longitude=spec["topology"]["approach_length_m"]
            - 8.0
            + (
                spec["topology"]["lane_width_m"] + synchronization_m
                if index % 2 == 0 and movement == "straight"
                else 0.0
            ),
            target_speed_mps=2.0,
        )
        agents.append(agent)
    spec.update(
        agents=agents,
        active_limit=count,
        density_level=count,
        seed=seed,
        horizon_s=horizon_s,
        phase="recovery_constructive_development",
        parent_id=f"recovery-crossing-{movement}",
        experiment_name=f"constructive-{movement}-{count}",
        development_only=True,
    )
    spec["id"] = digest(spec)[:24]
    return spec
