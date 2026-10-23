"""Configurations de développement physiquement distinctes sur les quatre carrefours M02."""

import copy
import random

from ..dataset.contract import digest
from ..dataset.generation import candidate

SITUATIONS = {
    "nominal": (0, 20, 40),
    "dense_progressing": (16, 36, 56),
    "temporary_interaction": (8, 24, 44),
    "individual_obstruction": (112, 116, 124),
    "partial_observability": (49, 57, 61),
    "natural_release": (64, 128, 192),
    "collective": (48, 52, 56, 60, 96),
}


def catalog():
    entries = []
    for situation, indices in SITUATIONS.items():
        for variant, index in enumerate(indices):
            entries.append(
                dict(
                    name=f"{situation}-{variant + 1}",
                    situation=situation,
                    index=index,
                    seed=21001 + len(entries),
                    variant=variant,
                )
            )
    return entries


def specification(name, seed=None, *, horizon_s=None, vehicle_count=None):
    """Fixe missions, flux et arrivées avant simulation ; aucun label n'est injecté."""
    entry = next((row for row in catalog() if row["name"] == name), None)
    if entry is None:
        raise ValueError("unknown experiment scenario")
    seed = entry["seed"] if seed is None else seed
    random_source = random.Random(seed)
    spec = candidate(entry["index"], phase="calibration")
    template = spec.pop("id")
    variant = entry["variant"]
    situation = entry["situation"]
    if situation == "nominal":
        count, cap, headway, horizon, movements = (
            12 + variant * 2,
            8,
            2.4 + variant * 0.2,
            110.0,
            "right",
        )
    elif situation == "dense_progressing":
        count, cap, headway, horizon, movements = (
            42 + variant * 3,
            30,
            0.6 + variant * 0.1,
            120.0,
            "right",
        )
    elif situation in {"temporary_interaction", "natural_release"}:
        count, cap, headway, horizon, movements = (
            12 + variant * 2,
            12,
            1.6 + variant * 0.3,
            120.0,
            "mixed",
        )
    else:
        count, cap, headway, horizon, movements = (
            42 + variant,
            30,
            0.65 + variant * 0.06,
            140.0,
            "conflicting",
        )
    if vehicle_count is not None:
        if not isinstance(vehicle_count, int) or not 2 <= vehicle_count <= 30:
            raise ValueError("Effectif de la scène hors domaine.")
        count, cap = vehicle_count, min(vehicle_count, cap)
    templates = copy.deepcopy(spec["agents"])
    agents = []
    lane_count = spec["topology"]["lane_num"]
    for index in range(count):
        agent = copy.deepcopy(templates[index % len(templates)])
        origin = index % spec["topology"]["arms"]
        choices = [
            road for road in spec["roads"] if road["start"] == f"a{origin}_entry"
        ]
        kinds = (
            {"right"}
            if movements == "right"
            else {"left", "straight"}
            if movements == "conflicting"
            else {"right", "left", "straight"}
        )
        choices = [road for road in choices if road["kind"] in kinds]
        if not choices:
            choices = [
                road for road in spec["roads"] if road["start"] == f"a{origin}_entry"
            ]
        road = random_source.choice(choices)
        kind = road["kind"]
        destination = int(road["end"].split("_")[0][1:])
        lane = (
            0
            if kind == "left"
            else lane_count - 1
            if kind == "right"
            else min(1, lane_count - 1)
        )
        agent.update(
            id=f"agent{index}",
            origin=origin,
            destination=destination,
            lane=lane,
            movement_id=f"a{origin}_a{destination}_l{lane}",
            movement_kind=kind,
            departure_s=round(
                index * headway + (random_source.uniform(0.0, 0.25) if index else 0.0),
                3,
            ),
            target_speed_mps=round(random_source.uniform(3.2, 5.0), 4),
            reaction_s=round(random_source.uniform(0.25, 0.55), 4),
            clearance_m=round(random_source.uniform(0.55, 0.9), 4),
        )
        agents.append(agent)
    if (situation == "collective" and variant == 4) or situation == "natural_release":
        # Un flux transversal soutenu et une file concurrente peuvent produire
        # une chaîne sans cycle ; aucun véhicule ne reçoit une consigne d'arrêt.
        stream_count = (max(2, count - 2) if vehicle_count is not None else
                        34 if situation == "collective" else 18 + 4 * variant)
        competing_count = (count - stream_count if vehicle_count is not None else 6)
        arrivals = [
            (index * 1.2, 1, 3, index % 2, 4.6, 12.0) for index in range(stream_count)
        ]
        arrivals += [
            (2.0 + index * 2.0, 0, 2, min(1, lane_count - 1), 3.6, 4.0)
            for index in range(competing_count)
        ]
        agents = []
        for index, (
            departure,
            origin,
            destination,
            lane,
            speed,
            longitude,
        ) in enumerate(sorted(arrivals)):
            agent = copy.deepcopy(templates[index % len(templates)])
            agent.update(
                id=f"agent{index}",
                origin=origin,
                destination=destination,
                lane=lane,
                movement_id=f"a{origin}_a{destination}_l{lane}",
                movement_kind="straight",
                departure_s=round(departure, 3),
                target_speed_mps=speed,
                longitude=longitude,
            )
            agents.append(agent)
        movements = "sustained_two_lane_transverse_stream_and_competing_queue"
    spec.update(
        agents=agents,
        active_limit=cap,
        density_level=cap,
        horizon_s=horizon if horizon_s is None else horizon_s,
        seed=seed,
        phase="integrated_development",
        parent_id="integrated:" + name,
        experiment_name=name,
        requested_situation=situation,
        geometry_template_id=template,
        nominal_controller="local_forward_clearance_v4",
        development_only=True,
        movement_profile=movements,
        demand=dict(headway_s=headway, profile=movements),
    )
    if vehicle_count is not None:
        spec["construction"] = "direct_agent_generation"
    spec["id"] = digest(spec)[:24]
    return spec
