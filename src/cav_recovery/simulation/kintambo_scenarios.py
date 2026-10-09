"""Prépare des vagues ciblées sur les routes canoniques de Kintambo."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

from .traffic_demand import Mission

CASES = Path(__file__).parents[1] / "scenarios/kintambo/cases.json"


def case_names():
    return tuple(json.loads(CASES.read_text(encoding="utf-8")))


def read_case(name):
    data = json.loads(CASES.read_text(encoding="utf-8"))
    if name not in data:
        raise ValueError("Variante Kintambo inconnue.")
    return {"name": name, **data[name], "definition_sha256": hashlib.sha256(CASES.read_bytes()).hexdigest()}


def case_missions(case, routes, seed):
    """Fixe les destinations et départs avant SUMO, sans modifier les routes."""
    if type(seed) is not int or seed < 0:
        raise ValueError("La seed doit être un entier positif ou nul.")
    rng, sequences, missions = random.Random(seed), defaultdict(int), []
    for stream in case["streams"]:
        route_id = stream["route"]
        if route_id not in routes:
            raise ValueError(f"Route canonique absente : {route_id}.")
        route = tuple(routes[route_id])
        origin, destination = route_id.split("__")
        for number in range(stream["count"]):
            sampled = stream["start_s"] + number * stream["headway_s"] + rng.uniform(0, case["jitter_s"])
            vehicle = f"{origin}_{sequences[origin]:06d}"
            sequences[origin] += 1
            missions.append(Mission(vehicle, case["name"], origin, destination, route_id,
                                    route, route[-1], math.ceil(sampled * 1000) / 1000, sampled))
    return sorted(missions, key=lambda mission: (mission.scheduled_s, mission.vehicle_id))
