"""Prépare des vagues ciblées sur Kintambo et les variantes déclarées."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET

from .traffic_demand import Mission

CASES = Path(__file__).parents[1] / "scenarios/kintambo/cases.json"


def case_names():
    return tuple(json.loads(CASES.read_text(encoding="utf-8")))


def read_case(name):
    data = json.loads(CASES.read_text(encoding="utf-8"))
    if name not in data:
        raise ValueError("Variante Kintambo inconnue.")
    case = data[name]
    if "same_demand_as" in case:
        case = {**data[case["same_demand_as"]], **case}
    return {"name": name, **case, "definition_sha256": hashlib.sha256(CASES.read_bytes()).hexdigest()}


def prepare_case_network(case, routes, path):
    """Valide les chemins avant départ et applique seulement la variante déclarée."""
    import sumolib
    from .road_network import write_xml
    net = sumolib.net.readNet(str(path))
    for name, route in case.get("routes", {}).items():
        if name in routes:
            raise ValueError("Une route de variante ne doit pas écraser une route canonique.")
        for a, b in zip(route, route[1:]):
            links = net.getEdge(a).getOutgoing().get(net.getEdge(b), [])
            if not any(link.getDirection() != "t" and link.getFromLane().allows("passenger")
                       and link.getToLane().allows("passenger") for link in links):
                raise ValueError(f"Chemin de variante non raccordé : {a} → {b}.")
        routes[name] = list(route)
    variant = case.get("network_variant")
    if not variant:
        return None
    if variant["keep_clear"] is not False:
        raise ValueError("La seule variante réseau prise en charge est keepClear=false sur les mouvements déclarés.")
    root = ET.parse(path).getroot()
    movements = {tuple(pair) for pair in variant["movements"]}
    connections = [connection for connection in root.findall("connection")
                   if (connection.get("from"), connection.get("to")) in movements]
    if {(c.get("from"), c.get("to")) for c in connections} != movements:
        raise ValueError("Mouvement keepClear absent du réseau canonique.")
    original_sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    shutil.copyfile(path, Path(path).with_name("canonical.net.xml"))
    changes = []
    for connection in connections:
        changes.append({"from": connection.get("from"), "to": connection.get("to"),
                        "fromLane": connection.get("fromLane"), "toLane": connection.get("toLane"),
                        "previous_keepClear": connection.get("keepClear")})
        connection.set("keepClear", "false")
    write_xml(path, root)
    return {"canonical_sha256": original_sha, "keepClear": False, "changes": changes,
            "justification": variant["justification"]}


def case_missions(case, routes, seed):
    """Fixe les destinations, routes et départs avant SUMO."""
    if type(seed) is not int or seed < 0:
        raise ValueError("La seed doit être un entier positif ou nul.")
    rng, sequences, missions = random.Random(seed), defaultdict(int), []
    for stream in case["streams"]:
        route_id = stream["route"]
        if route_id not in routes:
            raise ValueError(f"Route de la variante absente : {route_id}.")
        route = tuple(routes[route_id])
        origin, destination = route_id.split("__")
        destination = stream.get("exit", destination)
        for number in range(stream["count"]):
            sampled = stream["start_s"] + number * stream["headway_s"] + rng.uniform(0, case["jitter_s"])
            vehicle = f"{origin}_{sequences[origin]:06d}"
            sequences[origin] += 1
            missions.append(Mission(vehicle, case["name"], origin, destination, route_id,
                                    route, route[-1], math.ceil(sampled * 1000) / 1000, sampled))
    return sorted(missions, key=lambda mission: (mission.scheduled_s, mission.vehicle_id))
