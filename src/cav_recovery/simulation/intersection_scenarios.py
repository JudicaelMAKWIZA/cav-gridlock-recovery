"""Prépare de petits croisements contrôlés, séparés du réseau de Kintambo."""

import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import xml.etree.ElementTree as ET

from .road_network import require_binary, write_xml
from .traffic_demand import Mission

DEFINITIONS = Path(__file__).parents[1] / "scenarios/intersections/cases.json"


def scenario_names() -> tuple[str, ...]:
    return tuple(json.loads(DEFINITIONS.read_text(encoding="utf-8"))["cases"])


def read_scenario(name: str) -> dict:
    """Lit le mécanisme visé et ses paramètres, sans lui attribuer un résultat."""
    data = json.loads(DEFINITIONS.read_text(encoding="utf-8"))
    if name not in data["cases"]:
        raise ValueError("Scénario d'intersection inconnu.")
    return {"name": name, "case": data["cases"][name], "crdg": data["crdg"],
            "simulation": {"drain_horizon_s": data["cases"][name]["drain_s"]},
            "definition_sha256": hashlib.sha256(DEFINITIONS.read_bytes()).hexdigest()}


def build_intersection(config: dict, output: Path, seed: int):
    """Construit le réseau natif et des missions finies à départs explicites."""
    if type(seed) is not int or seed < 0:
        raise ValueError("La seed doit être un entier positif ou nul.")
    case = config["case"]
    nodes, edges = ET.Element("nodes"), ET.Element("edges")
    arm = case["arm_length_m"]
    double = case["geometry"] == "double_cross"
    centres = {"Jw": (0, 0), "Je": (case["link_length_m"], 0)} if double else {"J": (0, 0)}
    for name, (x, y) in centres.items():
        ET.SubElement(nodes, "node", id=name, x=str(x), y=str(y), type=case["junction_type"], keepClear="true")
    if double:
        link = case["link_length_m"]
        approaches = {"west": ("Jw", -arm, 0), "wnorth": ("Jw", 0, arm), "wsouth": ("Jw", 0, -arm),
                      "east": ("Je", link + arm, 0), "enorth": ("Je", link, arm), "esouth": ("Je", link, -arm)}
    else:
        approaches = {"west": ("J", -arm, 0), "east": ("J", arm, 0),
                      "north": ("J", 0, arm), "south": ("J", 0, -arm)}
    for name, (centre, x, y) in approaches.items():
        ET.SubElement(nodes, "node", id=name, x=str(x), y=str(y))
        for direction, source, target in (("in", name, centre), ("out", centre, name)):
            ET.SubElement(edges, "edge", id=name + "_" + direction, **{"from": source, "to": target},
                          numLanes="1", speed=str(case["speed_m_per_s"]),
                          priority="3" if name in ("west", "east") else "1",
                          name="Approche " + {"west": "ouest", "east": "est", "north": "nord", "south": "sud",
                          "wnorth": "nord-ouest", "wsouth": "sud-ouest", "enorth": "nord-est", "esouth": "sud-est"}[name])
    if double:
        for source, target, name in (("Jw", "Je", "west_east"), ("Je", "Jw", "east_west")):
            ET.SubElement(edges, "edge", id=name, **{"from": source, "to": target}, numLanes="1",
                          speed=str(case["speed_m_per_s"]), priority="3", name="Liaison centrale")
    write_xml(output / "nodes.nod.xml", nodes)
    write_xml(output / "edges.edg.xml", edges)
    command = [require_binary("netconvert"), "--node-files", str(output / "nodes.nod.xml"),
               "--edge-files", str(output / "edges.edg.xml"), "--output-file", str(output / "network.net.xml"),
               "--no-turnarounds", "true", "--output.street-names", "true", "--tls.cycle.time", "90"]
    conversion = subprocess.run(command, capture_output=True, text=True)
    (output / "netconvert.log").write_text(conversion.stdout + conversion.stderr, encoding="utf-8")
    if conversion.returncode:
        raise RuntimeError("Conversion du croisement impossible ; consulter netconvert.log.")
    # La date et les chemins du commentaire netconvert ne décrivent pas le réseau.
    write_xml(output / "network.net.xml", ET.parse(output / "network.net.xml").getroot())
    import sumolib
    net = sumolib.net.readNet(str(output / "network.net.xml"), withInternal=True)
    routes, missions, positions = {}, [], {}
    rng = random.Random(seed)
    for index, stream in enumerate(case["streams"]):
        route_id = f"movement_{index}"
        route = tuple(stream["route"])
        for a, b in zip(route, route[1:]):
            if net.getEdge(b) not in net.getEdge(a).getOutgoing():
                raise ValueError("Mouvement non raccordé à sa destination.")
            if not any(c.getFromLane().allows("passenger") and c.getToLane().allows("passenger")
                       for c in net.getEdge(a).getOutgoing()[net.getEdge(b)]):
                raise ValueError("Mouvement interdit aux véhicules passenger.")
        routes[route_id] = list(route)
        for number in range(stream["count"]):
            sampled = stream["start_s"] + number * stream["headway_s"] + rng.uniform(0, case["jitter_s"])
            item = f"{route[0].removesuffix('_in')}_{number:06d}"
            missions.append(Mission(item, config["name"], route[0], route[-1], route_id, route, route[-1],
                                    math.ceil(sampled * 1000) / 1000, sampled))
            if "initial_distance_to_end_m" in case:
                position = net.getEdge(route[0]).getLanes()[0].getLength() - case["initial_distance_to_end_m"]
                if position < 5:
                    raise ValueError("La position initiale ne contient pas la carrosserie.")
                positions[item] = position
    missions.sort(key=lambda m: (m.scheduled_s, m.vehicle_id))
    if len({m.vehicle_id for m in missions}) != len(missions):
        raise ValueError("Les missions du scénario doivent avoir des IDs uniques.")
    bounds = net.getBoundary()
    inspection = {"view_boundary_m": list(bounds), "junction_count": len(net.getNodes()),
                  "edge_count": len(net.getEdges()), "lane_count": sum(len(e.getLanes()) for e in net.getEdges()),
                  "controller_count": len(net.getTrafficLights()), "conversion_command": command}
    return inspection, routes, missions, positions
