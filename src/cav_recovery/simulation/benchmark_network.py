"""Conversion reproductible de la topologie OSM en réseau expérimental."""

import gzip
import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET


SCENARIO_DIR = Path(__file__).parents[1] / "scenarios" / "kintambo"
SOURCE_SHA256 = "93a31d50a909221924ef820747d87035bbea5451a3bdd339f689d3783e40c45b"
SUMO_VERSION = "1.27.1"


def write_xml(path: Path, root: ET.Element) -> None:
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def read_config(path: str | Path | None = None) -> dict:
    config = json.loads(Path(path or SCENARIO_DIR / "experiment.json").read_text(encoding="utf-8"))
    if config["scenario"] != "kintambo":
        raise ValueError("Le scénario attendu est kintambo.")
    for group in ("major", "intermediate", "minor"):
        lanes = config["network"]["lanes_per_direction"][group]
        speed = config["network"]["speed_kmh"][group]
        if type(lanes) is not int or lanes < 1 or not math.isfinite(speed) or speed <= 0:
            raise ValueError("Nombre de voies ou vitesse invalide.")
    for value in (config["network"]["cycle_s"], config["network"]["signal_radius_m"],
                  config["demand"]["duration_s"], config["simulation"]["drain_horizon_s"]):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Durée ou rayon invalide.")
    if config["simulation"]["step_s"] != 0.5:
        raise ValueError("Le suivi utilise un pas de 0,5 s.")
    return config


def require_binary(name: str) -> str:
    binary = shutil.which(name)
    if binary is None:
        raise RuntimeError(f"{name} introuvable ; activer l'environnement Python du projet.")
    version = subprocess.run([binary, "--version"], capture_output=True, text=True, check=True)
    match = re.search(r"\b\d+\.\d+\.\d+\b", version.stdout)
    if match is None or match.group() != SUMO_VERSION:
        raise RuntimeError(f"{name} {SUMO_VERSION} est requis.")
    for package in ("traci", "sumolib", "eclipse-sumo"):
        if importlib.metadata.version(package) != SUMO_VERSION:
            raise RuntimeError(f"{package} {SUMO_VERSION} est requis.")
    return binary


def adapt_roads(config: dict, target: Path) -> None:
    """Conserve les positions et sens OSM, mais déclare des voies synthétiques."""
    data = gzip.decompress((SCENARIO_DIR / "roads.osm.gz").read_bytes())
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("L'extrait OSM distribué a été modifié.")
    root = ET.fromstring(data)
    for way in root.findall("way"):
        tags = {t.get("k"): t.get("v") for t in way.findall("tag")}
        highway = tags["highway"].split("_link")[0]
        group = ("major" if highway in ("motorway", "trunk", "primary", "secondary")
                 else "intermediate" if highway in ("tertiary", "unclassified") else "minor")
        lanes = config["network"]["lanes_per_direction"][group]
        # Les marquages par voie de l'OSM ne décrivent plus les voies adaptées.
        for tag in list(way.findall("tag")):
            key = tag.get("k")
            if key.startswith(("lanes", "turn:lanes", "maxspeed", "width", "change:lanes")):
                way.remove(tag)
        one_way = tags.get("oneway") in ("yes", "true", "1", "-1") or tags.get("junction") == "roundabout"
        ET.SubElement(way, "tag", k="lanes", v=str(lanes if one_way else 2 * lanes))
        if not one_way:
            ET.SubElement(way, "tag", k="lanes:forward", v=str(lanes))
            ET.SubElement(way, "tag", k="lanes:backward", v=str(lanes))
        ET.SubElement(way, "tag", k="maxspeed", v=str(config["network"]["speed_kmh"][group]))
    write_xml(target, root)


def build_network(config: dict, directory: Path) -> tuple[dict, dict]:
    """Regroupe les raccords décrits et ajoute des feux fixes, avec keepClear."""
    import sumolib

    source = directory / "adapted.osm"
    adapt_roads(config, source)
    joins = ET.Element("nodes")
    for name, nodes in config["network"]["junction_groups"].items():
        ET.SubElement(joins, "join", id=name, nodes=" ".join(nodes))
    write_xml(directory / "junctions.nod.xml", joins)
    binary = require_binary("netconvert")
    base = [binary, "--osm-files", str(source), "--keep-edges.by-vclass", "passenger",
            "--node-files", str(directory / "junctions.nod.xml"),
            "--output.street-names", "true", "--output.original-names", "true",
            "--osm.annotate-defaults", "true", "--tls.default-type", "static",
            "--tls.cycle.time", str(config["network"]["cycle_s"]),
            "--tls.join", "false", "--junctions.join", "true",
            "--junctions.join-dist", str(config["network"]["junction_join_distance_m"]),
            "--no-turnarounds", "true", "--seed", "0"]
    preliminary = directory / "unsignalized.net.xml"
    with (directory / "preliminary.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(base + ["--output-file", str(preliminary)], stdout=log, stderr=log)
    if completed.returncode:
        raise RuntimeError("Conversion initiale échouée ; consulter preliminary.log.")
    net = sumolib.net.readNet(str(preliminary))
    center = net.convertLonLat2XY(*config["center_lon_lat"])
    signals = sorted(node.getID() for node in net.getNodes()
                     if math.dist(node.getCoord(), center) < config["network"]["signal_radius_m"]
                     and len(node.getIncoming()) >= 3 and len(node.getOutgoing()) >= 2)
    if not signals:
        raise ValueError("Aucun carrefour à plusieurs approches dans la zone.")
    command = base + ["--tls.set", ",".join(signals), "--tls.allred.time", str(config["network"]["all_red_s"]),
                      "--tls.layout", config["network"]["signal_layout"],
                      "--tls.minor-left.max-speed", "0",
                      "--output-file", str(directory / "network.net.xml")]
    with (directory / "conversion.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=log)
    if completed.returncode:
        raise RuntimeError("Conversion échouée ; consulter conversion.log.")
    # Les commentaires contiennent les chemins de sortie et une date variable.
    write_xml(directory / "network.net.xml", ET.parse(directory / "network.net.xml").getroot())
    net = sumolib.net.readNet(str(directory / "network.net.xml"), withPrograms=True)
    routes = build_routes(net, config)
    controllers = [{"id": row.getID(), "programs": [
        {"id": program_id, "type": p.getType(), "offset": p.getOffset(),
         "phases": [{"duration_s": phase.duration, "state": phase.state} for phase in p.getPhases()]}
        for program_id, p in row.getPrograms().items()]} for row in net.getTrafficLights()]
    inspection = {
        "source_sha256": SOURCE_SHA256, "conversion_command": command,
        "nodes": len(net.getNodes()), "edges": len(net.getEdges()),
        "lanes": sum(e.getLaneNumber() for e in net.getEdges()),
        "branching_junctions": sum(len(n.getIncoming()) >= 3 for n in net.getNodes()),
        "controllers": controllers, "center_xy_m": list(center),
        "edges_detail": [{"id": e.getID(), "from": e.getFromNode().getID(),
                          "to": e.getToNode().getID(), "lanes": e.getLaneNumber(),
                          "length_m": e.getLength(), "speed_m_per_s": e.getSpeed(),
                          "name": e.getName()} for e in net.getEdges()],
    }
    inspection["volatile_xml_comments_removed"] = True
    return inspection, routes


def build_routes(net, config: dict) -> dict:
    """Relie chaque OD par un itinéraire passenger passant dans le noyau central.

    Un passage par une edge centrale est imposé, pas une connexion inventée.
    Parmi ces chemins, on retient le plus court sans répétition ni demi-tour.
    """
    center = net.convertLonLat2XY(*config["center_lon_lat"])
    core = sorted((e for e in net.getEdges() if e.allows("passenger")
                   and math.dist(e.getToNode().getCoord(), center) < 100), key=lambda e: e.getID())
    routes = {}
    entries = config["demand"]["entries"]
    for origin, destinations in sorted(config["demand"]["destination_weights"].items()):
        start = net.getEdge(entries[origin]["edge"])
        for destination in sorted(destinations):
            end = net.getEdge(entries[destination]["exit"])
            candidates = []
            for middle in core:
                first, cost1 = net.getShortestPath(start, middle, vClass="passenger")
                last, cost2 = net.getShortestPath(middle, end, vClass="passenger")
                if first is None or last is None:
                    continue
                path = first + last[1:]
                ids = tuple(e.getID() for e in path)
                if len(ids) != len(set(ids)):
                    continue
                if all(any(c.getDirection() != "t" and c.getFromLane().allows("passenger")
                           and c.getToLane().allows("passenger") for c in a.getOutgoing().get(b, []))
                       for a, b in zip(path, path[1:])):
                    candidates.append((cost1 + cost2, ids))
            if not candidates:
                raise ValueError(f"Aucune route légale centrale : {origin} → {destination}.")
            routes[f"{origin}__{destination}"] = list(min(candidates)[1])
    return routes
