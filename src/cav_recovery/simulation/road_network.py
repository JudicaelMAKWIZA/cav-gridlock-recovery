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
SUMO_VERSION = "1.27.1"


def write_xml(path: Path, root: ET.Element) -> None:
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def read_config(path: str | Path | None = None) -> dict:
    config = json.loads(Path(path or SCENARIO_DIR / "scenario.json").read_text(encoding="utf-8"))
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
    groups = config["network"]["active_junctions"]
    if not groups or any(not isinstance(nodes, list) or not nodes for nodes in groups.values()):
        raise ValueError("La sélection des jonctions actives doit être explicite et non vide.")
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
    compressed = (SCENARIO_DIR / "roads.osm.gz").read_bytes()
    data = gzip.decompress(compressed)
    if (hashlib.sha256(compressed).hexdigest() != config["source"]["roads_sha256"]
            or hashlib.sha256(data).hexdigest() != config["source"]["roads_sha256_uncompressed"]):
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


def build_network(config: dict, directory: Path, *, vehicle_space_m: float = 7.5) -> tuple[dict, dict]:
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
    active = select_active_edges(net, config["network"]["active_junctions"])
    focused = directory / "focused.net.xml"
    focus_command = [binary, "--sumo-net-file", str(preliminary),
                     "--keep-edges.explicit", ",".join(active),
                     "--offset.disable-normalization", "true", "--no-turnarounds", "true",
                     "--output-file", str(focused)]
    with (directory / "focus.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(focus_command, stdout=log, stderr=log)
    if completed.returncode:
        raise RuntimeError("Découpe du réseau échouée ; consulter focus.log.")
    net = sumolib.net.readNet(str(focused))
    center = net.convertLonLat2XY(*config["center_lon_lat"])
    signals = sorted(node.getID() for node in net.getNodes()
                     if math.dist(node.getCoord(), center) < config["network"]["signal_radius_m"]
                     and len(node.getIncoming()) >= 3 and len(node.getOutgoing()) >= 2)
    if not signals:
        raise ValueError("Aucun carrefour à plusieurs approches dans la zone.")
    command = [binary, "--sumo-net-file", str(focused), "--tls.rebuild", "true",
                      "--offset.disable-normalization", "true", "--tls.default-type", "static",
                      "--tls.cycle.time", str(config["network"]["cycle_s"]), "--tls.join", "false",
                      "--no-turnarounds", "true", "--seed", "0",
                      "--tls.set", ",".join(signals), "--tls.allred.time", str(config["network"]["all_red_s"]),
                      "--tls.layout", config["network"]["signal_layout"],
                      "--tls.minor-left.max-speed", "0",
                      "--output-file", str(directory / "network.net.xml")]
    with (directory / "conversion.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=log)
    if completed.returncode:
        raise RuntimeError("Conversion échouée ; consulter conversion.log.")
    # Les commentaires contiennent les chemins de sortie et une date variable.
    write_xml(directory / "network.net.xml", ET.parse(directory / "network.net.xml").getroot())
    corrections = _correct_connections(config, directory, vehicle_space_m)
    net = sumolib.net.readNet(str(directory / "network.net.xml"), withPrograms=True)
    center = net.convertLonLat2XY(*config["center_lon_lat"])
    routes = build_routes(net, config)
    controllers = [{"id": row.getID(), "programs": [
        {"id": program_id, "type": p.getType(), "offset": p.getOffset(),
         "phases": [{"duration_s": phase.duration, "state": phase.state} for phase in p.getPhases()]}
        for program_id, p in row.getPrograms().items()]} for row in net.getTrafficLights()]
    inspection = {
        "source": config["source"], "conversion_command": command,
        "focus_command": focus_command, "active_edges": active,
        "connection_rules": corrections,
        "nodes": len(net.getNodes()), "edges": len(net.getEdges()),
        "lanes": sum(e.getLaneNumber() for e in net.getEdges()),
        "branching_junctions": sum(len(n.getIncoming()) >= 3 for n in net.getNodes()),
        "controllers": controllers, "center_xy_m": list(center),
        "edges_detail": [{"id": e.getID(), "from": e.getFromNode().getID(),
                          "to": e.getToNode().getID(), "lanes": e.getLaneNumber(),
                          "length_m": e.getLength(), "speed_m_per_s": e.getSpeed(),
                          "name": e.getName()} for e in net.getEdges()],
    }
    corners = [net.getNode(node).getCoord() for node in config["network"]["active_junctions"]["magasin"]]
    inspection["view_boundary_m"] = [min(p[0] for p in corners) - 150, min(p[1] for p in corners) - 150,
                                     max(p[0] for p in corners) + 150, max(p[1] for p in corners) + 150]
    points = [node.getCoord() for node in net.getNodes()]
    inspection["extent_m"] = [max(p[0] for p in points) - min(p[0] for p in points),
                              max(p[1] for p in points) - min(p[1] for p in points)]
    inspection["volatile_xml_comments_removed"] = True
    return inspection, routes


def select_active_edges(net, groups: dict) -> list[str]:
    """Conserve les routes reliant les jonctions retenues pour leur rôle local."""
    selected = {node for nodes in groups.values() for node in nodes}
    missing = selected - {node.getID() for node in net.getNodes()}
    if missing:
        raise ValueError(f"Jonctions actives absentes de la source convertie : {sorted(missing)}")
    edges = sorted(e.getID() for e in net.getEdges()
                   if e.getFromNode().getID() in selected and e.getToNode().getID() in selected)
    if not edges:
        raise ValueError("La sélection active ne contient aucune route.")
    connected = {node.getID() for e in net.getEdges() if e.getID() in edges
                 for node in (e.getFromNode(), e.getToNode())}
    if connected != selected:
        raise ValueError(f"Jonctions sélectionnées sans route active : {sorted(selected - connected)}")
    return edges


def _connection_patches(root: ET.Element, vehicle_space_m: float, anchors: set[str]) -> tuple:
    """Prépare les voies avant les raccords où un changement latéral ne tient pas.

    On ne crée aucun nouveau mouvement entre routes : seules les voies d'un
    mouvement déjà permis sont raccordées aux voies nécessaires en aval.
    Le CAV choisit sa voie avant la traversée, sans saut latéral dans le conflit.
    """
    edges = {e.get("id"): e for e in root.findall("edge") if e.get("function") != "internal"}
    connections = [c for c in root.findall("connection")
                   if c.get("from") in edges and c.get("to") in edges]
    edge_rules, connection_rules = ET.Element("edges"), ET.Element("connections")
    existing = {(c.get("from"), c.get("to"), int(c.get("fromLane")), int(c.get("toLane")))
                for c in connections}
    # SUMO recommande de conserver l'exception emergency. Nos CAV sont passenger.
    for connection in connections:
        ET.SubElement(connection_rules, "connection",
                      **{k: connection.get(k) for k in ("from", "to", "fromLane", "toLane")},
                      changeLeft="emergency", changeRight="emergency", contPos="0")
    short, added = [], []
    for target, edge in edges.items():
        # Le raccord doit offrir deux emplacements nominaux complets : entrée
        # et réception du véhicule. Sinon, sa voie est préparée avant le virage.
        if float(edge.find("lane").get("length")) >= 2 * vehicle_space_m or len(edge.findall("lane")) < 2:
            continue
        short.append(target)
        row = ET.SubElement(edge_rules, "edge", id=target)
        for lane in edge.findall("lane"):
            ET.SubElement(row, "lane", index=lane.get("index"),
                          changeLeft="emergency", changeRight="emergency")
        needed = {int(c.get("fromLane")) for c in connections if c.get("from") == target}
        incoming = {}
        for connection in connections:
            if connection.get("to") == target:
                incoming.setdefault(connection.get("from"), []).append(connection)
        for origin, candidates in incoming.items():
            for to_lane in sorted(needed):
                if any(int(c.get("toLane")) == to_lane for c in candidates):
                    continue
                selected = min(candidates, key=lambda c: (abs(int(c.get("toLane")) - to_lane),
                                                          int(c.get("fromLane"))))
                key = (origin, target, int(selected.get("fromLane")), to_lane)
                if key not in existing:
                    added.append(key)
                    ET.SubElement(connection_rules, "connection",
                                  **{"from": origin, "to": target, "fromLane": str(key[2]),
                                     "toLane": str(to_lane), "changeLeft": "emergency",
                                     "changeRight": "emergency", "contPos": "0"})
                    existing.add(key)
    graph = {}
    tls_nodes = {j.get("id") for j in root.findall("junction") if j.get("type") == "traffic_light"}
    for target in short:
        edge = edges[target]
        if float(edge.find("lane").get("length")) >= vehicle_space_m:
            continue
        start, end = edge.get("from"), edge.get("to")
        if start in tls_nodes and end in tls_nodes:
            graph.setdefault(start, set()).add(end)
            graph.setdefault(end, set()).add(start)
    node_rules, seen, groups = ET.Element("nodes"), set(), []
    for start in sorted(graph):
        if start in seen:
            continue
        component, pending = set(), [start]
        while pending:
            item = pending.pop()
            if item not in seen:
                seen.add(item)
                component.add(item)
                pending.extend(graph[item] - seen)
        if not component.intersection(anchors):
            continue
        groups.append(sorted(component))
        for item in sorted(component):
            ET.SubElement(node_rules, "node", id=item, type="traffic_light",
                          tl=f"kintambo_shared_{len(groups)}", tlLayout="incoming")
    return edge_rules, connection_rules, node_rules, {
        "vehicle_space_m": vehicle_space_m, "lane_change_clearance_m": 2 * vehicle_space_m,
        "controller_coupling_clearance_m": vehicle_space_m, "short_connectors": short,
        "added_lane_connections": added, "shared_controller_junctions": groups,
        "lane_changes_in_intersections": "emergency_only",
        "yielding_position": "before_conflict_zone",
    }


def _correct_connections(config: dict, directory: Path, vehicle_space_m: float) -> dict:
    """Reconstruit géométrie, priorités et feux avec netconvert, jamais à la main."""
    if not math.isfinite(vehicle_space_m) or vehicle_space_m <= 0:
        raise ValueError("L'espace nominal du véhicule doit être positif.")
    path = directory / "network.net.xml"
    patches = _connection_patches(ET.parse(path).getroot(), vehicle_space_m,
                                  set(config["network"]["junction_groups"]))
    for name, root in zip(("continuity.edg.xml", "continuity.con.xml", "control.nod.xml"), patches[:3]):
        write_xml(directory / name, root)
    output = directory / "connected.net.xml"
    command = [require_binary("netconvert"), "--sumo-net-file", str(path),
               "--offset.disable-normalization", "true",
               "--edge-files", str(directory / "continuity.edg.xml"),
               "--connection-files", str(directory / "continuity.con.xml"),
               "--node-files", str(directory / "control.nod.xml"),
               "--tls.rebuild", "true", "--tls.default-type", "static", "--tls.join", "false",
               "--tls.cycle.time", str(config["network"]["cycle_s"]),
               "--tls.layout", config["network"]["signal_layout"],
               "--tls.allred.time", str(config["network"]["all_red_s"]),
               "--tls.minor-left.max-speed", "0", "--no-turnarounds", "true",
               "--output-file", str(output)]
    with (directory / "continuity.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=log)
    if completed.returncode:
        raise RuntimeError("Correction des raccords échouée ; consulter continuity.log.")
    write_xml(path, ET.parse(output).getroot())
    return {**patches[3], "command": command}


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
