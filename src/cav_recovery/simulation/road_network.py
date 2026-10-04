"""Construction et contrôle d'un réseau routier avec les outils SUMO."""

from __future__ import annotations

from decimal import Decimal
import importlib.metadata
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

from .traffic_demand import TrafficInputError, ENTRY_GATES, EXIT_GATES, OSM_IDENTITY, file_hash, verify_identity


VERSION = "1.27.1"
CENTER_NODE = "250691665"
GATE_EDGES = {"W23183369_IN": "23183369#1", "W284241336_IN": "284241336#1",
              "W23183369_OUT": "23183369#5", "W284241336_OUT": "284241336#3"}
ROUTES = {
    "W23183369_IN__W23183369_OUT": ["23183369#1", "23183369#2", "23183369#3", "23183369#4", "23183369#5"],
    "W23183369_IN__W284241336_OUT": ["23183369#1", "23183369#2", "23183369#3", "284241336#3"],
    "W284241336_IN__W23183369_OUT": ["284241336#1", "284241336#2", "23183369#4", "23183369#5"],
    "W284241336_IN__W284241336_OUT": ["284241336#1", "284241336#2", "284241336#3"],
}
CENTER_PHASES = [(39, "GGrrr"), (6, "yyrrr"), (39, "rrGGG"), (6, "rryyy")]


def check_environment(*tools: str) -> dict:
    """Refuse une version différente de celle utilisée pour vérifier les routes."""
    versions = {}
    for tool in tools:
        binary = shutil.which(tool)
        if binary is None:
            raise TrafficInputError(f"Exécutable absent : {tool}. Activer l'environnement du projet.")
        output = subprocess.run([binary, "--version"], capture_output=True, text=True, check=True).stdout
        match = re.search(r"\b(\d+\.\d+\.\d+)\b", output)
        if not match or match.group(1) != VERSION:
            raise TrafficInputError(f"Version incompatible pour {tool} ; {VERSION} est requise.")
        versions[tool] = {"version": VERSION, "executable": Path(binary).name}
    for package in ("traci", "sumolib", "eclipse-sumo"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise TrafficInputError(f"Paquet absent : {package}") from error
        if version != VERSION:
            raise TrafficInputError(f"Version incompatible pour {package} : {version}")
        versions[package] = version
    return versions


def write_xml(path: Path, root: ET.Element) -> None:
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def convert_network(osm_path: Path, directory: Path) -> dict:
    """Convertit la source sans joindre les feux ni ajuster leur cycle au trafic."""
    identity = verify_identity(osm_path, OSM_IDENTITY)
    command = [shutil.which("netconvert"), "--osm-files", str(osm_path.resolve()),
               "--output-file", str(directory / "network.net.xml"),
               "--tls.default-type", "static", "--tls.cycle.time", "90", "--tls.join", "false",
               "--junctions.join", "false", "--output.original-names", "true",
               "--osm.annotate-defaults", "true", "--log", str(directory / "conversion.log")]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if completed.returncode:
        raise RuntimeError(f"Conversion SUMO en échec : {completed.stderr}")
    # Les commentaires de netconvert contiennent une date et des chemins de
    # sortie variables. Leur retrait ne change aucun élément du réseau.
    write_xml(directory / "network.net.xml", ET.parse(directory / "network.net.xml").getroot())
    return {"source": identity, "command": [Path(command[0]).name, *command[1:]],
            "warnings": completed.stderr.splitlines(), "volatile_xml_comments_removed": True}


def inspect_network(network_path: Path, osm_path: Path) -> dict:
    """Revérifie les routes, les permissions et les programmes réellement générés."""
    import sumolib

    root = ET.parse(network_path).getroot()
    net = sumolib.net.readNet(str(network_path), withInternal=True, withPrograms=True)
    junctions = {j.get("id"): j for j in root.findall("junction")}
    edges = {e.get("id"): e for e in root.findall("edge")}
    center = junctions.get(CENTER_NODE)
    if center is None or center.get("type") != "traffic_light":
        raise TrafficInputError("Le carrefour attendu n'est pas une jonction contrôlée.")
    lights = []
    for tls in root.findall("tlLogic"):
        connections = [dict(c.attrib) for c in root.findall("connection") if c.get("tl") == tls.get("id")]
        nodes = sorted({edges[c["from"]].get("to") for c in connections})
        phases = [(Decimal(p.get("duration")), p.get("state")) for p in tls.findall("phase")]
        if tls.get("type") != "static" or sum(p[0] for p in phases) != 90:
            raise TrafficInputError(f"Programme statique de 90 s non établi : {tls.get('id')}")
        if len(nodes) != 1:
            raise TrafficInputError("Plusieurs jonctions partagent un contrôleur non prévu.")
        lights.append({**tls.attrib, "junctions": nodes, "cycle_s": 90,
                       "phases": [{"duration_s": float(d), "state": s} for d, s in phases],
                       "connections": connections, "provenance": "hypothèse de simulation issue de netconvert"})
    c3 = [light for light in lights if light["junctions"] == [CENTER_NODE]]
    if (len(c3) != 1 or c3[0]["id"] != CENTER_NODE or c3[0]["programID"] != "0"
            or float(c3[0]["offset"]) != 0
            or [(p["duration_s"], p["state"]) for p in c3[0]["phases"]] != CENTER_PHASES):
        raise TrafficInputError("Le contrôleur du carrefour diffère du programme vérifié.")
    for route_id, expected in ROUTES.items():
        if any(edge not in edges for edge in expected):
            raise TrafficInputError(f"Edge de mission absent : {route_id}")
        route, _ = net.getShortestPath(net.getEdge(expected[0]), net.getEdge(expected[-1]),
                                      vClass="passenger", withInternal=True)
        if route is None or [e.getID() for e in route if e.getFunction() == ""] != expected:
            raise TrafficInputError(f"Route légale différente ou absente : {route_id}")
        if not all(e.allows("passenger") for e in route):
            raise TrafficInputError("Une voie de la route refuse les véhicules passenger.")
        crossings = 0
        for a, b in zip(expected, expected[1:]):
            first, second = net.getEdge(a), net.getEdge(b)
            if second not in first.getAllowedOutgoing("passenger") or first.getToNode() != second.getFromNode():
                raise TrafficInputError("Connexion ou sens de mission incompatible.")
            crossings += first.getToNode().getID() == CENTER_NODE
        if crossings != 1:
            raise TrafficInputError("La route ne traverse pas exactement une fois le carrefour.")
        if route_id.startswith("W23183369_IN") and ":2725672310_0" not in [e.getID() for e in route]:
            raise TrafficInputError("Le raccord de la porte d'entrée n'est plus sur la route.")
    source = ET.parse(osm_path).getroot()
    ways = {w.get("id"): {t.get("k"): t.get("v") for t in w.findall("tag")} for w in source.findall("way")}
    attributes = []
    for edge_id in sorted({e for route in ROUTES.values() for e in route}):
        edge = edges[edge_id]
        lanes = edge.findall("lane")
        origins = {p.get("value") for lane in lanes for p in lane.findall("param") if p.get("key") == "origId"}
        if len(origins) != 1 or next(iter(origins)) not in ways:
            raise TrafficInputError(f"Origine OSM inconnue pour {edge_id}")
        way_id = next(iter(origins))
        tags = ways[way_id]
        defaults = next((p.get("value", "") for p in edge.findall("param") if p.get("key") == "osmDefaults"), "")
        if tags.get("oneway") != "yes":
            raise TrafficInputError("Un sens unique attendu n'est plus attesté dans la source.")
        if tags.get("lanes") and int(tags["lanes"]) != len(lanes):
            raise TrafficInputError("Le nombre de voies explicite n'est pas conservé.")
        attributes.append({"id": edge_id, "osm_way": way_id, "from": edge.get("from"), "to": edge.get("to"),
                           "lane_count": len(lanes), "lanes": [dict(l.attrib) for l in lanes],
                           "source_tags": tags, "osm_defaults": defaults,
                           "lane_count_origin": "OSM explicite" if "lanes" in tags else "typemap/règle SUMO",
                           "speed_origin": "OSM explicite" if "maxspeed" in tags else "typemap/règle SUMO",
                           "length_origin": "projection et découpage SUMO", "direction_origin": "ordre OSM et oneway=yes"})
    return {"center": dict(center.attrib), "projection": dict(root.find("location").attrib),
            "traffic_lights": lights, "edges": attributes, "routes": ROUTES,
            "gate_mapping": {**GATE_EDGES, "entry_connector": ":2725672310_0"}}


def build_scenery(osm_path: Path, directory: Path) -> dict:
    """Importe seulement le décor présent dans la source, sans changer le réseau."""
    import sumo

    source = ET.parse(osm_path).getroot()
    keys = {"natural", "landuse", "leisure", "building", "waterway", "amenity"}
    objects = [{"kind": e.tag, "id": e.get("id"), "tags": {t.get("k"): t.get("v") for t in e.findall("tag") if t.get("k") in keys}}
               for e in source if any(t.get("k") in keys for t in e.findall("tag"))]
    result = {"source_objects": objects, "background": "fond cosmétique sans signification géographique"}
    if not objects:
        return result
    check_environment("polyconvert")
    typemap = Path(sumo.__file__).parent / "data/typemap/osmPolyconvert.typ.xml"
    command = [shutil.which("polyconvert"), "--osm-files", str(osm_path.resolve()),
               "--net-file", str(directory / "network.net.xml"), "--type-file", str(typemap),
               "--output-file", str(directory / "scenery.add.xml")]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if completed.returncode:
        raise RuntimeError(f"Import du décor en échec : {completed.stderr}")
    root = ET.parse(directory / "scenery.add.xml").getroot()
    write_xml(directory / "scenery.add.xml", root)
    return {**result, "polygons": len(root.findall("poly")), "pois": len(root.findall("poi")),
            "typemap_sha256": file_hash(typemap), "command": [Path(command[0]).name, *command[1:]],
            "warnings": completed.stderr.splitlines()}
