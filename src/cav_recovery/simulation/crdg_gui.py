"""Dessine le graphe courant dans SUMO, sans commande de trafic."""

import json
import math
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET
from .road_network import write_xml

COLOURS = {"leader": (40, 110, 220, 255), "connection_obstacle": (135, 60, 180, 255),
           "waits_for": (230, 140, 0, 255), "occupied_by": (230, 140, 0, 255),
           "blocked_by": (200, 50, 70, 255)}
EXPLANATIONS = {"leader": "attend son véhicule de tête", "connection_obstacle": "attend un obstacle de connexion",
                "waits_for": "attend", "occupied_by": "est occupée par",
                "blocked_by": "est bloquée par"}


def lane_shapes(path):
    return {lane.get("id"): [tuple(map(float, point.split(",")[:2])) for point in lane.get("shape").split()]
            for lane in ET.parse(path).findall("edge/lane")}


def configure_view(path):
    """Affiche les textes des POI natifs sans changer les réglages routiers."""
    root = ET.parse(path).getroot()
    # SUMO multiplie cette taille constante par 0,1 avant le dessin.
    ET.SubElement(root.find("scheme"), "pois", poiType_show="true", poiType_size="60",
                  poiType_constantSize="true", poiType_onlySelected="false",
                  poiType_color="20,30,45", poiType_bgColor="255,255,255,230")
    write_xml(path, root)


def focused_graph(current, focus, depth=2, limit=16):
    """Choisit un petit voisinage ; conserve explicitement les éléments non affichés."""
    root = "vehicle:" + focus
    nodes = {node["id"]: node for node in current["nodes"]}
    visible, frontier = {root}, {root}
    omitted = set()
    for _ in range(depth):
        adjacent = set()
        for edge in current["edges"]:
            if edge["source"] in frontier:
                adjacent.add(edge["target"])
            if edge["target"] in frontier:
                adjacent.add(edge["source"])
        new = sorted(adjacent - visible)
        room = max(0, limit - len(visible))
        frontier = set(new[:room])
        omitted.update(new[room:])
        visible.update(frontier)
    return {"nodes": [nodes[item] for item in sorted(visible) if item in nodes],
            "edges": [edge for edge in current["edges"] if edge["source"] in visible and edge["target"] in visible],
            "omitted_nodes": sorted(omitted)}


def arrow_shape(start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    distance = math.hypot(dx, dy)
    if distance == 0:
        return None
    ux, uy = dx / distance, dy / distance
    size = min(2, distance / 3)
    return [end, (end[0] - ux * size - uy * size / 2, end[1] - uy * size + ux * size / 2),
            (end[0] - ux * size + uy * size / 2, end[1] - uy * size - ux * size / 2), end]


class SceneAnnotations:
    """Écrit les repères d'un instant figé avec les formes natives SUMO."""

    def __init__(self, shapes, *, focus, depth=2):
        self.shapes, self.focus, self.depth = shapes, focus, depth
        self.polygons, self.pois = [], []
        self.root = ET.Element("additional")

    def clear(self):
        self.root.clear()
        self.polygons.clear()
        self.pois.clear()

    def text(self, position, text):
        item = f"crdg:text:{len(self.pois)}"
        node = ET.SubElement(self.root, "poi", id=item, x=str(position[0]), y=str(position[1]),
                             color="0,0,0,0", type=text, layer="100")
        self.pois.append(item)
        return node

    def polygon(self, shape, colour, *, fill=False):
        item = f"crdg:shape:{len(self.polygons)}"
        ET.SubElement(self.root, "poly", id=item, shape=" ".join(f"{x},{y}" for x, y in shape),
                      color=",".join(map(str, colour)), fill=str(fill).lower(), layer="100", lineWidth="0.7")
        self.polygons.append(item)

    def show(self, current, readings):
        self.clear()
        focus, time_s = self.focus, current["time_s"]
        if focus not in readings:
            raise ValueError("Le véhicule choisi n'est pas présent à cet instant.")
        view = focused_graph(current, focus, self.depth) if focus else {"nodes": [], "edges": [], "omitted_nodes": []}
        positions, unplaced = {}, []
        for node in view["nodes"]:
            if node["node_type"] == "vehicle":
                item = node["vehicle_id"]
                if item in readings:
                    positions[node["id"]] = tuple(readings[item]["position"])
            else:
                lanes = node.get("candidate_lanes", []) if node["resource_type"] == "receiving_space" else [node["via_lane"]]
                shapes = [self.shapes[lane] for lane in lanes if lane in self.shapes]
                for shape in shapes:
                    self.polygon(shape, (240, 160, 0, 160))
                if shapes:
                    # C'est un repère sur la voie native, pas une zone de conflit calculée.
                    positions[node["id"]] = shapes[0][0]
            if node["id"] not in positions:
                unplaced.append(node["id"])
        if focus in readings:
            positions.setdefault("vehicle:" + focus, tuple(readings[focus]["position"]))
        for index, (item, point) in enumerate(positions.items()):
            colour = (0, 160, 220, 255) if item == "vehicle:" + str(focus) else (230, 140, 0, 255)
            ring = [(point[0] + 2 * math.cos(i * math.tau / 24), point[1] + 2 * math.sin(i * math.tau / 24))
                    for i in range(25)]
            self.polygon(ring, colour)
            label = item.removeprefix("vehicle:") if item == "vehicle:" + focus else f"V{index}" if item.startswith("vehicle:") else f"R{index}"
            note = self.text((point[0], point[1] + 4 + (index % 2) * 5), label)
            ET.SubElement(note, "param", key="crdg_node", value=item)
        for edge in view["edges"]:
            a, b = positions.get(edge["source"]), positions.get(edge["target"])
            if a is None or b is None:
                continue
            head = arrow_shape(a, b)
            if head is None:
                continue
            colour = COLOURS.get(edge["edge_type"], (100, 100, 100, 255))
            self.polygon([a, b], colour)
            self.polygon(head, colour, fill=True)
        xs, ys = zip(*positions.values())
        centre_x, centre_y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        width, height = max(180, max(xs) - min(xs) + 50), max(140, max(ys) - min(ys) + 50)
        left, right, bottom, top = centre_x - width / 2, centre_x + width / 2, centre_y - height / 2, centre_y + height / 2
        message = f"État figé {time_s:g} s — C-RDG et scène — {focus}"
        lines = [message, "Bleu : voiture choisie / suivi ; orange : ressource ; rouge : conflit. Flèche A vers B : A dépend de B."]
        if focus and not view["edges"]:
            state = current.get("waiting_states", {}).get(focus)
            reason = {"signal": "attente liée au signal", "unknown": "cause inconnue"}.get(state, "aucune dépendance qualifiée")
            lines.append(f"{focus} : {reason} ; aucune cause supplémentaire n'est inventée.")
        if view["omitted_nodes"]:
            lines.append(f"Vue limitée : {len(view['omitted_nodes'])} nœuds voisins non affichés.")
        for index, line in enumerate(lines):
            self.text(((left + right) / 2, top - (top - bottom) * (.04 + index * .045)), line)
        for index, edge in enumerate(view["edges"][:4]):
            a, b = edge["source"].removeprefix("vehicle:"), edge["target"].removeprefix("vehicle:")
            if a.startswith("resource:"):
                a = "La ressource"
            if b.startswith("resource:"):
                b = "la réception aval" if "receiving" in b else "le conflit natif"
            detail = edge.get("evidence", "occupation observée" if edge["edge_type"] == "occupied_by" else "preuve sur la ressource")
            resource = next((node for node in view["nodes"] if node["id"] == edge["target"] and node["node_type"] == "resource"), None)
            if resource and resource["resource_type"] == "receiving_space":
                free = resource["max_free_space_m"]
                detail = f"réception {'inconnue' if free is None else f'{free:.2f} m'} / besoin {resource['required_space_m']:g} m"
            text = f"{a} {EXPLANATIONS.get(edge['edge_type'], edge['edge_type'])} {b} [{detail}]"
            point = ((left + right) / 2, bottom + (top - bottom) * (.04 + index * .04))
            note = self.text(point, text)
            ET.SubElement(note, "param", key="crdg_evidence", value=json.dumps(edge, ensure_ascii=False, sort_keys=True))
        return {"time_s": time_s, "focus": focus, **view, "positions": positions, "unplaced_nodes": unplaced,
                "vehicles": {item: readings[item] for item in sorted(readings) if "vehicle:" + item in positions},
                "explanations": [{"source": edge["source"], "target": edge["target"],
                                  "text": EXPLANATIONS.get(edge["edge_type"], edge["edge_type"]),
                                  "evidence": edge} for edge in view["edges"]],
                "legend": lines, "view_boundary": [left, bottom, right, top]}


def save_scene(connection, current, readings, directory, focus, depth, provenance):
    """Sauvegarde l'état natif au même instant ; ne fait aucun pas supplémentaire."""
    if connection.simulation.getTime() != current["time_s"]:
        raise ValueError("Le graphe et la scène doivent avoir le même instant.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    connection.simulation.saveState(str(directory / "state.xml.gz"))
    if connection.simulation.getTime() != current["time_s"]:
        raise ValueError("Le temps a changé pendant la sauvegarde.")
    data = {"snapshot": current, "readings": readings, "provenance": provenance,
            "state_sha256": hashlib.sha256((directory / "state.xml.gz").read_bytes()).hexdigest(),
            "focus": focus, "depth": depth}
    (directory / "scene.json").write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False), encoding="utf-8")
    if focus in readings:
        return render_scene(directory, focus=focus, depth=depth)
    return {"time_s": current["time_s"], "focus": focus, "focus_present": False if focus else None}


def render_scene(directory, *, focus=None, depth=None):
    """Prépare une vue locale ; seul le dessin peut changer, pas l'état enregistré."""
    directory = Path(directory).resolve()
    data = json.loads((directory / "scene.json").read_text(encoding="utf-8"))
    network = directory.parents[1] / "network.net.xml"
    if (hashlib.sha256(network.read_bytes()).hexdigest() != data["provenance"]["network_sha256"]
            or hashlib.sha256((directory / "state.xml.gz").read_bytes()).hexdigest() != data["state_sha256"]):
        raise ValueError("Le réseau ou l'état de la scène a changé.")
    chosen = focus or data["focus"]
    if chosen is None:
        raise ValueError("Choisir un véhicule avec --focus ; --list-vehicles donne les IDs présents.")
    depth = data["depth"] if depth is None else depth
    if type(depth) is not int or depth not in (1, 2, 3):
        raise ValueError("Le voisinage doit valoir 1, 2 ou 3.")
    notes = SceneAnnotations(lane_shapes(network), focus=chosen, depth=depth)
    shown = notes.show(data["snapshot"], data["readings"])
    shown["renderer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_xml(directory / "annotations.add.xml", notes.root)
    source_view = ET.parse(directory.parents[1] / "view.xml").getroot()
    left, bottom, right, top = shown["view_boundary"]
    import sumolib
    net = sumolib.net.readNet(str(network))
    bounds = net.getBoundary()
    zoom = 100 * min((bounds[2] - bounds[0]) / (right - left), (bounds[3] - bounds[1]) / (top - bottom))
    viewport = source_view.find("viewport")
    viewport.set("x", str((left + right) / 2))
    viewport.set("y", str((bottom + top) / 2))
    viewport.set("zoom", str(zoom))
    write_xml(directory / "view.xml", source_view)
    configure_view(directory / "view.xml")
    config = ET.Element("configuration")
    for group, values in {"input": {"net-file": "../../network.net.xml", "route-files": "../../traffic.rou.xml",
                                   "load-state": "state.xml.gz", "additional-files": "annotations.add.xml"},
                          "time": {"begin": str(data["snapshot"]["time_s"]), "end": str(data["snapshot"]["time_s"]), "step-length": "0.5"},
                          "processing": {"time-to-teleport": "-1", "collision.check-junctions": "true"},
                          "gui_only": {"gui-settings-file": "view.xml"}}.items():
        parent = ET.SubElement(config, group)
        for key, value in values.items():
            ET.SubElement(parent, key, value=value)
    write_xml(directory / "scene.sumocfg", config)
    (directory / "annotations.json").write_text(json.dumps(shown, ensure_ascii=False, sort_keys=True, allow_nan=False), encoding="utf-8")
    return shown
