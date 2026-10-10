"""Relie INFO C-RDG aux observations ; les écritures TraCI sont seulement graphiques."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

from .sumo_view import NativeView, visible_labels

SELECTED = (25, 145, 220, 255)
CONSTRAINED = (235, 155, 45, 255)
PARTICIPANT = (145, 110, 195, 255)
TEXT_KEY = "crdg_visual_id"
SECTOR_JUNCTIONS = ("3675784999", "magasin_nguma", "magasin_oua")
RELATIONS = {"leader": "Suivi limité", "waits_for": "Attend", "occupied_by": "Voie occupée",
             "blocked_by": "Passage contraint", "connection_obstacle": "Passage contraint"}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def visual_ids(missions):
    ordered = sorted(missions, key=lambda mission: (mission.scheduled_s, mission.vehicle_id))
    return {mission.vehicle_id: f"V{index:03d}" for index, mission in enumerate(ordered, 1)}


def configure_live_view(path):
    from .road_network import write_xml
    root = ET.parse(path).getroot()
    vehicles = root.find("scheme/vehicles")
    vehicles.set("vehicleText_show", "true")
    vehicles.set("vehicleExaggeration", "1")
    vehicles.set("vehicleMinSize", "4")
    vehicles.set("vehicleText_size", "44")
    vehicles.set("vehicleText_constantSize", "true")
    vehicles.set("vehicleText_color", "20,35,50")
    vehicles.set("vehicleText_bgColor", "250,250,250,230")
    vehicles.set("vehicleTextParam", TEXT_KEY)
    write_xml(path, root)


def edge_key(edge):
    return edge["dependency_id"] if "dependency_id" in edge else "|".join((edge["edge_type"], edge["source"], edge["target"]))


def dependency_groups(graph, current):
    """Les composantes servent à parcourir les dépendances, pas à diagnostiquer."""
    import networkx as nx
    nodes = {row["id"]: row for row in current["nodes"]}
    groups = []
    for component in nx.weakly_connected_components(graph):
        members = sorted(component)
        edges = [edge for edge in current["edges"] if edge["source"] in component]
        kinds = {edge["edge_type"] for edge in edges}
        resources = {nodes[item].get("resource_type") for item in members}
        categories = []
        if "leader" in kinds:
            categories.append("following")
        if "receiving_space" in resources:
            categories.append("receiving")
        if "junction_conflict" in resources or "connection_obstacle" in kinds:
            categories.append("junction")
        if any(set(c["nodes"]).issubset(component) for c in current.get("cycle_candidates", [])):
            categories.append("cycle")
        vehicles = sum(nodes[item]["node_type"] == "vehicle" for item in members)
        if categories == ["following"]:
            chain = len(edges) == len(members) - 1 and all(graph.in_degree(n) <= 1 and graph.out_degree(n) <= 1 for n in members)
            label = f"File de {vehicles} voitures" if chain else "Suivi entre véhicules"
        else:
            label = {"receiving": "Réception occupée", "junction": "Contrainte de passage",
                     "cycle": "Cycle candidat"}.get(categories[0], "Contraintes observées") if len(categories) == 1 else "Contraintes multiples"
        if "cycle" in categories:
            label = "Cycle candidat · " + label if label != "Cycle candidat" else label
        groups.append({"nodes": members,
                       "vehicles": vehicles, "label": label, "categories": categories,
                       "resources": sum(nodes[item]["node_type"] == "resource" for item in members),
                       "edges": len(edges),
                       "streets": sorted({nodes[item]["street_name"] for item in members if nodes[item].get("street_name")})})
    return sorted(groups, key=lambda row: (-row["vehicles"], row["nodes"]))


def view_boundary(positions, *, minimum=(100, 80), margin=15):
    """Ajoute une marge de lecture à la carte, jamais à l'espace physique."""
    points = [point for point in positions if len(point) == 2 and all(math.isfinite(v) for v in point)]
    if not points:
        return None
    left, right = min(p[0] for p in points), max(p[0] for p in points)
    bottom, top = min(p[1] for p in points), max(p[1] for p in points)
    width, height = max(minimum[0], right - left + 2 * margin), max(minimum[1], top - bottom + 2 * margin)
    x, y = (left + right) / 2, (bottom + top) / 2
    return (x - width / 2, y - height / 2, x + width / 2, y + height / 2)


def local_sector(path, fallback):
    points = []
    for junction in ET.parse(path).getroot().findall("junction"):
        if junction.get("id") in SECTOR_JUNCTIONS:
            points.extend(tuple(map(float, point.split(","))) for point in junction.get("shape", "").split())
    return view_boundary(points, minimum=(180, 160), margin=55) or tuple(fallback)


def whole_network(path, fallback):
    """Cadre toutes les voies du réseau actif, pas seulement le noyau Magasin."""
    points = [tuple(map(float, point.split(","))) for lane in ET.parse(path).getroot().iter("lane")
              for point in lane.get("shape", "").split()]
    return view_boundary(points, margin=30) or tuple(fallback)


def _enter_pressed():
    if os.name == "nt":
        import msvcrt
        return msvcrt.kbhit() and msvcrt.getwch() in ("\r", "\n")
    import select
    if select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
        return True
    return False


class CrdgPanelLink:
    """Garde les fichiers d'échange et les couleurs d'origine de cette exécution."""

    def __init__(self, output, missions, *, focus=None, sector_boundary=None, network_boundary=None):
        self.output_directory = Path(output) / "info_crdg"
        self.output_directory.mkdir()
        # Les clics ne doivent pas attendre des écritures synchronisées par OneDrive.
        self._temporary = tempfile.TemporaryDirectory(prefix="cgr-info-")
        self.directory = Path(self._temporary.name)
        self.aliases = visual_ids(missions)
        self.focus, self.paused = focus, False
        self.highlighted = set()
        self.sector_boundary, self.network_boundary = sector_boundary, network_boundary
        self.camera_request, self.camera_sequence, self.camera_status = None, None, None
        self.tracked_vehicle = None
        self.close_requested = False
        self.sumo_process = None
        self.graph = None
        self.frozen_at = None
        self.command_sent_ns = None
        self.original, self.colours, self.resource_ids, self.group_ids = {}, {}, {}, {}
        self.last_frame = self.current = None
        self.labels = {}
        self.original_labels = {}
        self.label_mode = "selected"
        self.native_view, self.native_attempted, self.native_status = None, False, "not_initialized"
        self.label_information = {}
        self.process = self.log = None
        self.stats = {"updates": 0, "colour_changes": 0, "colour_reads": 0, "label_changes": 0,
                      "cpu_s": 0.0, "camera_changes": 0, "closed_by_user": False, "errors": []}
        atomic_json(self.directory / "ids.json", self.aliases)
        atomic_json(self.output_directory / "ids.json", self.aliases)
        atomic_json(self.directory / "control.json", {"focus": focus, "highlighted": [], "paused": False,
                                                       "close_requested": False, "labels": "selected"})
        atomic_json(self.directory / "clock.json", {"time_s": 0, "status": "starting", "inactive_edges": []})

    def start(self):
        self.log = (self.output_directory / "panel.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen([sys.executable, "-m", "cav_recovery.simulation.crdg_panel", str(self.directory)],
                                        stdout=self.log, stderr=self.log)
        self.wait_ready()

    def wait_ready(self):
        started = time.monotonic()
        while not (self.directory / "ready.json").exists():
            if self.process.poll() is not None:
                raise RuntimeError("INFO C-RDG n'a pas pu s'ouvrir ; voir panel.log.")
            if time.monotonic() - started > 15:
                raise RuntimeError("INFO C-RDG ne répond pas à l'ouverture ; voir panel.log.")
            time.sleep(0.05)
        self.stats["startup_s"] = time.monotonic() - started

    def control(self, *, check_sumo=True):
        command = json.loads((self.directory / "control.json").read_text(encoding="utf-8"))
        self.close_requested = bool(command.get("close_requested"))
        if self.close_requested:
            self.stats["closed_by_user"] = True
            return False
        if check_sumo and command.get("ui_error"):
            raise RuntimeError(f"Erreur INFO C-RDG : {command['ui_error']} ; voir panel.log.")
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError("INFO C-RDG s'est fermé sans demande explicite ; voir panel.log.")
        if check_sumo and self.sumo_process is not None and self.sumo_process.poll() is not None:
            raise RuntimeError("SUMO s'est arrêté ou sa fenêtre a été fermée ; voir sumo.log.")
        focus = command.get("focus")
        if focus is not None and focus not in self.aliases:
            raise ValueError("Sélection INFO C-RDG inconnue.")
        self.focus, self.paused = focus, bool(command.get("paused"))
        self.highlighted = set(command.get("highlighted", []))
        if not self.highlighted.issubset(self.aliases):
            raise ValueError("Voitures du groupe INFO C-RDG inconnues.")
        self.camera_request = command.get("camera")
        self.command_sent_ns = command.get("sent_ns")
        self.label_mode = command.get("labels", "selected")
        if self.label_mode not in ("selected", "all", "none"):
            raise ValueError("Mode d'étiquettes inconnu.")
        return True

    def prepare_native_view(self):
        if not self.native_attempted and self.sumo_process is not None and type(self.sumo_process.pid) is int:
            self.native_attempted = True
            try:
                self.native_view = NativeView(self.sumo_process.pid)
                self.native_status = "x11_ready"
            except RuntimeError as error:
                self.native_status = str(error)

    def redraw(self):
        if self.native_view:
            try:
                self.native_view.redraw()
                return time.monotonic_ns()
            except RuntimeError as error:
                self.native_status = str(error)
                self.native_view.close()
                self.native_view = None
        return None

    def colour(self, connection, current, readings):
        selected = self.highlighted | ({self.focus} if self.focus else set())
        boundary = connection.gui.getBoundary("View #0") if self.label_mode != "none" and self.native_view else None
        size = self.native_view.size if boundary is not None else None
        labelled = visible_labels(self.aliases, readings, selected, self.label_mode, boundary, size)
        wanted = {row["vehicle_id"]: PARTICIPANT for row in current["nodes"] if row["node_type"] == "vehicle"}
        for edge in current["edges"]:
            if edge["source"].startswith("vehicle:"):
                wanted[edge["source"].removeprefix("vehicle:")] = CONSTRAINED
        if self.focus in readings:
            wanted[self.focus] = SELECTED
        for item in self.highlighted.intersection(readings):
            wanted[item] = SELECTED
        for item in sorted(set(self.colours) | set(wanted) | set(readings)):
            if item not in readings:
                self.colours.pop(item, None)
                self.labels.pop(item, None)
                self.original.pop(item, None)
                self.original_labels.pop(item, None)
                continue
            if item not in wanted and item in self.original:
                colour, label = self.original.pop(item)
                connection.vehicle.setColor(item, colour)
                self.colours.pop(item)
                self.stats["colour_changes"] += 1
            elif item in wanted:
                if item not in self.original:
                    original_label = self.original_labels.get(item)
                    if original_label is None:
                        original_label = connection.vehicle.getParameter(item, TEXT_KEY)
                    self.original[item] = (connection.vehicle.getColor(item), original_label)
                    self.stats["colour_reads"] += 1
                if self.colours.get(item) != wanted[item]:
                    connection.vehicle.setColor(item, wanted[item])
                    self.stats["colour_changes"] += 1
                self.colours[item] = wanted[item]
            if item not in self.original_labels:
                self.original_labels[item] = self.original[item][1] if item in self.original else connection.vehicle.getParameter(item, TEXT_KEY)
            label = self.aliases[item] if item in labelled else ""
            if self.labels.get(item, "") != label:
                connection.vehicle.setParameter(item, TEXT_KEY, label)
                self.stats["label_changes"] += 1
            self.labels[item] = label
        self.label_information = {"mode": self.label_mode, "enabled": len(labelled), "present": len(readings),
                                  "suppressed": len(readings) - len(labelled)}

    def apply_controls(self, connection, readings, time_s):
        self.prepare_native_view()
        before = (self.stats["colour_changes"], self.stats["label_changes"])
        self.colour(connection, self.current or {"nodes": [], "edges": []}, readings)
        if self.tracked_vehicle and self.tracked_vehicle not in readings:
            connection.gui.trackVehicle("View #0", "")
            self.tracked_vehicle = None
            self.camera_status["message"] = "Suivi terminé : voiture absente maintenant."
        request = self.camera_request
        if not request or request["sequence"] == self.camera_sequence:
            if before != (self.stats["colour_changes"], self.stats["label_changes"]):
                self.redraw()
            return
        received_ns = time.monotonic_ns()
        self.camera_sequence = request["sequence"]
        kind = request["kind"]
        requested = sorted(set(request.get("vehicles", [])))
        if not set(requested).issubset(self.aliases):
            raise ValueError("Cadrage demandé sur une voiture inconnue.")
        present = [item for item in requested if item in readings]
        boundary = None
        message = ""
        connection.gui.trackVehicle("View #0", "")
        self.tracked_vehicle = None
        if kind == "sector":
            boundary = self.sector_boundary
        elif kind == "network":
            boundary = self.network_boundary
        elif kind in ("fit", "follow"):
            boundary = view_boundary([readings[item]["position"] for item in present])
            if not boundary:
                message = "Voitures absentes maintenant : aucun ancien cadrage utilisé."
        elif kind != "manual":
            raise ValueError("Commande de caméra INFO C-RDG inconnue.")
        if boundary:
            connection.gui.setBoundary("View #0", *boundary)
        if kind == "follow" and len(present) == 1:
            connection.gui.trackVehicle("View #0", present[0])
            self.tracked_vehicle = present[0]
        returned_ns = time.monotonic_ns()
        self.colour(connection, self.current or {"nodes": [], "edges": []}, readings)
        redraw_ns = self.redraw()
        self.camera_status = {"kind": kind, "sequence": self.camera_sequence, "time_s": time_s, "vehicles": present,
                              "missing": sorted(set(requested) - set(present)), "boundary_m": boundary, "message": message,
                              "sent_ns": self.command_sent_ns, "received_ns": received_ns,
                              "traci_returned_ns": returned_ns, "redraw_requested_ns": redraw_ns,
                              "native_redraw": self.native_status, "sumo_rendered_ns": None}
        with (self.output_directory / "camera_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(self.camera_status, ensure_ascii=False, sort_keys=True) + "\n")
        self.stats["camera_changes"] += 1

    def before_step(self, connection, readings, time_s):
        """La pause permet de consulter et cadrer sans nouveau pas physique."""
        while self.control():
            self.apply_controls(connection, readings, time_s)
            if not self.paused:
                self.frozen_at = None
                return True
            self.freeze("paused", time_s, readings)
            self.status("paused", time_s)
            time.sleep(0.1)
        self.restore(connection, readings)
        return False

    def hold_view(self, connection, readings, status, time_s, *, reason=None, counts=None):
        """À la fin, on garde le graphe ; la carte reste accessible si SUMO répond."""
        self.freeze(status, time_s, readings)
        print("Exécution terminée. Fermer explicitement la démonstration dans INFO C-RDG.", flush=True)
        while self.control(check_sumo=status != "failed"):
            if status != "failed" or (self.sumo_process is not None and self.sumo_process.poll() is None):
                try:
                    self.apply_controls(connection, readings, time_s)
                except Exception:
                    if status != "failed":
                        raise
                    self.camera_status = {"kind": "manual", "vehicles": [], "time_s": time_s,
                                          "message": "SUMO ne répond plus ; le graphe reste consultable."}
            self.status(status, time_s, reason=reason, counts=counts)
            if sys.stdin.isatty() and _enter_pressed():
                self.stats["closed_by_user"] = True
                return
            time.sleep(0.1)

    def observe(self, connection, graph, current, readings):
        started = time.perf_counter()
        if not self.control():
            self.restore(connection, readings)
            return
        self.current = current
        self.graph = graph
        self.apply_controls(connection, readings, current["time_s"])
        for row in current["nodes"]:
            if row["node_type"] == "resource":
                self.resource_ids.setdefault(row["id"], f"R{len(self.resource_ids) + 1:03d}")
        if current["time_s"] % 5 == 0:
            groups = dependency_groups(graph, current)
            for group in groups:
                key = tuple(group["nodes"])
                self.group_ids.setdefault(key, f"G{len(self.group_ids) + 1:03d}")
                group["id"] = self.group_ids[key]
            self.last_frame = {"snapshot": current, "groups": groups, "resources": dict(self.resource_ids),
                               "aliases": self.aliases}
            atomic_json(self.directory / "frame.json", self.last_frame)
            with (self.output_directory / "frames.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(self.last_frame, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            self.stats["updates"] += 1
        active = {edge_key(edge) for edge in current["edges"]}
        previous = {edge_key(edge) for edge in self.last_frame["snapshot"]["edges"]} if self.last_frame else set()
        atomic_json(self.directory / "clock.json", {"time_s": current["time_s"], "status": "running",
                    "inactive_edges": sorted(previous - active), "focus": self.focus, "present": sorted(readings),
                    "constrained": sorted({edge["source"].removeprefix("vehicle:") for edge in current["edges"]
                                           if edge["source"].startswith("vehicle:")}),
                    "participants": sorted(row["vehicle_id"] for row in current["nodes"] if row["node_type"] == "vehicle"),
                    "highlighted": sorted(self.highlighted), "camera": self.camera_status,
                    "labels": self.label_information,
                    "native_redraw": self.native_status,
                    "render_time_s": None})
        self.stats["cpu_s"] += time.perf_counter() - started

    def freeze(self, status, time_s, readings):
        if self.frozen_at == (status, time_s):
            return
        self.frozen_at = (status, time_s)
        if self.current is not None:
            if self.graph is None:
                import networkx as nx
                self.graph = nx.DiGraph()
                self.graph.add_nodes_from(n["id"] for n in self.current["nodes"])
                self.graph.add_edges_from((e["source"], e["target"]) for e in self.current["edges"])
            groups = dependency_groups(self.graph, self.current)
            for group in groups:
                key = tuple(group["nodes"])
                self.group_ids.setdefault(key, f"G{len(self.group_ids) + 1:03d}")
                group["id"] = self.group_ids[key]
            self.last_frame = {"snapshot": self.current, "groups": groups, "resources": dict(self.resource_ids),
                               "aliases": self.aliases}
            atomic_json(self.directory / "frame.json", self.last_frame)
        frozen = {"time_s": time_s, "status": status, "readings": readings, "snapshot": self.current,
                  "aliases": self.aliases}
        atomic_json(self.directory / "frozen.json", frozen)
        atomic_json(self.output_directory / f"inspection_{time_s:g}.json", frozen)

    def status(self, status, time_s, *, reason=None, counts=None):
        path = self.directory / "clock.json"
        clock = json.loads(path.read_text(encoding="utf-8"))
        clock.update(status=status, time_s=time_s, focus=self.focus, highlighted=sorted(self.highlighted), camera=self.camera_status)
        clock.update(labels=self.label_information, native_redraw=self.native_status)
        if reason is not None:
            clock["reason"] = reason
        if counts is not None:
            clock["counts"] = counts
        atomic_json(path, clock)

    def restore(self, connection, readings):
        if self.tracked_vehicle:
            connection.gui.trackVehicle("View #0", "")
            self.tracked_vehicle = None
        for item, (colour, label) in list(self.original.items()):
            if item in readings:
                connection.vehicle.setColor(item, colour)
                connection.vehicle.setParameter(item, TEXT_KEY, self.original_labels[item])
        for item in set(self.labels) - set(self.original):
            if item in readings:
                connection.vehicle.setParameter(item, TEXT_KEY, self.original_labels[item])
        self.original.clear()
        self.colours.clear()
        self.labels.clear()
        self.original_labels.clear()

    def finish(self, connection, readings, status, time_s):
        errors = []
        def attempt(action):
            try:
                return action()
            except Exception as error:
                errors.append(error)
                self.stats["errors"].append(str(error))

        attempt(lambda: self.restore(connection, readings))
        attempt(lambda: self.status(status, time_s))
        # Même si une étape échoue, on ferme les autres ressources.
        if self.process is not None and attempt(self.process.poll) is None:
            attempt(lambda: self.status("closing", time_s))
            for stop in (None, self.process.terminate, self.process.kill):
                if stop is not None:
                    attempt(stop)
                    self.stats["errors"].append("Arrêt forcé du panneau demandé.")
                try:
                    self.process.wait(timeout=5)
                    break
                except Exception as error:
                    if not isinstance(error, subprocess.TimeoutExpired):
                        errors.append(error)
                        self.stats["errors"].append(str(error))
            if attempt(self.process.poll) is None:
                error = RuntimeError("Arrêt du panneau non confirmé.")
                errors.append(error)
                self.stats["errors"].append(str(error))
        if self.log:
            attempt(self.log.close)
        import shutil
        for name in ("ui_events.jsonl", "clock.json", "control.json"):
            source = self.directory / name
            attempt(lambda source=source, name=name: shutil.copyfile(source, self.output_directory / name)
                    if source.exists() else None)
        attempt(self._temporary.cleanup)
        if self.native_view:
            attempt(self.native_view.close)
        attempt(lambda: atomic_json(self.output_directory / "summary.json", self.stats))
        if errors:
            raise errors[0]
