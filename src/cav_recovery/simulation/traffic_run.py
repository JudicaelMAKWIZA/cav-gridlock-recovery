"""Prépare et lance le trafic sans intervention sur les véhicules."""

import csv
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

from .road_network import build_network, read_config, require_binary, write_xml
from .traffic_demand import poisson_missions
from .sumo_process import close_sumo
from .vehicle_tracking import TrafficLedger, verify_trips, STEP_S
from .kintambo_scenarios import read_case, case_missions, prepare_case_network
from .sumo_snapshot import save_scene
from .crdg_live import CrdgPanelLink, configure_live_view, local_sector, whole_network
from .. import crdg as dependency_graph
from ..blockage_observation import BlockageObservation

import xml.etree.ElementTree as ET


VEHICLE_TYPE = {
    "id": "passenger_CAV", "vClass": "passenger", "carFollowModel": "Krauss",
    "length": "5.0", "minGap": "2.5", "accel": "2.6", "decel": "4.5",
    "tau": "1.0", "sigma": "0", "speedFactor": "1.0", "guiShape": "passenger/sedan",
}


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def code_provenance() -> dict:
    """Identifie le code utilisé pour produire les résultats."""
    digest = hashlib.sha256()
    for name in ("road_network.py", "traffic_demand.py", "traffic_run.py",
                 "vehicle_tracking.py", "sumo_process.py", "sumo_snapshot.py",
                 "kintambo_scenarios.py", "crdg_live.py", "crdg_panel.py", "sumo_view.py"):
        digest.update(name.encode() + Path(__file__).with_name(name).read_bytes())
    digest.update(b"crdg.py" + (Path(__file__).parents[1] / "crdg.py").read_bytes())
    digest.update(b"blockage_observation.py" + (Path(__file__).parents[1] / "blockage_observation.py").read_bytes())
    digest.update(b"cli.py" + (Path(__file__).parents[1] / "cli.py").read_bytes())
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[3])
        state = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               cwd=Path(__file__).resolve().parents[3])
    except OSError:
        # Un package installé reste identifiable même sans Git.
        return {"sha256": digest.hexdigest(), "git_commit": None, "git_state": None}
    return {"sha256": digest.hexdigest(), "git_commit": commit.stdout.strip() or None,
            "git_state": ("dirty" if state.stdout.strip() else "clean") if state.returncode == 0 else None}


def write_manifest(output, prepared, result, output_mode):
    """Garde les entrées et l'état réel du run ; une erreur n'est pas un résultat positif."""
    inputs = ("network.net.xml", "traffic.rou.xml", "simulation.sumocfg", "view.xml", "scenario.json")
    write_json(output / "manifest.json", {
        "format_version": 1, "method": "traffic_observation", "crdg_enabled": result["crdg_enabled"], "diagnostic": "not_implemented",
        "scenario": result.get("kintambo_case", "kintambo"), "seed": result["seed"],
        "configuration": prepared.get("configuration"), "code": result["code"],
        "sumo_version": result.get("sumo_version"), "traci_version": result.get("traci_version"),
        "traci_protocol": result.get("traci_protocol"), "step_s": STEP_S,
        "status": result["status"], "reason": result["reason"],
        "display_state": result.get("display_state"), "output_mode": output_mode,
        "inputs_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() if (output / name).exists() else None for name in inputs},
        "artifacts": sorted(p.name for p in output.iterdir() if p.is_file()),
        "omitted_in_interactive": ["lanes.csv", "observations.jsonl", "crdg.jsonl", "crdg_events.jsonl"] if output_mode == "interactive" else [],
        "units": {"time": "s", "position": "m", "speed": "m/s", "rate": "vehicles/hour/entry"}})


def prepare_traffic(config: dict, output: Path, demand: str, seed: int,
                       duration_s: float | None, rate: float | None, *, street_names: bool = False) -> tuple[dict, list]:
    """Prépare le réseau, les missions et les fichiers SUMO."""
    vehicle_space = float(VEHICLE_TYPE["length"]) + float(VEHICLE_TYPE["minGap"])
    inspection, routes = build_network(config, output, vehicle_space_m=vehicle_space)
    case = config.get("controlled_case")
    if case:
        inspection["experimental_variant"] = prepare_case_network(case, routes, output / "network.net.xml")
        missions = case_missions(case, routes, seed)
        duration = math.ceil(max(m.scheduled_s for m in missions)) + STEP_S
        rates, demand = None, "CONTROLLED"
    else:
        duration = config["demand"]["duration_s"] if duration_s is None else duration_s
        intensity = config["demand"]["rates_veh_per_hour_per_entry"][demand] if rate is None else rate
        rates = {entry: intensity for entry in config["demand"]["entries"]}
        missions = poisson_missions(routes, config["demand"]["destination_weights"], rates, duration, seed, demand)
    write_sumo_inputs(output, inspection, routes, missions, seed, street_names=street_names)
    prepared = {"configuration": config, "network": inspection, "routes": routes,
                "seed": seed, "demand": demand, "duration_s": duration,
                "rates_veh_per_hour_per_entry": rates, "vehicle_type": VEHICLE_TYPE,
                "missions": [asdict(m) for m in missions]}
    if case:
        prepared.update({"controlled_case": case, "arrival_model": "explicit_seeded_schedule",
                         "phenomenon_observed": "not_evaluated"})
    prepared["files_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(output.iterdir()) if p.suffix not in (".log",)}
    write_json(output / "scenario.json", prepared)
    return prepared, missions


def write_sumo_inputs(output, inspection, routes, missions, seed, *, street_names=False):
    """Écrit les entrées avec le même véhicule et les mêmes contrôles de sécurité."""
    root = ET.Element("routes")
    ET.SubElement(root, "vType", VEHICLE_TYPE)
    for name, edges in sorted(routes.items()):
        ET.SubElement(root, "route", id=name, edges=" ".join(edges))
    for mission in missions:
        ET.SubElement(root, "vehicle", id=mission.vehicle_id, type=VEHICLE_TYPE["id"],
                      route=mission.route_id, depart=repr(mission.scheduled_s),
                      departLane="best", departSpeed="0")
    write_xml(output / "traffic.rou.xml", root)
    view = ET.Element("viewsettings")
    scheme = ET.SubElement(view, "scheme", name="kintambo")
    ET.SubElement(scheme, "edges", streetName_show=str(street_names).lower(), streetName_size="24",
                  streetName_constantSize="true", streetName_onlySelected="false",
                  streetName_color="0,0,160", streetName_bgColor="255,255,255")
    ET.SubElement(scheme, "vehicles", vehicleQuality="2", vehicleExaggeration="1.5", vehicleMinSize="1")
    ET.SubElement(scheme, "background", backgroundColor="238,240,235")
    left, bottom, right, top = inspection["view_boundary_m"]
    ET.SubElement(view, "viewport", x=str((left + right) / 2),
                  y=str((bottom + top) / 2), zoom="100")
    write_xml(output / "view.xml", view)
    root = ET.Element("configuration")
    for group, options in {
        "input": {"net-file": "network.net.xml", "route-files": "traffic.rou.xml"},
        "time": {"step-length": "0.5"},
        "processing": {"time-to-teleport": "-1", "max-depart-delay": "-1",
                       "collision.action": "warn", "collision.check-junctions": "true"},
        "random_number": {"seed": str(seed)},
        "gui_only": {"gui-settings-file": "view.xml"},
    }.items():
        parent = ET.SubElement(root, group)
        for name, value in options.items():
            ET.SubElement(parent, name, value=value)
    write_xml(output / "simulation.sumocfg", root)


def sample_physics(connection, ledger: TrafficLedger, lanes: dict, previous: dict,
                   readings: dict) -> tuple[list, list]:
    """Observe les files sans en déduire une capacité ni un gridlock.

    Un véhicule est arrêté en dessous de 0,1 m/s, comme dans SUMO.
    Une file atteint le début de voie si au moins deux véhicules sont arrêtés
    et si l'arrière du dernier est à 7,5 m ou moins (longueur + minGap).
    """
    vehicles = []
    by_lane = {}
    for item in sorted(ledger.observed_active):
        reading = readings[item]
        lane = reading["lane"]
        distance = reading["distance"]
        old = previous.get(item)
        leader = connection.vehicle.getLeader(item)
        row = {"time_s": ledger.time_s, "vehicle_id": item,
               "edge": reading["road_id"], "lane": lane,
               "lane_position_m": reading["lane_position"],
               "speed_m_per_s": reading["speed"],
               "distance_m": distance, "progress_m": distance - old if old is not None else None,
               "leader_id": leader[0] if leader else None,
               "leader_gap_m": leader[1] if leader else None,
               "route_index": reading["route_index"],
               "next_tls": connection.vehicle.getNextTLS(item)}
        previous[item] = distance
        vehicles.append(row)
        by_lane.setdefault(lane, []).append(row)
    rows = []
    for lane, length in sorted(lanes.items()):
        present = by_lane.get(lane, [])
        halted = [r for r in present if r["speed_m_per_s"] < 0.1]
        rear = min((r["lane_position_m"] - 5 for r in halted), default=None)
        nearest_rear = min((r["lane_position_m"] - 5 for r in present), default=length)
        rows.append({"time_s": ledger.time_s, "lane": lane, "length_m": length,
                     "vehicles": len(present), "halting": len(halted),
                     # SUMO 1.27.1 renvoie un ratio, même si sa docstring indique « % ».
                     "occupancy_ratio": connection.lane.getLastStepOccupancy(lane),
                     "upstream_free_m": max(0, nearest_rear),
                     "queue_extent_m": max(0, length - rear) if rear is not None else 0,
                     "queue_reaches_upstream": len(halted) >= 2 and rear <= 7.5})
    return vehicles, rows


def subscribed_readings(connection, *, include_length: bool = False, include_visual: bool = False) -> dict:
    """Lit les données du pas courant ensemble pour limiter les appels TraCI."""
    import traci.constants as tc

    variables = {"route": tc.VAR_EDGES, "position": tc.VAR_POSITION,
                 "road_id": tc.VAR_ROAD_ID, "route_index": tc.VAR_ROUTE_INDEX,
                 "shape": tc.VAR_SHAPECLASS, "lane": tc.VAR_LANE_ID,
                 "lane_position": tc.VAR_LANEPOSITION, "speed": tc.VAR_SPEED,
                 "distance": tc.VAR_DISTANCE}
    if include_length:
        variables["length"] = tc.VAR_LENGTH
    if include_visual:
        variables.update(angle=tc.VAR_ANGLE, width=tc.VAR_WIDTH)
    for item in connection.simulation.getDepartedIDList():
        connection.vehicle.subscribe(item, list(variables.values()))
    results = connection.vehicle.getAllSubscriptionResults()
    return {item: {name: results[item][code] for name, code in variables.items()}
            for item in connection.vehicle.getIDList()}


def selected_lanes(row: dict, best_lanes: tuple, lanes: dict, movements: dict) -> set[str]:
    connections = dependency_graph.junction_movements(row, lanes, movements)
    origin = row["lane"] if not lanes[row["lane"]]["internal"] else (
        connections[0]["from_lane"] if len(connections) == 1 else None)
    best = [r for r in best_lanes if r[0] == origin]
    path = {row["lane"]}
    if len(best) != 1:
        return path
    sequence = [lane for lane in best[0][5] if lane]
    path.update(sequence)
    for a, b in zip(sequence, sequence[1:]):
        matches = [c for c in movements.get((a, lanes[b]["edge"]), []) if c["lane"] == b]
        if len(matches) == 1:
            path.update(matches[0]["internal_lanes"])
    return path


def sample_following(connection, readings: dict, leaders: dict, lanes: dict, movements: dict,
                     cache: dict, *, halting_speed: float, leader_decel: float) -> dict:
    """Demande au modèle natif si le leader limite encore le progrès sûr."""
    following = {}
    for item, row in sorted(readings.items()):
        native = leaders.get(item)
        if row["speed"] >= halting_speed or not native or native[0] not in readings or native[0] == item:
            continue
        target, gap = native
        speed = connection.vehicle.getFollowSpeed(item, row["speed"], gap,
                                                  readings[target]["speed"], leader_decel, target)
        relation = "unknown"
        if readings[target]["lane"] == row["lane"]:
            relation = "longitudinal_following"
        elif speed < halting_speed:
            best = connection.vehicle.getBestLanes(item)
            path = selected_lanes(row, best, lanes, movements)
            if readings[target]["lane"] in path:
                relation = "longitudinal_following"
            elif lanes[readings[target]["lane"]]["internal"]:
                ego = dependency_graph.junction_movements(row, lanes, movements)
                foe = dependency_graph.junction_movements(readings[target], lanes, movements)
                # Deux traversées distinctes peuvent converger vers la même voie.
                converges = any(a["lane"] == b["lane"] for a in ego for b in foe)
                internal = set()
                for lane in {via for c in ego for via in c["internal_lanes"]}:
                    key = ("internal", lane)
                    if key not in cache:
                        cache[key] = connection.lane.getInternalFoes(lane)
                    internal.update(cache[key])
                if converges or readings[target]["lane"] in internal:
                    relation = "connection_obstacle"
        following[item] = dependency_graph.Following(target, gap, speed, relation)
    return following


def sample_junctions(connection, readings: dict, lanes: dict, movements: dict,
                     cache: dict, *, halting_speed: float, following: dict | None = None) -> dict:
    """Lit les conflits natifs des seuls véhicules arrêtés près d'une traversée."""
    observations = {}
    links_by_lane = {}
    for item, row in sorted(readings.items()):
        if row["speed"] >= halting_speed:
            continue
        info = lanes[row["lane"]]
        connections = dependency_graph.junction_movements(row, lanes, movements)
        if not connections:
            continue
        if row["lane"] not in links_by_lane:
            links_by_lane[row["lane"]] = connection.lane.getLinks(row["lane"], extended=True)
        links = [link for link in links_by_lane[row["lane"]]
                 if link[5] not in ("r", "y", "u") and link[3]
                 and any(link[0] == c["lane"] and (link[4] or row["lane"]) in c["internal_lanes"]
                         for c in connections)]
        if not links:
            continue
        # On couvre la traversée identifiée, pas une longueur de voiture arbitraire.
        remaining = info["length_m"] - row["lane_position"]
        def remaining_traversal(c):
            chain = c["internal_lanes"]
            ahead = chain[chain.index(row["lane"]) + 1:] if row["lane"] in chain else chain
            return sum(lanes[lane]["length_m"] for lane in ahead)
        traversal = max(remaining_traversal(c) for c in connections)
        foes = connection.vehicle.getJunctionFoes(item, max(remaining + traversal, math.ulp(1.0)))
        if not foes:
            continue
        internal = {}
        for lane in sorted({foe[5] for foe in foes}):
            key = ("internal", lane)
            if key not in cache:
                cache[key] = connection.lane.getInternalFoes(lane)
            internal[lane] = cache[key]
        priorities = {}
        for c in connections:
            key = ("priority", c["from_lane"], c["lane"])
            if key not in cache:
                cache[key] = connection.lane.getFoes(c["from_lane"], c["lane"])
            priorities[(c["from_lane"], c["lane"])] = cache[key]
        evidence = dependency_graph.limiting_following(item, readings, following or {}, halting_speed)
        # SUMO 1.27.1 garde au moins POSITION_EPS=0,1 m avant une ligne d'attente.
        stop_speed = connection.vehicle.getStopSpeed(item, row["speed"], max(0, remaining - 0.1))
        observations[item] = {"foes": foes, "links": links, "internal_foes": internal,
                              "priority_foes": priorities, "stop_line_speed": stop_speed,
                              "halting_speed": halting_speed,
                              "connection_blocker": evidence.vehicle_id if evidence and
                              evidence.relation == "connection_obstacle" else None}
    return observations


def run_traffic(output_dir: str | Path, *, demand: str = "LOW", seed: int = 1,
                   gui: bool = False, gui_delay_ms: int = 100, street_names: bool = False, config_path=None,
                   duration_s: float | None = None, rate: float | None = None,
                   drain_horizon_s: float | None = None, crdg: bool = False,
                   blockage_evidence: bool = False, scenario: str = "kintambo",
                   kintambo_case: str | None = None, crdg_scene_times: tuple = (),
                   crdg_focus: str | None = None, crdg_depth: int = 2, crdg_live: bool = False,
                   close_on_end: bool = False, output_mode: str = "full") -> dict:
    """Prépare et suit le trafic sans arrêt imposé, changement de route ni assistance.

    À la limite de temps, un trafic non vidé reste incomplet, pas un gridlock
    confirmé. Les erreurs de circulation ou de fermeture restent des échecs,
    et les fichiers existants ne sont jamais écrasés.
    """
    if type(gui_delay_ms) is not int or gui_delay_ms < 0:
        raise ValueError("Le délai graphique doit être un entier positif ou nul.")
    if type(close_on_end) is not bool or output_mode not in ("full", "interactive"):
        raise ValueError("Mode de fermeture ou de sortie invalide.")
    if crdg_live and not gui:
        raise ValueError("INFO C-RDG animé demande --gui.")
    if (crdg_focus is not None or crdg_depth != 2) and not crdg_scene_times and not crdg_live:
        raise ValueError("Focus et voisinage demandent un instant de scène C-RDG.")
    if type(crdg_depth) is not int or crdg_depth not in (1, 2, 3):
        raise ValueError("Le voisinage doit valoir 1, 2 ou 3.")
    crdg = crdg or blockage_evidence or bool(crdg_scene_times) or crdg_live
    if scenario != "kintambo":
        raise ValueError("Les anciens carrefours indépendants sont retirés ; choisir un scénario Kintambo avec cgr scenarios.")
    if kintambo_case and (config_path is not None or rate is not None or duration_s is not None or demand != "LOW"):
        raise ValueError("La variante Kintambo utilise le réseau canonique et ses départs définis.")
    config = read_config(config_path)
    if kintambo_case:
        config["controlled_case"] = read_case(kintambo_case)
    if demand not in config["demand"]["rates_veh_per_hour_per_entry"]:
        raise ValueError("Niveau de demande absent de la configuration.")
    default_drain = config.get("controlled_case", {}).get("drain_s", config["simulation"]["drain_horizon_s"])
    drain = default_drain if drain_horizon_s is None else drain_horizon_s
    if not math.isfinite(drain) or drain <= 0 or drain % STEP_S:
        raise ValueError("L'horizon doit être positif et multiple de 0,5 s.")
    graph_settings = {key: value for key, value in config["crdg"].items() if key != "min_wait_s"} if crdg else None
    if crdg:
        interval = graph_settings["sample_interval_s"]
        calculation = graph_settings.get("calculation_interval_s", STEP_S)
        if (not math.isfinite(interval) or interval <= 0 or interval % STEP_S
                or not math.isfinite(calculation) or calculation <= 0 or calculation % STEP_S
                or interval % calculation):
            raise ValueError("Cadence de calcul ou d'export C-RDG invalide.")
        if crdg_live and 5 % calculation:
            raise ValueError("INFO C-RDG à 5 s demande une cadence de calcul qui divise 5 s.")
    if any(not math.isfinite(time) or time <= 0 or time % calculation for time in crdg_scene_times):
        raise ValueError("Chaque scène demande un instant positif de calcul C-RDG.")
    scene_times = set(crdg_scene_times)
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Le dossier de résultats doit être absent ou vide.")
    output.mkdir(parents=True, exist_ok=True)
    try:
        prepared, missions = prepare_traffic(config, output, demand, seed, duration_s, rate,
                                            street_names=street_names)
    except Exception as error:
        write_json(output / "preparation_error.json", {"status": "failed", "reason": str(error)})
        raise
    ledger = TrafficLedger(missions)
    if crdg_focus is not None and crdg_focus not in ledger.missions:
        raise ValueError("Le véhicule choisi n'appartient pas aux missions préparées.")
    if crdg_live:
        configure_live_view(output / "view.xml")
        prepared["files_sha256"]["view.xml"] = hashlib.sha256((output / "view.xml").read_bytes()).hexdigest()
        prepared["info_crdg"] = {"update_interval_s": 5, "focus": crdg_focus}
        write_json(output / "scenario.json", prepared)
    horizon = math.ceil(prepared["duration_s"] / STEP_S) * STEP_S + drain
    result = {"scenario": scenario, "demand": prepared.get("demand", demand), "seed": seed,
              "rates_veh_per_hour_per_entry": prepared["rates_veh_per_hour_per_entry"],
              "duration_s": prepared["duration_s"], "horizon_s": horizon,
              "scenario_sha256": hashlib.sha256((output / "scenario.json").read_bytes()).hexdigest(),
              "network_sha256": hashlib.sha256((output / "network.net.xml").read_bytes()).hexdigest(),
              "step_s": STEP_S, "gui": gui, "status": "failed", "reason": None,
              "output_mode": output_mode, "crdg_enabled": crdg,
              "code": code_provenance(), "connection_closed": False, "process_stopped": False,
              "process_returncode": None, "forced_process_stop": False, "cleanup_errors": [],
              "collision_ids": [], "gridlock": "not_evaluated",
              "observation_interval_s": 5, "halting_speed_m_per_s": 0.1,
              "upstream_queue_margin_m": 7.5,
              "max_halting": 0, "max_lane_occupancy_ratio": 0, "upstream_queue_samples": 0}
    connection = process = None
    panel = CrdgPanelLink(output, missions, focus=crdg_focus,
                         sector_boundary=local_sector(output / "network.net.xml", prepared["network"]["view_boundary_m"]),
                         network_boundary=whole_network(output / "network.net.xml", prepared["network"]["view_boundary_m"])) if crdg_live else None
    if kintambo_case:
        result["kintambo_case"] = kintambo_case
    if scene_times:
        result["crdg_scenes"] = {"requested_times_s": sorted(scene_times), "recorded_times_s": [], "focus_absent_times_s": []}
    previous = {}
    trips = {}
    first_halted = {}
    junction_cache = {}
    trails, dependency_history = {}, {}
    graph_summary, graph_peak = dependency_graph.empty_summary(), None
    evidence = BlockageObservation() if blockage_evidence else None
    reader = (lambda connection: subscribed_readings(connection, include_length=True, include_visual=crdg_live)) if crdg else subscribed_readings
    with (output / "sumo.log").open("w", encoding="utf-8") as log, \
            (output / "timeline.csv").open("w", encoding="utf-8", newline="") as trace, \
            (output / "lanes.csv" if output_mode == "full" else Path(os.devnull)).open("w", encoding="utf-8", newline="") as lane_file, \
            (output / "observations.jsonl" if output_mode == "full" else Path(os.devnull)).open("w", encoding="utf-8") as observations, \
            ((output / "crdg.jsonl" if output_mode == "full" else Path(os.devnull)).open("w", encoding="utf-8") if crdg else nullcontext()) as graph_file, \
            ((output / "crdg_events.jsonl" if output_mode == "full" else Path(os.devnull)).open("w", encoding="utf-8") if crdg else nullcontext()) as event_file, \
            ((output / "blockage_events.jsonl").open("w", encoding="utf-8") if evidence else nullcontext()) as evidence_file:
        timeline = csv.DictWriter(trace, fieldnames=list(ledger.snapshot()))
        timeline.writeheader()
        lane_writer = csv.DictWriter(lane_file, fieldnames=["time_s", "lane", "length_m", "vehicles",
                                                          "halting", "occupancy_ratio", "upstream_free_m", "queue_extent_m",
                                                          "queue_reaches_upstream"])
        lane_writer.writeheader()
        try:
            import traci
            if crdg:
                graph_lanes, movements = dependency_graph.read_network(output / "network.net.xml")
            binary = require_binary("sumo-gui" if gui else "sumo")
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            command = [binary, "-c", str(output / "simulation.sumocfg"), "--remote-port", str(port),
                       "--tripinfo-output", str(output / "tripinfo.xml"),
                       "--tripinfo-output.write-unfinished", "true", "--no-step-log", "true",
                       "--duration-log.disable", "true"]
            if scene_times:
                # La précision concerne le fichier d'état, pas les calculs du trafic.
                command.extend(["--save-state.precision", "17"])
            if gui:
                # La connexion reste ouverte pendant l'inspection ; close termine ensuite la GUI.
                command.extend(["--start", "true", "--quit-on-end", "true", "--delay", str(gui_delay_ms)])
            result["command"] = command
            process = subprocess.Popen(command, stdout=log, stderr=log)
            connection = traci.connect(port=port, proc=process, numRetries=300, waitBetweenRetries=0.1)
            if gui:
                connection.gui.setSchema("View #0", prepared.get("gui_scheme", "kintambo"))
                left, bottom, right, top = prepared["network"]["view_boundary_m"]
                connection.gui.setBoundary("View #0", *(panel.sector_boundary if panel else (left, bottom, right, top)))
            if panel:
                panel.sumo_process = process
                panel.start()
            result["traci_protocol"], result["sumo_version"] = connection.getVersion()
            result["traci_version"] = traci.__version__
            if abs(connection.simulation.getDeltaT() - STEP_S) > 1e-9:
                raise RuntimeError("Pas de simulation chargé incorrect.")
            for route_id, edges in prepared["routes"].items():
                if list(connection.route.getEdges(route_id)) != edges:
                    raise RuntimeError("La route chargée diffère de la mission.")
            route_edges = {edge for mission in missions for edge in mission.route}
            lanes = {lane: connection.lane.getLength(lane) for lane in connection.lane.getIDList()
                     if not lane.startswith(":") and connection.lane.getEdgeID(lane) in route_edges}
            for _ in range(int(horizon / STEP_S)):
                if panel:
                    if not panel.before_step(connection, ledger.readings or {}, ledger.time_s):
                        result["status"] = "user_closed"
                        result["reason"] = "Démonstration fermée explicitement avant la fin des missions."
                        break
                connection.simulationStep()
                result["collision_ids"].extend(connection.simulation.getCollidingVehiclesIDList())
                timeline.writerow(ledger.observe(connection, reader))
                if evidence:
                    # Les arrivées validées ne doivent pas attendre le prochain graphe.
                    for event in evidence.record_arrivals(ledger.arrivals):
                        evidence_file.write(json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                vehicles = tls_states = None
                if crdg:
                    dependency_graph.update_waiting(first_halted, ledger.readings, ledger.time_s,
                                                     result["halting_speed_m_per_s"])
                    footprints = dependency_graph.update_footprints(trails, ledger.readings, graph_lanes, movements)
                if ledger.time_s % 5 == 0:
                    vehicles, lane_rows = sample_physics(connection, ledger, lanes, previous, ledger.readings)
                    tls_states = {item: connection.trafficlight.getRedYellowGreenState(item)
                                  for item in connection.trafficlight.getIDList()}
                    observations.write(json.dumps({"time_s": ledger.time_s, "vehicles": vehicles,
                                                   "tls": tls_states}) + "\n")
                    lane_writer.writerows(lane_rows)
                    result["max_halting"] = max(result["max_halting"], sum(r["halting"] for r in lane_rows))
                    result["max_lane_occupancy_ratio"] = max(result["max_lane_occupancy_ratio"],
                                                           max((r["occupancy_ratio"] for r in lane_rows), default=0))
                    result["upstream_queue_samples"] += sum(r["queue_reaches_upstream"] for r in lane_rows)
                    if gui and ledger.time_s == min(300, math.floor(horizon / 10) * 5):
                        connection.gui.screenshot("View #0", str(output / "view.png"))
                if crdg and ledger.time_s % calculation == 0:
                    # Les lectures déjà faites pour les files servent aussi au graphe.
                    if vehicles is not None:
                        leaders = {r["vehicle_id"]: (r["leader_id"], r["leader_gap_m"])
                                   for r in vehicles if r["leader_id"]}
                    else:
                        leaders = {item: connection.vehicle.getLeader(item) for item in first_halted}
                    if tls_states is None:
                        tls_states = {item: connection.trafficlight.getRedYellowGreenState(item)
                                      for item in connection.trafficlight.getIDList()}
                    following = sample_following(connection, ledger.readings, leaders, graph_lanes, movements,
                                                  junction_cache, halting_speed=result["halting_speed_m_per_s"],
                                                  leader_decel=float(VEHICLE_TYPE["decel"]))
                    junctions = sample_junctions(connection, ledger.readings, graph_lanes, movements,
                                                junction_cache, halting_speed=result["halting_speed_m_per_s"],
                                                following=following)
                    graph = dependency_graph.build_graph(
                        ledger.readings, dependency_graph.receiving_spaces(ledger.readings, graph_lanes, footprints),
                        graph_lanes, movements, tls_states, first_halted, ledger.time_s,
                        halting_speed=result["halting_speed_m_per_s"], min_gap_m=float(VEHICLE_TYPE["minGap"]),
                        junctions=junctions, following=following)
                    dependency_events = dependency_graph.track_dependencies(graph, dependency_history)
                    for event in dependency_events:
                        event_file.write(json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                    current = dependency_graph.snapshot(graph)
                    graph_peak = dependency_graph.record_snapshot(graph_summary, current, graph_peak)
                    if panel:
                        panel.observe(connection, graph, current, ledger.readings)
                    if ledger.time_s in scene_times:
                        saved = save_scene(connection, current, ledger.readings,
                                           output / "crdg_scenes" / f"{ledger.time_s:g}", crdg_focus, crdg_depth,
                                           {"code": result["code"], "network_sha256": result["network_sha256"],
                                            "scenario_sha256": result["scenario_sha256"], "seed": seed,
                                            "sumo_version": result["sumo_version"]})
                        result["crdg_scenes"]["recorded_times_s"].append(ledger.time_s)
                        if saved.get("focus_present") is False:
                            result["crdg_scenes"]["focus_absent_times_s"].append(ledger.time_s)
                    if evidence:
                        for event in evidence.observe(ledger.time_s, ledger.readings, current, graph_lanes, movements,
                                                      arrivals=ledger.arrivals, dependency_events=dependency_events):
                            evidence_file.write(json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                    if ledger.time_s % interval == 0:
                        graph_file.write(json.dumps(current, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                        graph_summary["exported_sample_count"] = graph_summary.get("exported_sample_count", 0) + 1
                if len(ledger.arrivals) == len(missions):
                    result["status"] = "completed"
                    break
            else:
                result["status"] = "horizon_reached"
                result["reason"] = "Des missions restent présentes ou en attente à l'horizon."
            if gui and not close_on_end and result["status"] != "user_closed":
                result.update(counts=ledger.snapshot(), display_state="awaiting_close",
                              last_validated_state=ledger.last_validated_state)
                write_json(output / "summary.json", result)
                write_manifest(output, prepared, result, output_mode)
                for stream in (trace, lane_file, observations, graph_file, event_file, evidence_file):
                    if stream is not None:
                        stream.flush()
                if panel:
                    panel.hold_view(connection, ledger.readings, result["status"], ledger.time_s,
                                    reason=result["reason"] or "Toutes les missions sont arrivées.", counts=ledger.snapshot())
                elif sys.stdin.isatty():
                    input("Observer la vue, puis appuyer sur Entrée pour fermer SUMO-GUI. ")
        except Exception as error:
            result["status"] = "failed"
            result["reason"] = str(error)
            result.update(counts=ledger.snapshot(), display_state="error")
            write_json(output / "summary.json", result)
            write_manifest(output, prepared, result, output_mode)
            if panel and not close_on_end and connection is not None and panel.process is not None and panel.process.poll() is None:
                try:
                    panel.hold_view(connection, ledger.readings or {}, "failed", ledger.time_s,
                                    reason=str(error), counts=ledger.snapshot())
                except Exception as inspection_error:
                    result["inspection_error"] = str(inspection_error)
        finally:
            if panel:
                try:
                    panel.finish(connection, ledger.readings, result["status"], ledger.time_s)
                except Exception as error:
                    panel.stats["errors"].append(str(error))
                    result["info_crdg_error"] = str(error)
            close_sumo(connection, process, result)
    result["simulation_outcome"] = result["status"]
    if result.get("info_crdg_error"):
        result["status"] = "failed"
        result["reason"] = f"Fermeture INFO C-RDG en erreur : {result['info_crdg_error']}"
    if (result["cleanup_errors"] or result["forced_process_stop"] or not result["connection_closed"]
            or not result["process_stopped"] or result["process_returncode"] != 0):
        result["status"] = "failed"
        result["reason"] = "Fermeture normale non confirmée." + (f" Avant fermeture : {result['reason']}" if result["reason"] else "")
    try:
        trips = verify_trips(output / "tripinfo.xml", ledger)
    except Exception as error:
        result["status"] = "failed"
        result["reason"] = result["reason"] or str(error)
    result["counts"] = ledger.snapshot()
    result["display_state"] = "closed" if gui else "not_requested"
    if panel:
        result["info_crdg"] = {"directory": "info_crdg", "update_interval_s": 5, **panel.stats}
    if scene_times:
        result["crdg_scenes"]["unobserved_times_s"] = sorted(scene_times - set(result["crdg_scenes"]["recorded_times_s"]))
    if evidence:
        with (output / "blockage_events.jsonl").open("a", encoding="utf-8") as evidence_file:
            for event in evidence.finish(ledger.time_s, result["status"]):
                evidence_file.write(json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        evidence_summary = {**evidence.summary(), "code": result["code"], "network_sha256": result["network_sha256"],
                            "seed": seed, "demand": result["demand"], "run_status": result["status"],
                            "calculation_interval_s": calculation}
        write_json(output / "blockage_summary.json", evidence_summary)
        result["blockage_evidence"] = {"enabled": True, "events": "blockage_events.jsonl",
                                      "summary": "blockage_summary.json"}
    result["failure_observation"] = ledger.failure_observation
    result["last_validated_state"] = ledger.last_validated_state
    result["remaining_ids"] = sorted(set(ledger.missions) - set(ledger.arrivals))
    records = ledger.vehicle_records(trips)
    with (output / "vehicles.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = list(records[0]) if records else [
            "vehicle_id", "regime", "entry_gate", "exit_gate", "route_id", "route",
            "destination", "scheduled_s", "sampled_s", "status", "actual_departure_s",
            "observed_active", "validated_active", "actual_arrival_s",
            "arrival_event_s", "last_observation",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    write_json(output / "summary.json", result)
    if crdg:
        graph_summary.update({"settings": graph_settings, "networkx_version": dependency_graph.nx.__version__,
                              "halting_speed_m_per_s": result["halting_speed_m_per_s"],
                              "calculation_interval_s": calculation, "export_interval_s": interval,
                              "dependency_history": dependency_history.get("statistics", {}),
                              "active_dependencies": list(dependency_history.get("active", {}).values()),
                              "required_space_basis": "vehicle_length + minGap",
                              "internal_conflicts": "native_active_constraints_only", "gridlock": "not_evaluated",
                              "junction_evidence": "native_priority_at_stopline_or_connection_obstacle",
                              "receiving_release_mode": "one_candidate_free",
                              "closed_cycles": "observed_release_rules_fixed_point",
                              "run_status": result["status"], "network_sha256": result["network_sha256"],
                              "scenario_sha256": result["scenario_sha256"], "code": result["code"]})
        write_json(output / "crdg_summary.json", graph_summary)
        write_json(output / "crdg_peak.json", graph_peak)
    write_manifest(output, prepared, result, output_mode)
    return result
