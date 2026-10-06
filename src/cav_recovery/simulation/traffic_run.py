"""Préparation et exécution d'une expérience sans intervention sur les véhicules."""

import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess
import sys

from .road_network import build_network, read_config, require_binary, write_xml
from .traffic_demand import poisson_missions
from .sumo_process import close_sumo
from .vehicle_tracking import TrafficLedger, verify_trips, STEP_S

import xml.etree.ElementTree as ET


VEHICLE_TYPE = {
    "id": "passenger_CAV", "vClass": "passenger", "carFollowModel": "Krauss",
    "length": "5.0", "minGap": "2.5", "accel": "2.6", "decel": "4.5",
    "tau": "1.0", "sigma": "0", "speedFactor": "1.0", "guiShape": "passenger/sedan",
}


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def code_provenance() -> dict:
    digest = hashlib.sha256()
    for name in ("road_network.py", "traffic_demand.py", "traffic_run.py",
                 "vehicle_tracking.py", "sumo_process.py"):
        digest.update(name.encode() + Path(__file__).with_name(name).read_bytes())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[3])
    state = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                           cwd=Path(__file__).resolve().parents[3])
    return {"sha256": digest.hexdigest(), "git_commit": commit.stdout.strip() or None,
            "git_state": ("dirty" if state.stdout.strip() else "clean") if state.returncode == 0 else None}


def prepare_traffic(config: dict, output: Path, demand: str, seed: int,
                       duration_s: float | None, rate: float | None) -> tuple[dict, list]:
    vehicle_space = float(VEHICLE_TYPE["length"]) + float(VEHICLE_TYPE["minGap"])
    inspection, routes = build_network(config, output, vehicle_space_m=vehicle_space)
    duration = config["demand"]["duration_s"] if duration_s is None else duration_s
    intensity = config["demand"]["rates_veh_per_hour_per_entry"][demand] if rate is None else rate
    rates = {entry: intensity for entry in config["demand"]["entries"]}
    missions = poisson_missions(routes, config["demand"]["destination_weights"], rates, duration, seed, demand)
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
    ET.SubElement(scheme, "edges", streetName_show="true", streetName_size="30",
                  streetName_constantSize="true", streetName_onlySelected="false",
                  streetName_color="0,0,160", streetName_bgColor="255,255,255")
    ET.SubElement(scheme, "vehicles", vehicleQuality="2", vehicleExaggeration="1.5", vehicleMinSize="1")
    ET.SubElement(scheme, "background", backgroundColor="238,240,235")
    ET.SubElement(view, "viewport", x=str(inspection["center_xy_m"][0]),
                  y=str(inspection["center_xy_m"][1]), zoom="1500")
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
    prepared = {"configuration": config, "network": inspection, "routes": routes,
                "seed": seed, "demand": demand, "duration_s": duration,
                "rates_veh_per_hour_per_entry": rates, "vehicle_type": VEHICLE_TYPE,
                "missions": [asdict(m) for m in missions]}
    prepared["files_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(output.iterdir()) if p.suffix not in (".log",)}
    write_json(output / "scenario.json", prepared)
    return prepared, missions


def sample_physics(connection, ledger: TrafficLedger, lanes: dict, previous: dict,
                   readings: dict) -> tuple[list, list]:
    """Mesure les files, pas une capacité ni un diagnostic de gridlock.

    L'arrêt suit la définition SUMO (< 0,1 m/s). Une file atteignant l'amont
    est signalée quand au moins deux véhicules sont arrêtés et que l'arrière
    du dernier laisse moins d'un gabarit nominal (5 + 2,5 m) au début de voie.
    Ce signal géométrique seul reste un candidat de spillback.
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
                     # SUMO 1.27.1 renvoie un ratio, malgré la docstring Python « % ».
                     "occupancy_ratio": connection.lane.getLastStepOccupancy(lane),
                     "upstream_free_m": max(0, nearest_rear),
                     "queue_extent_m": max(0, length - rear) if rear is not None else 0,
                     "queue_reaches_upstream": len(halted) >= 2 and rear <= 7.5})
    return vehicles, rows


def subscribed_readings(connection) -> dict:
    """Lit le même pas en une réponse groupée plutôt qu'un appel par attribut."""
    import traci.constants as tc

    variables = {"route": tc.VAR_EDGES, "position": tc.VAR_POSITION,
                 "road_id": tc.VAR_ROAD_ID, "route_index": tc.VAR_ROUTE_INDEX,
                 "shape": tc.VAR_SHAPECLASS, "lane": tc.VAR_LANE_ID,
                 "lane_position": tc.VAR_LANEPOSITION, "speed": tc.VAR_SPEED,
                 "distance": tc.VAR_DISTANCE}
    for item in connection.simulation.getDepartedIDList():
        connection.vehicle.subscribe(item, list(variables.values()))
    results = connection.vehicle.getAllSubscriptionResults()
    return {item: {name: results[item][code] for name, code in variables.items()}
            for item in connection.vehicle.getIDList()}


def run_traffic(output_dir: str | Path, *, demand: str = "LOW", seed: int = 1,
                   gui: bool = False, gui_delay_ms: int = 100, config_path=None,
                   duration_s: float | None = None, rate: float | None = None,
                   drain_horizon_s: float | None = None) -> dict:
    """Prépare puis observe une demande sans stop, reroutage ni assistance.

    Un horizon atteint est un résultat incomplet, pas un gridlock confirmé.
    Les erreurs d'intégrité ou de fermeture ont un statut distinct.
    Aucun fichier existant n'est écrasé.
    """
    if type(gui_delay_ms) is not int or gui_delay_ms < 0:
        raise ValueError("Le délai graphique doit être un entier positif ou nul.")
    config = read_config(config_path)
    if demand not in config["demand"]["rates_veh_per_hour_per_entry"]:
        raise ValueError("Niveau de demande absent de la configuration.")
    drain = config["simulation"]["drain_horizon_s"] if drain_horizon_s is None else drain_horizon_s
    if not math.isfinite(drain) or drain <= 0 or drain % STEP_S:
        raise ValueError("L'horizon doit être positif et multiple de 0,5 s.")
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Le dossier de résultats doit être absent ou vide.")
    output.mkdir(parents=True, exist_ok=True)
    try:
        prepared, missions = prepare_traffic(config, output, demand, seed, duration_s, rate)
    except Exception as error:
        write_json(output / "preparation_error.json", {"status": "failed", "reason": str(error)})
        raise
    ledger = TrafficLedger(missions)
    horizon = math.ceil(prepared["duration_s"] / STEP_S) * STEP_S + drain
    result = {"scenario": "kintambo", "demand": demand, "seed": seed,
              "rates_veh_per_hour_per_entry": prepared["rates_veh_per_hour_per_entry"],
              "duration_s": prepared["duration_s"], "horizon_s": horizon,
              "scenario_sha256": hashlib.sha256((output / "scenario.json").read_bytes()).hexdigest(),
              "network_sha256": hashlib.sha256((output / "network.net.xml").read_bytes()).hexdigest(),
              "step_s": STEP_S, "gui": gui, "status": "failed", "reason": None,
              "code": code_provenance(), "connection_closed": False, "process_stopped": False,
              "process_returncode": None, "forced_process_stop": False, "cleanup_errors": [],
              "collision_ids": [], "gridlock": "not_evaluated",
              "observation_interval_s": 5, "halting_speed_m_per_s": 0.1,
              "upstream_queue_margin_m": 7.5,
              "max_halting": 0, "max_lane_occupancy_ratio": 0, "upstream_queue_samples": 0}
    connection = process = None
    previous = {}
    trips = {}
    with (output / "sumo.log").open("w", encoding="utf-8") as log, \
            (output / "timeline.csv").open("w", encoding="utf-8", newline="") as trace, \
            (output / "lanes.csv").open("w", encoding="utf-8", newline="") as lane_file, \
            (output / "observations.jsonl").open("w", encoding="utf-8") as observations:
        timeline = csv.DictWriter(trace, fieldnames=list(ledger.snapshot()))
        timeline.writeheader()
        lane_writer = csv.DictWriter(lane_file, fieldnames=["time_s", "lane", "length_m", "vehicles",
                                                          "halting", "occupancy_ratio", "upstream_free_m", "queue_extent_m",
                                                          "queue_reaches_upstream"])
        lane_writer.writeheader()
        try:
            import traci
            binary = require_binary("sumo-gui" if gui else "sumo")
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            command = [binary, "-c", str(output / "simulation.sumocfg"), "--remote-port", str(port),
                       "--tripinfo-output", str(output / "tripinfo.xml"),
                       "--tripinfo-output.write-unfinished", "true", "--no-step-log", "true",
                       "--duration-log.disable", "true"]
            if gui:
                command.extend(["--start", "true", "--quit-on-end", "true", "--delay", str(gui_delay_ms)])
            result["command"] = command
            process = subprocess.Popen(command, stdout=log, stderr=log)
            connection = traci.connect(port=port, proc=process, numRetries=300, waitBetweenRetries=0.1)
            if gui:
                connection.gui.setSchema("View #0", "kintambo")
                connection.gui.setOffset("View #0", *prepared["network"]["center_xy_m"])
                connection.gui.setZoom("View #0", 1500)
            result["sumo_version"] = connection.getVersion()[1]
            if abs(connection.simulation.getDeltaT() - STEP_S) > 1e-9:
                raise RuntimeError("Pas de simulation chargé incorrect.")
            for route_id, edges in prepared["routes"].items():
                if list(connection.route.getEdges(route_id)) != edges:
                    raise RuntimeError("La route chargée diffère de la mission.")
            route_edges = {edge for mission in missions for edge in mission.route}
            lanes = {lane: connection.lane.getLength(lane) for lane in connection.lane.getIDList()
                     if not lane.startswith(":") and connection.lane.getEdgeID(lane) in route_edges}
            for _ in range(int(horizon / STEP_S)):
                connection.simulationStep()
                result["collision_ids"].extend(connection.simulation.getCollidingVehiclesIDList())
                timeline.writerow(ledger.observe(connection, subscribed_readings))
                if ledger.time_s % 5 == 0:
                    vehicles, lane_rows = sample_physics(connection, ledger, lanes, previous, ledger.readings)
                    observations.write(json.dumps({"time_s": ledger.time_s, "vehicles": vehicles,
                        "tls": {item: connection.trafficlight.getRedYellowGreenState(item)
                                for item in connection.trafficlight.getIDList()}}) + "\n")
                    lane_writer.writerows(lane_rows)
                    result["max_halting"] = max(result["max_halting"], sum(r["halting"] for r in lane_rows))
                    result["max_lane_occupancy_ratio"] = max(result["max_lane_occupancy_ratio"],
                                                           max((r["occupancy_ratio"] for r in lane_rows), default=0))
                    result["upstream_queue_samples"] += sum(r["queue_reaches_upstream"] for r in lane_rows)
                    if gui and ledger.time_s == min(300, math.floor(horizon / 10) * 5):
                        connection.gui.screenshot("View #0", str(output / "view.png"))
                if len(ledger.arrivals) == len(missions):
                    result["status"] = "completed"
                    break
            else:
                result["status"] = "horizon_reached"
                result["reason"] = "Des missions restent présentes ou en attente à l'horizon."
            if gui and sys.stdin.isatty():
                input("Observer la vue, puis appuyer sur Entrée pour fermer SUMO-GUI. ")
        except Exception as error:
            result["status"] = "failed"
            result["reason"] = str(error)
        finally:
            close_sumo(connection, process, result)
    if (result["cleanup_errors"] or result["forced_process_stop"] or not result["connection_closed"]
            or not result["process_stopped"] or result["process_returncode"] != 0):
        result["status"] = "failed"
        result["reason"] = result["reason"] or "Fermeture normale non confirmée."
    try:
        trips = verify_trips(output / "tripinfo.xml", ledger)
    except Exception as error:
        result["status"] = "failed"
        result["reason"] = result["reason"] or str(error)
    result["counts"] = ledger.snapshot()
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
    return result
