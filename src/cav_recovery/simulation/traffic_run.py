"""Exécution des missions et vérification de leur conservation jusqu'à l'arrivée."""

from __future__ import annotations

import csv
import hashlib
import importlib
import math
from pathlib import Path
import shutil
import socket
import subprocess
import xml.etree.ElementTree as ET

from .road_network import CENTER_NODE, CENTER_PHASES, check_environment
from .sumo_smoke import _close_run
from .traffic_demand import Mission, STEP_S, TrafficInputError, file_hash
from .traffic_scenario import new_output_directory, read_scenario, write_json


class TrafficLedger:
    """Suit des missions immuables ; aucun événement ne corrige une destination."""

    def __init__(self, missions: list[Mission]):
        self.missions = {mission.vehicle_id: mission for mission in missions}
        if not missions or len(self.missions) != len(missions):
            raise TrafficInputError("Les missions doivent avoir des identifiants uniques et être présentes.")
        self.departures: dict[str, float] = {}
        self.arrivals: dict[str, float] = {}
        self.last: dict[str, dict] = {}
        self.observed_active: set[str] = set()
        self.validated_active: set[str] = set()
        self.observed_time_s = 0.0
        self.failure_observation: dict | None = None
        self.last_validated_state = {"time_s": 0.0, "active_ids": [], "departed_ids": [], "arrived_ids": []}
        self.teleport_starts: list[dict] = []
        self.teleport_ends: list[dict] = []
        self.time_s = 0.0

    def observe(self, connection) -> dict:
        """Conserve ce qui a été lu, même si le pas ne peut pas être validé."""
        observation = {"time_s": None, "active_ids": None, "departed_ids": None, "arrived_ids": None,
                       "vehicles": {}}
        try:
            time_s = connection.simulation.getTime()
            observation["time_s"] = time_s
            if not math.isfinite(time_s) or abs(time_s - self.time_s - STEP_S) > 1e-7:
                raise RuntimeError("Le temps simulé ne progresse pas par pas de 0,5 s.")
            self.time_s = time_s
            self.observed_active = set(connection.vehicle.getIDList())
            self.observed_time_s = time_s
            observation["active_ids"] = sorted(self.observed_active)
            observation["departed_ids"] = list(connection.simulation.getDepartedIDList())
            observation["arrived_ids"] = list(connection.simulation.getArrivedIDList())
            for name, events, getter in (
                ("teleport_starts", self.teleport_starts, connection.simulation.getStartingTeleportIDList),
                ("teleport_ends", self.teleport_ends, connection.simulation.getEndingTeleportIDList),
            ):
                observation[name] = list(getter())
                events.extend({"vehicle_id": item, "time_s": time_s} for item in observation[name])
            observation["collision_ids"] = list(connection.simulation.getCollidingVehiclesIDList())
            self._validate_observation(connection, observation)
        except Exception as error:
            self.failure_observation = {**observation, "reason": str(error)}
            raise
        # La présence observée n'est pas une preuve de validation du pas entier.
        self.validated_active = self.observed_active.copy()
        self.last_validated_state = {"time_s": time_s, "active_ids": sorted(self.validated_active),
                                     "departed_ids": sorted(self.departures), "arrived_ids": sorted(self.arrivals)}
        self.failure_observation = None
        return self.snapshot()

    def _validate_observation(self, connection, observation: dict) -> None:
        time_s = self.time_s
        if self.teleport_starts or self.teleport_ends:
            raise RuntimeError("Téléportation détectée : aucune arrivée assistée n'est admise.")
        if observation["collision_ids"]:
            raise RuntimeError("Collision détectée.")
        departed = observation["departed_ids"]
        arrived = observation["arrived_ids"]
        active = self.observed_active
        if not (set(departed) | set(arrived) | active).issubset(self.missions):
            raise RuntimeError("Véhicule inconnu dans la simulation.")
        if len(set(departed)) != len(departed) or len(set(arrived)) != len(arrived):
            raise RuntimeError("Événement de départ ou d'arrivée répété.")
        for vehicle_id in departed:
            if vehicle_id in self.departures or vehicle_id not in active:
                raise RuntimeError("Départ répété ou sans véhicule actif observable.")
            actual = connection.vehicle.getDeparture(vehicle_id)
            if not math.isfinite(actual) or actual < self.missions[vehicle_id].scheduled_s or actual >= time_s:
                raise RuntimeError("Temps réel d'insertion incompatible avec la mission.")
            self.departures[vehicle_id] = actual
        for vehicle_id in sorted(active):
            mission = self.missions[vehicle_id]
            if vehicle_id not in self.departures or vehicle_id in self.arrivals:
                raise RuntimeError("Présence active incompatible avec les événements.")
            route = list(connection.vehicle.getRoute(vehicle_id))
            observation["vehicles"][vehicle_id] = {"route": route}
            if tuple(route) != mission.route:
                raise RuntimeError("Route ou destination modifiée pendant l'exécution.")
            position = list(connection.vehicle.getPosition(vehicle_id))
            if len(position) != 2 or not all(math.isfinite(value) for value in position):
                raise RuntimeError("Position non finie.")
            self.last[vehicle_id] = {"road_id": connection.vehicle.getRoadID(vehicle_id),
                                     "route_index": connection.vehicle.getRouteIndex(vehicle_id),
                                     "time_s": time_s, "position_m": position,
                                     "shape": connection.vehicle.getShapeClass(vehicle_id)}
        for vehicle_id in arrived:
            mission = self.missions[vehicle_id]
            last = self.last.get(vehicle_id)
            # L'événement retire le véhicule : la destination doit déjà avoir
            # été constatée, puis être corroborée par la sortie tripinfo.
            if (vehicle_id not in self.departures or vehicle_id in self.arrivals or vehicle_id in active
                    or last is None or last["road_id"] != mission.destination
                    or last["route_index"] != len(mission.route) - 1 or last["time_s"] >= time_s):
                raise RuntimeError("Arrivée non corroborée à la destination assignée.")
            self.arrivals[vehicle_id] = time_s
        expected_active = set(self.departures) - set(self.arrivals)
        if active != expected_active:
            raise RuntimeError("Disparition sans arrivée ou bilan des véhicules incohérent.")

    def snapshot(self) -> dict:
        pending = set(self.missions) - set(self.departures) - self.observed_active
        # SUMO traite [t - pas, t) lors du dernier pas : un départ à t reste futur.
        future = {item for item in pending if self.missions[item].scheduled_s >= self.time_s}
        delays = [time - self.missions[item].scheduled_s for item, time in self.departures.items()]
        return {"simulation_time_s": self.time_s, "scheduled": len(self.missions),
                "future": len(future), "delayed_not_inserted": len(pending - future),
                "pending": len(pending), "departed": len(self.departures), "active": len(self.observed_active),
                "observed_active": len(self.observed_active), "validated_active": len(self.validated_active),
                "observed_time_s": self.observed_time_s,
                "missing_without_arrival": len(set(self.departures) - set(self.arrivals) - self.observed_active),
                "active_unvalidated": len(self.observed_active) if self.failure_observation is not None else 0,
                "arrived": len(self.arrivals), "teleport_starts": len(self.teleport_starts),
                "teleport_ends": len(self.teleport_ends),
                "max_insertion_delay_s": max(delays, default=0),
                "mean_insertion_delay_s": sum(delays) / len(delays) if delays else None}

    def vehicle_records(self, trips: dict[str, dict]) -> list[dict]:
        records = []
        for item, mission in sorted(self.missions.items()):
            if item in self.observed_active:
                status = "active" if self.failure_observation is None else "active_unvalidated"
            elif item in self.arrivals:
                status = "arrived"
            elif item in self.departures:
                status = "missing_without_arrival"
            else:
                status = "future" if mission.scheduled_s >= self.time_s else "delayed_not_inserted"
            records.append({**mission.__dict__, "status": status, "actual_departure_s": self.departures.get(item),
                            "observed_active": item in self.observed_active, "validated_active": item in self.validated_active,
                            "actual_arrival_s": trips.get(item, {}).get("arrival_s"),
                            "arrival_event_s": self.arrivals.get(item), "last_observation": self.last.get(item)})
        return records


def verify_trips(path: Path, ledger: TrafficLedger) -> dict[str, dict]:
    """La sortie SUMO confirme une arrivée normale en fin d'arête, sans suppression."""
    trips = {}
    for row in ET.parse(path).getroot().findall("tripinfo"):
        item = row.attrib["id"]
        if item not in ledger.missions or item in trips:
            raise RuntimeError("La sortie tripinfo contient un véhicule inconnu ou répété.")
        mission = ledger.missions[item]
        arrival = float(row.attrib["arrival"])
        if arrival < 0:
            continue
        departure = float(row.attrib["depart"])
        arrival_edge = row.attrib["arrivalLane"].rsplit("_", 1)[0]
        if (item not in ledger.arrivals or not all(math.isfinite(t) for t in (arrival, departure))
                or arrival_edge != mission.destination or row.get("vaporized", "")
                or abs(departure - ledger.departures[item]) > 1e-7
                or not ledger.arrivals[item] - STEP_S <= arrival < ledger.arrivals[item]):
            raise RuntimeError("La sortie tripinfo ne confirme pas l'arrivée normale assignée.")
        trips[item] = {"arrival_s": arrival, "departure_s": departure, "arrival_lane": row.attrib["arrivalLane"]}
    if set(trips) != set(ledger.arrivals):
        raise RuntimeError("Des arrivées observées ne sont pas confirmées par tripinfo.")
    return trips


def verify_loaded_scenario(connection, missions: list[Mission]) -> None:
    """Contrôle les routes et le feu avant le premier pas, sans les modifier."""
    if abs(connection.simulation.getDeltaT() - STEP_S) > 1e-9:
        raise RuntimeError("Le pas de simulation chargé n'est pas de 0,5 s.")
    for mission in missions:
        if tuple(connection.route.getEdges(mission.route_id)) != mission.route:
            raise RuntimeError("La route chargée diffère de la mission assignée.")
    logics = connection.trafficlight.getAllProgramLogics(CENTER_NODE)
    current = connection.trafficlight.getProgram(CENTER_NODE)
    program = next((row for row in logics if row.programID == current), None)
    if (current != "0" or program is None or program.type != 0
            or [(p.duration, p.state) for p in program.phases] != CENTER_PHASES):
        raise RuntimeError("Le programme de feux chargé à C3 diffère du programme prévu.")


def code_provenance() -> dict:
    """Identifie le code utilisé, y compris avant son éventuel commit de revue."""
    digest = hashlib.sha256()
    for name in ("traffic_demand.py", "road_network.py", "traffic_scenario.py", "traffic_run.py", "sumo_smoke.py"):
        path = Path(__file__).with_name(name)
        digest.update(name.encode() + path.read_bytes())
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    state = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    return {"sha256": digest.hexdigest(), "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "git_state": ("dirty" if state.stdout.strip() else "clean") if state.returncode == 0 else None}


def observe_traffic(connection, ledger: TrafficLedger, horizon_s: float, result: dict, writer) -> None:
    """Avance jusqu'à vidange, sans dépasser l'horizon ni assister les véhicules."""
    verify_loaded_scenario(connection, list(ledger.missions.values()))
    previous_state = None
    for _ in range(int(horizon_s / STEP_S)):
        connection.simulationStep()
        result["steps"] += 1
        writer.writerow(ledger.observe(connection))
        state = connection.trafficlight.getRedYellowGreenState(CENTER_NODE)
        if state != previous_state:
            result["tls_states"].append({"time_s": ledger.time_s, "state": state})
            previous_state = state
        if len(ledger.arrivals) == len(ledger.missions):
            result["status"] = "passed"
            return
    raise RuntimeError("Horizon atteint : des missions ne sont pas arrivées normalement.")


def group_counts(records: list[dict], ledger: TrafficLedger) -> dict:
    ids = {row["vehicle_id"] for row in records}
    return {"scheduled": len(records), "departed": sum(row["actual_departure_s"] is not None for row in records),
            "active": sum(row["observed_active"] for row in records),
            "observed_active": sum(row["observed_active"] for row in records),
            "validated_active": sum(row["validated_active"] for row in records),
            **{state: sum(row["status"] == state for row in records)
               for state in ("arrived", "future", "delayed_not_inserted", "missing_without_arrival", "active_unvalidated")},
            "teleport_starts": sum(e["vehicle_id"] in ids for e in ledger.teleport_starts),
            "teleport_ends": sum(e["vehicle_id"] in ids for e in ledger.teleport_ends)}


def run_traffic(scenario_dir: str | Path, regime: str, output_dir: str | Path, *,
                gui: bool = False, gui_delay_ms: int = 100, drain_horizon_s: float = 600) -> dict:
    """Exécute la demande sans assistance et conserve aussi les bilans d'échec.

    L'affichage ne change pas le pas simulé. Après l'injection, l'horizon borne
    strictement l'attente des arrivées ; aucune mission restante n'est supprimée.
    """
    scenario_dir = Path(scenario_dir).resolve()
    output = new_output_directory(output_dir)
    manifest = read_scenario(scenario_dir)
    if regime not in manifest["regimes"]:
        raise TrafficInputError("Le niveau de charge doit être LOW, MID ou HIGH.")
    if (not math.isfinite(drain_horizon_s) or drain_horizon_s <= 0 or drain_horizon_s % STEP_S
            or type(gui_delay_ms) is not int or gui_delay_ms < 0):
        raise TrafficInputError("Horizon ou délai graphique invalide.")
    records = manifest["regimes"][regime]["missions"]
    missions = [Mission(**{**row, "route": tuple(row["route"])}) for row in records]
    ledger = TrafficLedger(missions)
    binary = "sumo-gui" if gui else "sumo"
    versions = check_environment(binary)
    horizon = manifest["regimes"][regime]["plan"]["injection_s"] + drain_horizon_s
    output.mkdir(parents=True, exist_ok=True)
    result = {"status": "failed", "reason": None, "regime": regime, "versions": versions,
              "scenario_sha256": file_hash(scenario_dir / "scenario.json"), "code": code_provenance(),
              "gui": gui, "gui_delay_ms": gui_delay_ms if gui else None, "step_s": STEP_S,
              "seed": 0, "time_to_teleport_s": -1, "drain_horizon_s": drain_horizon_s,
              "horizon_s": horizon, "steps": 0, "connection_closed": False, "process_stopped": False,
              "process_returncode": None, "cleanup_errors": [], "forced_process_stop": False,
              "tls_states": [], "assistance_commands": []}
    result["prepared_scenario"] = {"file": str(scenario_dir / "scenario.json"),
                                    "sha256": result["scenario_sha256"]}
    result["input_identities"] = {"contract": manifest.get("contract"),
                                   "osm": manifest.get("conversion", {}).get("source")}
    result["prepared_files_sha256"] = manifest.get("files_sha256", {})
    result["injection_s"] = manifest["regimes"][regime]["plan"]["injection_s"]
    process = connection = None
    trips = {}
    with (output / "sumo.log").open("w", encoding="utf-8") as log, \
            (output / "timeline.csv").open("w", encoding="utf-8", newline="") as trace:
        writer = csv.DictWriter(trace, fieldnames=list(ledger.snapshot()))
        writer.writeheader()
        writer.writerow(ledger.snapshot())
        try:
            traci = importlib.import_module("traci")
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            command = [shutil.which(binary), "-c", str(scenario_dir / regime / "simulation.sumocfg"),
                       "--remote-port", str(port), "--tripinfo-output", str(output / "tripinfo.xml"),
                       "--tripinfo-output.write-unfinished", "true", "--no-step-log", "true",
                       "--duration-log.disable", "true"]
            if gui:
                command.extend(["--start", "true", "--quit-on-end", "true", "--delay", str(gui_delay_ms)])
            result["command"] = command
            process = subprocess.Popen(command, stdout=log, stderr=log)
            connection = traci.connect(port=port, host="127.0.0.1", proc=process,
                                       numRetries=20, waitBetweenRetries=0.1)
            result["sumo_version"] = connection.getVersion()[1]
            observe_traffic(connection, ledger, horizon, result, writer)
        except Exception as error:
            result["status"] = "failed"
            result["reason"] = str(error)
        finally:
            # Le repli déjà éprouvé ferme aussi le processus après une erreur TraCI.
            try:
                _close_run(connection, process, result)
            except Exception as error:
                result["cleanup_errors"].append(f"Arrêt du processus non confirmé : {error}")
    if (result["cleanup_errors"] or result["forced_process_stop"] or not result["connection_closed"]
            or not result["process_stopped"] or result["process_returncode"] != 0):
        result["status"] = "failed"
        result["reason"] = result["reason"] or "Fermeture normale de SUMO/TraCI non confirmée."
    try:
        trips = verify_trips(output / "tripinfo.xml", ledger)
    except Exception as error:
        result["status"] = "failed"
        result["reason"] = result["reason"] or str(error)
    result["counts"] = ledger.snapshot()
    result["failure_observation"] = ledger.failure_observation
    result["last_validated_state"] = ledger.last_validated_state
    result["vehicles"] = ledger.vehicle_records(trips)
    result["teleport_events"] = {"starts": ledger.teleport_starts, "ends": ledger.teleport_ends}
    result["remaining_ids"] = sorted(set(ledger.missions) - set(ledger.arrivals))
    result["by_movement"] = []
    for entry, exit_gate in sorted({(m.entry_gate, m.exit_gate) for m in missions}):
        rows = [row for row in result["vehicles"] if row["entry_gate"] == entry and row["exit_gate"] == exit_gate]
        result["by_movement"].append({"entry_gate": entry, "exit_gate": exit_gate, **group_counts(rows, ledger)})
    result["by_entry"] = [{"entry_gate": entry, **group_counts(
        [row for row in result["vehicles"] if row["entry_gate"] == entry], ledger)}
        for entry in sorted({m.entry_gate for m in missions})]
    write_json(output / "summary.json", result)
    with (output / "vehicles.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["vehicle_id", "regime", "entry_gate", "exit_gate", "route_id", "route", "destination",
                  "scheduled_s", "actual_departure_s", "actual_arrival_s", "status", "observed_active", "validated_active"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**row, "route": " ".join(row["route"])} for row in result["vehicles"])
    return result
