"""Vérifie un trajet de test avec SUMO et TraCI."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from .sumo_process import close_sumo


class SmokeInputError(ValueError):
    """Signale un trajet de test ou un horizon invalide."""


def _read_mission(fixture_dir: Path) -> dict:
    """Lit la route et la destination prévues avant le départ."""
    try:
        config = ET.parse(fixture_dir / "simulation.sumocfg").getroot()
        routes = ET.parse(fixture_dir / "traffic.rou.xml").getroot()
        vehicles = routes.findall("vehicle")
        if len(vehicles) != 1:
            raise SmokeInputError("La fixture doit contenir exactement un véhicule.")
        vehicle = vehicles[0]
        route = next(row for row in routes.findall("route") if row.get("id") == vehicle.get("route"))
        edges = route.attrib["edges"].split()
        step = float(config.find("time/step-length").attrib["value"])
        depart = float(vehicle.attrib["depart"])
        if len(edges) < 2 or not math.isfinite(step) or step <= 0 or not math.isfinite(depart) or depart < 0:
            raise SmokeInputError("Route, départ ou pas de simulation invalide.")
        if "arrivePos" in vehicle.attrib:
            raise SmokeInputError("Ce contrôle utilise l'arrivée normale en fin d'arête, sans arrivePos.")
        return {"vehicle_id": vehicle.attrib["id"], "route_id": route.attrib["id"],
                "route": edges, "destination": edges[-1], "depart_s": depart, "step_s": step}
    except (OSError, ET.ParseError, KeyError, AttributeError, StopIteration, ValueError) as error:
        raise SmokeInputError(f"Fixture invalide : {error}") from error


def _observe_trip(connection, mission: dict, result: dict) -> None:
    """Suit les événements et les lectures jusqu'à l'arrivée ou à l'horizon."""
    vehicle_id = mission["vehicle_id"]
    if list(connection.route.getEdges(mission["route_id"])) != mission["route"]:
        raise RuntimeError("La route chargée par SUMO diffère de la mission assignée.")
    for _ in range(result["max_steps"]):
        connection.simulationStep()
        result["steps"] += 1
        time_s = connection.simulation.getTime()
        result["simulation_time_s"] = time_s
        for name, getter in (
            ("departures", connection.simulation.getDepartedIDList),
            ("arrivals", connection.simulation.getArrivedIDList),
            ("teleport_starts", connection.simulation.getStartingTeleportIDList),
            ("teleport_ends", connection.simulation.getEndingTeleportIDList),
        ):
            result[name].extend({"vehicle_id": item, "time_s": time_s} for item in getter())
        if result["teleport_starts"] or result["teleport_ends"]:
            raise RuntimeError("Téléportation détectée : l'arrivée ne peut pas valider le trajet.")
        if not math.isfinite(time_s) or time_s > result["horizon_s"] + 1e-9:
            raise RuntimeError("Temps simulé invalide ou supérieur à l'horizon.")

        active = connection.vehicle.getIDList()
        if vehicle_id in active:
            position = list(connection.vehicle.getPosition(vehicle_id))
            route = list(connection.vehicle.getRoute(vehicle_id))
            road = connection.vehicle.getRoadID(vehicle_id)
            route_index = connection.vehicle.getRouteIndex(vehicle_id)
            if len(position) != 2 or not all(math.isfinite(value) for value in position):
                raise RuntimeError("Position du véhicule non finie.")
            if route != mission["route"]:
                raise RuntimeError("La route observée diffère de la mission assignée.")
            result["position_reads"] += 1
            observation = {"time_s": time_s, "position_m": position,
                           "route": route, "road_id": road, "route_index": route_index}
            if result["first_observation"] is None:
                result["first_observation"] = observation
            result["last_observation"] = observation

        arrived = [event for event in result["arrivals"] if event["vehicle_id"] == vehicle_id]
        departed = [event for event in result["departures"] if event["vehicle_id"] == vehicle_id]
        if arrived:
            if len(departed) != 1 or len(arrived) != 1 or departed[0]["time_s"] >= arrived[0]["time_s"]:
                raise RuntimeError("Arrivée sans départ unique observé auparavant.")
            last = result["last_observation"]
            # On vérifie la destination pendant que le véhicule est encore présent,
            # car SUMO le retire à l'arrivée.
            if (
                last is None or last["road_id"] != mission["destination"]
                or last["route_index"] != len(mission["route"]) - 1
                or last["time_s"] >= arrived[0]["time_s"] or vehicle_id in active
            ):
                raise RuntimeError("L'arrivée n'est pas corroborée sur l'arête de destination.")
            result["destination_confirmed"] = mission["destination"]
            result["arrival_evidence"] = "last_observed_destination_edge_and_sumo_arrival_event"
            return
        if result["position_reads"] and vehicle_id not in active:
            raise RuntimeError("Disparition du véhicule sans événement d'arrivée.")
    raise RuntimeError("Horizon atteint sans arrivée normale du véhicule attendu.")


def run_sumo_smoke(
    fixture_dir: str | Path,
    *,
    sumo_binary: str = "sumo",
    horizon_s: float = 60.0,
) -> dict:
    """Vérifie un trajet sans assistance, dans une limite en secondes simulées.

    L'arrivée exige un départ, des positions et une route vérifiés jusqu'à la
    destination, puis l'événement d'arrivée SUMO. On rend aussi un bilan
    en cas d'erreur et on tente toujours de fermer le processus.
    """
    fixture = Path(fixture_dir).resolve()
    mission = _read_mission(fixture)
    if not math.isfinite(horizon_s) or horizon_s < mission["step_s"]:
        raise SmokeInputError("L'horizon doit être fini et couvrir au moins un pas.")
    result = {
        "status": "failed", "reason": None, "sumo_version": None, "traci_version": None,
        "mission": mission, "horizon_s": horizon_s,
        "max_steps": math.floor(horizon_s / mission["step_s"] + 1e-9),
        "steps": 0, "simulation_time_s": 0.0, "connected": False,
        "departures": [], "arrivals": [], "teleport_starts": [], "teleport_ends": [],
        "position_reads": 0, "first_observation": None, "last_observation": None,
        "destination_confirmed": None, "arrival_evidence": None,
        "connection_closed": False, "process_stopped": False, "process_returncode": None,
        "forced_process_stop": False, "cleanup_errors": [], "assistance_commands": [],
        "time_to_teleport_s": -1, "seed": 0,
    }
    process = connection = None
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as log:
        try:
            binary = shutil.which(sumo_binary)
            if binary is None:
                raise RuntimeError("SUMO introuvable : activer l'environnement contenant le binaire sumo.")
            try:
                traci = importlib.import_module("traci")
            except ImportError as error:
                raise RuntimeError("TraCI introuvable dans cet environnement Python.") from error
            result["traci_version"] = getattr(traci, "__version__", "non déclarée")
            # Le port peut être repris avant le lancement. Même sans connexion,
            # on tente d'arrêter SUMO.
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            command = [binary, "-c", str(fixture / "simulation.sumocfg"),
                       "--remote-port", str(port), "--time-to-teleport", "-1",
                       "--seed", "0", "--no-step-log", "true", "--duration-log.disable", "true"]
            process = subprocess.Popen(command, stdout=log, stderr=log)
            connection = traci.connect(port=port, host="127.0.0.1", proc=process,
                                       numRetries=20, waitBetweenRetries=0.1)
            result["connected"] = True
            result["sumo_version"] = connection.getVersion()[1]
            _observe_trip(connection, mission, result)
            result["status"] = "passed"
        except Exception as error:
            result["reason"] = str(error)
        finally:
            try:
                close_sumo(connection, process, result)
            except Exception as error:
                result["cleanup_errors"].append(f"Arrêt du processus impossible : {error}")
        if result["cleanup_errors"] or result["forced_process_stop"] or (
            process is not None and result["process_returncode"] != 0
        ):
            result["status"] = "failed"
            result["reason"] = result["reason"] or "La fermeture normale de SUMO/TraCI a échoué."
        if result["status"] == "failed":
            log.seek(0)
            result["sumo_diagnostic"] = log.read()[-4000:]
    return result
