"""Suit les missions et vérifie les arrivées à destination."""

import math
from pathlib import Path
from typing import Callable
import xml.etree.ElementTree as ET

from .traffic_demand import Mission

STEP_S = 0.5


class TrafficLedger:
    """Suit les départs et arrivées sans changer les destinations.

    On distingue le dernier état observé du dernier état validé,
    pour expliquer une erreur sans perdre le dernier pas accepté.
    """

    def __init__(self, missions: list[Mission]):
        self.missions = {mission.vehicle_id: mission for mission in missions}
        if len(self.missions) != len(missions):
            raise ValueError("Les missions doivent avoir des identifiants uniques.")
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
        self.readings: dict | None = None

    def observe(self, connection, vehicle_readings: dict | Callable | None = None) -> dict:
        """Garde ce qui a été lu, même si un contrôle échoue."""
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
            self._validate_observation(connection, observation, vehicle_readings)
        except Exception as error:
            self.failure_observation = {**observation, "reason": str(error)}
            raise
        # On ne valide la présence des véhicules qu'après tous les contrôles.
        self.validated_active = self.observed_active.copy()
        self.last_validated_state = {"time_s": time_s, "active_ids": sorted(self.validated_active),
                                     "departed_ids": sorted(self.departures), "arrived_ids": sorted(self.arrivals)}
        self.failure_observation = None
        return self.snapshot()

    def _validate_observation(self, connection, observation: dict, vehicle_readings: dict | Callable | None) -> None:
        """Vérifie le pas sans effacer les événements déjà lus.

        Un départ enregistré reste visible si un contrôle échoue ; le nouvel
        état actif n'est validé qu'une fois tous les contrôles passés.
        """
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
        # On enregistre d'abord les départs : une erreur de lecture ensuite
        # ne doit pas les faire passer pour des véhicules encore en attente.
        self.readings = vehicle_readings(connection) if callable(vehicle_readings) else vehicle_readings
        for vehicle_id in sorted(active):
            mission = self.missions[vehicle_id]
            reading = self.readings[vehicle_id] if self.readings is not None else None
            if vehicle_id not in self.departures or vehicle_id in self.arrivals:
                raise RuntimeError("Présence active incompatible avec les événements.")
            route = list(reading["route"] if reading is not None else connection.vehicle.getRoute(vehicle_id))
            observation["vehicles"][vehicle_id] = {"route": route}
            if tuple(route) != mission.route:
                raise RuntimeError("Route ou destination modifiée pendant l'exécution.")
            position = list(reading["position"] if reading is not None else connection.vehicle.getPosition(vehicle_id))
            if len(position) != 2 or not all(math.isfinite(value) for value in position):
                raise RuntimeError("Position non finie.")
            self.last[vehicle_id] = {"road_id": reading["road_id"] if reading is not None else connection.vehicle.getRoadID(vehicle_id),
                                     "route_index": reading["route_index"] if reading is not None else connection.vehicle.getRouteIndex(vehicle_id),
                                     "time_s": time_s, "position_m": position,
                                     "shape": reading["shape"] if reading is not None else connection.vehicle.getShapeClass(vehicle_id)}
        for vehicle_id in arrived:
            mission = self.missions[vehicle_id]
            last = self.last.get(vehicle_id)
            # On doit voir le véhicule à destination avant que SUMO le retire ;
            # tripinfo confirme ensuite son arrivée.
            if (vehicle_id not in self.departures or vehicle_id in self.arrivals or vehicle_id in active
                    or last is None or last["road_id"] != mission.destination
                    or last["route_index"] != len(mission.route) - 1 or last["time_s"] >= time_s):
                raise RuntimeError("Arrivée non corroborée à la destination assignée.")
            self.arrivals[vehicle_id] = time_s
        expected_active = set(self.departures) - set(self.arrivals)
        if active != expected_active:
            raise RuntimeError("Disparition sans arrivée ou bilan des véhicules incohérent.")

    def snapshot(self) -> dict:
        """Résume les véhicules observés et les erreurs de suivi.

        Le dernier état validé ne s'ajoute pas aux comptes courants.
        Un véhicule parti, sans arrivée ni présence, est signalé comme disparu,
        jamais comme encore en attente de départ.
        """
        pending = set(self.missions) - set(self.departures) - self.observed_active
        # SUMO traite [t - pas, t) lors du dernier pas : un départ à t reste futur.
        future = {item for item in pending if self.missions[item].scheduled_s >= self.time_s}
        delays = [time - self.missions[item].scheduled_s for item, time in self.departures.items()]
        return {
            "simulation_time_s": self.time_s,
            "scheduled": len(self.missions),
            "future": len(future),
            "delayed_not_inserted": len(pending - future),
            "pending": len(pending),
            "departed": len(self.departures),
            "active": len(self.observed_active),
            "observed_active": len(self.observed_active),
            "validated_active": len(self.validated_active),
            "observed_time_s": self.observed_time_s,
            "missing_without_arrival": len(set(self.departures) - set(self.arrivals) - self.observed_active),
            "active_unvalidated": len(self.observed_active) if self.failure_observation is not None else 0,
            "arrived": len(self.arrivals),
            "teleport_starts": len(self.teleport_starts),
            "teleport_ends": len(self.teleport_ends),
            "max_insertion_delay_s": max(delays, default=None),
            "mean_insertion_delay_s": sum(delays) / len(delays) if delays else None,
        }

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
    """Vérifie dans tripinfo les arrivées à destination, sans véhicule supprimé."""
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
