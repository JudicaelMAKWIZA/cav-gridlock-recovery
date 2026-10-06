"""Incident aval réversible et preuve physique d'un blocage local."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from ..c3_reference import EXIT_GATES
from .road_network import CENTER_NODE, CENTER_PHASES, GATE_EDGES
from .traffic_demand import Mission, STEP_S, TrafficInputError


# Définition SUMO d'un véhicule en arrêt (halting), pas seuil de formation.
HALTING_SPEED_MPS = 0.1


def position_tolerance(*values: float) -> float:
    """Tolérance d'arrondi de deux doubles, sans seuil de déplacement physique."""
    return 2 * math.ulp(max(1.0, *(abs(value) for value in values)))


@dataclass(frozen=True)
class IncidentPlacement:
    vehicle_id: str
    edge_id: str
    lane_id: str
    position_m: float
    lane_length_m: float
    vehicle_length_m: float
    min_gap_m: float
    lookahead_m: float
    incoming_edges: tuple[str, ...]
    link_index: int


def incident_placement(network_path: Path, missions: list[Mission], vehicle_type: dict) -> IncidentPlacement:
    """Choisit la première mission vers la sortie à une voie, sans modifier sa route."""
    edge_id = GATE_EDGES[EXIT_GATES[0]]
    candidates = [m for m in missions if m.destination == edge_id and edge_id in m.route]
    if not candidates:
        raise TrafficInputError("Aucune mission normale ne rejoint la zone d'incident.")
    mission = min(candidates, key=lambda m: (m.scheduled_s, m.vehicle_id))
    root = ET.parse(network_path).getroot()
    edges = {e.get("id"): e for e in root.findall("edge")}
    target = edges.get(edge_id)
    if target is None or target.get("function", "") != "" or len(target.findall("lane")) != 1:
        raise TrafficInputError("L'incident exige une sortie normale à une seule voie.")
    lane = target.find("lane")
    length = float(vehicle_type["length"])
    gap = float(vehicle_type["minGap"])
    lane_length = float(lane.get("length"))
    # Deux empreintes véhicule + espacement placent l'arrêt dans le lien aval :
    # le véhicule entier et un suiveur tiennent avant son nez, pas dans un raccord.
    position = 2 * (length + gap)
    if (not all(math.isfinite(v) for v in (length, gap, lane_length))
            or length <= 0 or gap < 0 or not length < position < lane_length - length):
        raise TrafficInputError("La géométrie ne permet pas l'arrêt aval avec marge d'arrivée.")
    if "passenger" in lane.get("disallow", "").split() or (
            lane.get("allow") is not None and "passenger" not in lane.get("allow").split()):
        raise TrafficInputError("La voie d'incident refuse passenger.")
    crossings = [(a, b) for a, b in zip(mission.route, mission.route[1:])
                 if edges[a].get("to") == CENTER_NODE]
    if len(crossings) != 1:
        raise TrafficInputError("La mission d'incident doit traverser une fois le carrefour.")
    links = [c for c in root.findall("connection")
             if (c.get("from"), c.get("to")) == crossings[0] and c.get("tl") == CENTER_NODE]
    if len(links) != 1:
        raise TrafficInputError("Le mouvement de référence n'a pas un lien de feu unique.")
    incoming = tuple(sorted(e.get("id") for e in edges.values() if e.get("to") == CENTER_NODE))
    lookahead = sum(float(e.find("lane").get("length")) for e in edges.values() if e.find("lane") is not None)
    return IncidentPlacement(mission.vehicle_id, edge_id, lane.get("id"), position,
                             lane_length, length, gap, lookahead, incoming, int(links[0].get("linkIndex")))


def leader_chain(vehicle_id: str, observations: dict, incident_id: str) -> list[str]:
    """Suit les leaders TraCI présents jusqu'au véhicule incident, sans graphe de diagnostic."""
    chain = [vehicle_id]
    while chain[-1] != incident_id:
        row = observations.get(chain[-1])
        leader = row.get("leader_id") if row else None
        if leader is None or leader in chain or leader not in observations:
            return []
        chain.append(leader)
    return chain


def _stationary(row: dict) -> bool:
    progress = row["progress_m"]
    return (progress is not None and not row["stop_state"] & 1
            and row["speed_mps"] < HALTING_SPEED_MPS
            and abs(progress) <= position_tolerance(row["distance_m"]))


class LocalBlockage:
    """Sépare arrêt constaté, phase de service empêchée et libération externe de faisabilité."""

    def __init__(self, placement: IncidentPlacement, horizon_s: float, feasibility_release: bool = False):
        self.placement = placement
        self.horizon_s = horizon_s
        self.feasibility_release = feasibility_release
        self.state = "not_requested"
        self.t_phys: float | None = None
        self.t_form: float | None = None
        self.release_time_s: float | None = None
        self.resume_time_s: float | None = None
        self.events: list[dict] = []
        self.opportunities: list[dict] = []
        self.evidence: dict | None = None
        self.window: dict | None = None
        self.previous: dict[str, dict] = {}
        self.affected_ids: set[str] = set()
        self.last_queue: dict = {}

    def observe(self, connection, time_s: float, active_ids: set[str]) -> dict:
        """Lit les mesures après validation des missions, puis applique uniquement les événements déclarés."""
        observations = {}
        for item in sorted(active_ids):
            speed = connection.vehicle.getSpeed(item)
            position = connection.vehicle.getLanePosition(item)
            distance = connection.vehicle.getDistance(item)
            if not all(math.isfinite(v) for v in (speed, position, distance)) or speed < 0:
                raise RuntimeError("Mesure physique du véhicule non finie ou vitesse négative.")
            leader = connection.vehicle.getLeader(item, self.placement.lookahead_m)
            previous = self.previous.get(item)
            observations[item] = {
                "edge_id": connection.vehicle.getRoadID(item),
                "lane_id": connection.vehicle.getLaneID(item),
                "position_m": position, "speed_mps": speed, "distance_m": distance,
                "progress_m": distance - previous["distance_m"] if previous else None,
                "stop_state": connection.vehicle.getStopState(item),
                "leader_id": leader[0] if leader and leader[0] else None,
                "leader_gap_after_min_gap_m": leader[1] if leader and leader[0] else None,
                "next_tls": [{"id": tls, "link_index": link, "distance_m": d, "state": state}
                             for tls, link, d, state in connection.vehicle.getNextTLS(item)],
            }
        item = self.placement.vehicle_id
        if self.state == "not_requested" and item in active_ids:
            if (connection.vehicle.getLength(item) != self.placement.vehicle_length_m
                    or connection.vehicle.getMinGap(item) != self.placement.min_gap_m
                    or connection.lane.getLength(self.placement.lane_id) != self.placement.lane_length_m):
                raise RuntimeError("Le véhicule ou la voie chargés diffèrent du placement contrôlé.")
            # La durée dépasse la fin de l'expérience même si l'arrêt est atteint
            # plus tard. Aucune expiration ne peut être confondue avec une reprise.
            connection.vehicle.setStop(item, self.placement.edge_id, pos=self.placement.position_m,
                                       laneIndex=0, duration=self.horizon_s + STEP_S)
            self.events.append({"kind": "stop_requested", "time_s": time_s, "vehicle_id": item,
                                "position_m": self.placement.position_m,
                                "duration_s": self.horizon_s + STEP_S})
            self.state = "requested"
        root = observations.get(item)
        at_placement = bool(root and root["edge_id"] == self.placement.edge_id
                            and root["lane_id"] == self.placement.lane_id
                            and abs(root["position_m"] - self.placement.position_m)
                            <= position_tolerance(root["position_m"], self.placement.position_m, root["distance_m"]))
        if self.state == "requested" and root and root["stop_state"] & 1:
            if not at_placement:
                raise RuntimeError("L'arrêt physique n'est pas à la position déclarée.")
            self.state = "active"
            self.t_phys = time_s
            self.events.append({"kind": "physically_active", "time_s": time_s, "observation": dict(root)})
        if self.state == "active" and (not at_placement or not root["stop_state"] & 1
                                       or root["speed_mps"] >= HALTING_SPEED_MPS):
            self.state = "unexpected_release"
            self.events.append({"kind": "unexpected_release", "time_s": time_s, "observation": root})
            raise RuntimeError("L'incident s'est libéré sans commande externe déclarée.")
        phase = connection.trafficlight.getPhase(CENTER_NODE)
        end = connection.trafficlight.getNextSwitch(CENTER_NODE)
        start = end - CENTER_PHASES[phase][0]
        frame = self.update(time_s, observations, phase, start, end)
        if self.feasibility_release and self.t_form is not None and self.release_time_s is None:
            if self.state != "active":
                raise RuntimeError("La libération exige l'arrêt physique encore actif.")
            connection.vehicle.resume(item)
            self.release_time_s = time_s
            self.state = "release_requested"
            self.events.append({"kind": "feasibility_release", "time_s": time_s, "vehicle_id": item})
        elif self.state == "release_requested" and root and (
                root["speed_mps"] > 0 and not root["stop_state"] & 1
                and root["distance_m"] > self.previous[item]["distance_m"]):
            self.resume_time_s = time_s
            self.state = "released"
            self.events.append({"kind": "physical_resume", "time_s": time_s, "observation": dict(root)})
        self.previous = observations
        return frame

    def update(self, time_s: float, observations: dict, phase: int, start_s: float, end_s: float) -> dict:
        """Exige un même groupe immobile sur une phase verte complète, relié à l'occupation aval."""
        root_id = self.placement.vehicle_id
        root = observations.get(root_id)
        occupied = bool(self.state == "active" and root and root["stop_state"] & 1
                        and root["speed_mps"] < HALTING_SPEED_MPS
                        and root["lane_id"] == self.placement.lane_id)
        green = CENTER_PHASES[phase][1][self.placement.link_index] == "G"
        eligible = {}
        queue = {}
        for item, row in observations.items():
            if item == root_id or not _stationary(row):
                continue
            chain = leader_chain(item, observations, root_id)
            if occupied and chain and all(_stationary(observations[leader]) for leader in chain[1:-1]):
                queue[item] = chain
                # Le petit vert « g » impose de céder : il ne suffit pas
                # à exclure une attente normale due aux mouvements prioritaires.
                if all(t["state"] == "G" for t in row["next_tls"]):
                    eligible[item] = chain
        self.last_queue = {"time_s": time_s, "ids": sorted(queue),
                           "propagation_ids": sorted(item for item in queue
                                                     if observations[item]["edge_id"] in self.placement.incoming_edges)}
        key = (phase, start_s, end_s)
        if self.window is not None and self.window["key"] != key:
            self.window = None
        if self.window is None and occupied and green and abs(time_s - STEP_S - start_s) <= 1e-7:
            self.window = {"key": key, "start_s": start_s, "end_s": end_s,
                           "cohort": {item: {"start_distance_m": observations[item]["distance_m"]
                                           - observations[item]["progress_m"],
                                           "start_chain": chain}
                                      for item, chain in eligible.items()},
                           "samples": 0, "last_time_s": start_s, "propagated_all_samples": True}
        completed = None
        if self.window is not None:
            window = self.window
            if abs(time_s - window["last_time_s"] - STEP_S) > 1e-7:
                raise RuntimeError("La phase de preuve contient une mesure manquante ou répétée.")
            window["last_time_s"] = time_s
            window["samples"] += 1
            window["cohort"] = {item: row for item, row in window["cohort"].items()
                                if occupied and green and item in eligible
                                and abs(observations[item]["distance_m"] - row["start_distance_m"])
                                <= position_tolerance(observations[item]["distance_m"], row["start_distance_m"])}
            propagated = any(observations[item]["edge_id"] in self.placement.incoming_edges
                             for item in window["cohort"])
            window["propagated_all_samples"] &= propagated
            if abs(time_s - end_s) <= 1e-7:
                cohort = window["cohort"]
                complete = window["samples"] * STEP_S == end_s - start_s
                formed = complete and len(cohort) >= 2 and window["propagated_all_samples"]
                completed = {"start_s": start_s, "end_s": end_s, "phase_index": phase,
                             "samples": window["samples"], "complete": complete,
                             "affected_ids": sorted(cohort), "propagation_all_samples": window["propagated_all_samples"],
                             "formed": formed,
                             "vehicles": {item: {**row,
                                                "end_distance_m": observations[item]["distance_m"],
                                                "total_progress_m": observations[item]["distance_m"] - row["start_distance_m"],
                                                "end_chain": eligible[item]}
                                          for item, row in cohort.items()}}
                self.opportunities.append(completed)
                if formed and self.t_form is None:
                    self.t_form = time_s
                    self.evidence = completed
                self.window = None
        self.affected_ids.update(eligible)
        return {"time_s": time_s, "incident_state": self.state, "phase_index": phase,
                "phase_start_s": start_s, "phase_end_s": end_s, "occupied": occupied,
                "affected_ids": sorted(eligible), "stationary_queue": self.last_queue,
                "vehicles": observations, "completed_opportunity": completed}

    def summary(self) -> dict:
        """Expose la preuve et les événements, sans calculer de métrique de récupération."""
        return {"id": "downstream_stop", "cause": "reversible_vehicle_immobilization",
                "state": self.state, "placement": self.placement.__dict__,
                "t_phys": self.t_phys, "t_form": self.t_form, "formed": self.t_form is not None,
                "affected_ids": sorted(self.affected_ids), "evidence": self.evidence,
                "stationary_queue_at_end": self.last_queue,
                "service_opportunities": self.opportunities, "events": self.events,
                "feasibility_release": self.feasibility_release,
                "feasibility_release_time_s": self.release_time_s, "physical_resume_time_s": self.resume_time_s,
                "claim": "local_non_cyclic_blockage_not_network_gridlock",
                "timestamp_convention": "first_observed_TraCI_sample_step_0_5_s",
                "criteria": {"plurality": "at_least_two_non_incident_missions",
                             "persistence": "same_stationary_cohort_over_one_complete_green_phase",
                             "cause": "observed_leader_chain_to_occupied_downstream_lane",
                             "chain": "every_intermediate_leader_stationary_without_own_scheduled_stop",
                             "propagation": "cohort_present_on_a_C3_incoming_edge_at_every_sample",
                             "service": "all_upcoming_tls_protected_green_at_every_sample",
                             "halt_speed_mps": HALTING_SPEED_MPS,
                             "progress_tolerance": "two_double_precision_ulps"}}
