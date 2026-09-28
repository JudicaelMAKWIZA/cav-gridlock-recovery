"""Géométrie métrique et franchissements de portes virtuelles pour CGR-E02."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


EARTH_RADIUS_M = 6_371_008.8
BranchRole = Literal["entry", "exit"]


class SectorConfigurationError(ValueError):
    """Signale une géométrie ou une configuration de secteur inexploitable."""


@dataclass(frozen=True)
class Point:
    """Point dans le repère métrique local, en mètres."""

    x_m: float
    y_m: float


@dataclass(frozen=True)
class TrackObservation:
    """Observation CGR-E01 minimale nécessaire à la détection des portes."""

    source_line: int
    group_index: int
    track_id: str
    category: str
    lat: float
    lon: float
    time_s: float
    break_before: bool = False


@dataclass(frozen=True)
class Gate:
    """Porte finie dont le côté intérieur est défini par le centre du secteur."""

    identifier: str
    role: BranchRole
    first: Point
    second: Point
    center: Point
    inside_sign: int


@dataclass(frozen=True)
class Sector:
    """Secteur projeté et provenance géométrique suffisante pour CGR-E02."""

    identifier: str
    center_lat: float
    center_lon: float
    gates: tuple[Gate, ...]
    source_document: dict


@dataclass(frozen=True)
class CrossingParameters:
    """Seuils du détecteur de portes, figés indépendamment du jeu de test."""

    deduplication_s: float
    max_time_gap_s: float
    max_space_gap_m: float

    def validate(self) -> None:
        if any(not math.isfinite(value) or value <= 0 for value in self.__dict__.values()):
            raise SectorConfigurationError("Tous les seuils CGR-E02 doivent être finis et strictement positifs.")


@dataclass(frozen=True)
class Crossing:
    """Franchissement observé entre deux observations encadrantes."""

    source_line: int
    track_id: str
    category: str
    gate_id: str
    role: BranchRole
    continuity_id: int
    from_group_index: int
    to_group_index: int
    estimated_time_s: float
    interval_start_s: float
    interval_end_s: float


class LocalMetricProjection:
    """Projection équirectangulaire locale réservée au voisinage du secteur."""

    def __init__(self, center_lat: float, center_lon: float) -> None:
        self.center_lat = center_lat
        self.center_lon = center_lon
        self._cos_lat = math.cos(math.radians(center_lat))

    def project(self, lat: float, lon: float) -> Point:
        return Point(
            math.radians(lon - self.center_lon) * EARTH_RADIUS_M * self._cos_lat,
            math.radians(lat - self.center_lat) * EARTH_RADIUS_M,
        )


def _norm(vector: Point) -> float:
    return math.hypot(vector.x_m, vector.y_m)


def _subtract(first: Point, second: Point) -> Point:
    return Point(first.x_m - second.x_m, first.y_m - second.y_m)


def _cross(first: Point, second: Point) -> float:
    return first.x_m * second.y_m - first.y_m * second.x_m


def _mapping(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise SectorConfigurationError(f"{label} doit être un objet JSON.")
    return value


def load_sector(path: str | Path) -> Sector:
    """Charge et valide le seed figé sans lui ajouter de seuil implicite."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != "CGR-E02-sector-seed-1":
        raise SectorConfigurationError("Schéma de secteur CGR-E02 non pris en charge.")
    geometry = _mapping(document.get("geometry"), "geometry")
    if geometry.get("crs_source") != "EPSG:4326":
        raise SectorConfigurationError("Le seed doit déclarer les coordonnées source EPSG:4326.")
    sector_data = _mapping(document.get("sector"), "sector")
    center_data = _mapping(sector_data.get("center"), "sector.center")
    try:
        center_lat = float(center_data["lat"])
        center_lon = float(center_data["lon"])
    except (KeyError, TypeError, ValueError) as error:
        raise SectorConfigurationError("Centre du secteur absent ou invalide.") from error
    projection = LocalMetricProjection(center_lat, center_lon)
    center = Point(0.0, 0.0)

    branch_rows = document.get("branches")
    gate_rows = document.get("gates")
    if not isinstance(branch_rows, list) or not isinstance(gate_rows, list):
        raise SectorConfigurationError("Les listes branches et gates sont obligatoires.")
    branch_contract: dict[str, BranchRole] = {}
    for row in branch_rows:
        data = _mapping(row, "branche")
        identifier = str(data.get("id", ""))
        role = data.get("role")
        if not identifier or role not in {"entry", "exit"} or identifier in branch_contract:
            raise SectorConfigurationError("Identifiant ou rôle de branche invalide ou dupliqué.")
        branch_contract[identifier] = role

    gates: list[Gate] = []
    for row in gate_rows:
        data = _mapping(row, "porte")
        identifier = str(data.get("id", ""))
        if identifier not in branch_contract:
            raise SectorConfigurationError(f"La porte {identifier!r} ne correspond à aucune branche.")
        endpoints = data.get("endpoints")
        center_geo = _mapping(data.get("center"), f"porte {identifier}.center")
        if not isinstance(endpoints, list) or len(endpoints) != 2:
            raise SectorConfigurationError(f"La porte {identifier} doit avoir exactement deux extrémités.")
        try:
            endpoint_points = [projection.project(float(point["lat"]), float(point["lon"])) for point in endpoints]
            gate_center = projection.project(float(center_geo["lat"]), float(center_geo["lon"]))
            declared_length = float(data["length_m"])
        except (KeyError, TypeError, ValueError) as error:
            raise SectorConfigurationError(f"Coordonnées invalides pour la porte {identifier}.") from error
        gate_vector = _subtract(endpoint_points[1], endpoint_points[0])
        measured_length = _norm(gate_vector)
        if measured_length < 1.0 or abs(measured_length - declared_length) > 1.0:
            raise SectorConfigurationError(f"La porte {identifier} est dégénérée ou sa longueur est incohérente.")
        center_side = _cross(gate_vector, _subtract(center, endpoint_points[0]))
        if abs(center_side) < 1e-9:
            raise SectorConfigurationError(f"Le côté intérieur de la porte {identifier} est indéterminé.")
        gates.append(
            Gate(
                identifier,
                branch_contract[identifier],
                endpoint_points[0],
                endpoint_points[1],
                gate_center,
                1 if center_side > 0 else -1,
            )
        )
    if set(branch_contract) != {gate.identifier for gate in gates}:
        raise SectorConfigurationError("Chaque branche doit disposer d'une porte unique.")
    if not any(gate.role == "entry" for gate in gates) or not any(gate.role == "exit" for gate in gates):
        raise SectorConfigurationError("Le secteur doit comporter au moins une entrée et une sortie.")
    return Sector(str(sector_data.get("id", "")), center_lat, center_lon, tuple(gates), document)


def is_rupture(
    previous: TrackObservation,
    current: TrackObservation,
    projection: LocalMetricProjection,
    parameters: CrossingParameters,
) -> bool:
    """Détecte une lacune temporelle, un saut spatial ou un temps non croissant."""
    if current.break_before:
        return True
    delta_t = current.time_s - previous.time_s
    distance = _norm(_subtract(projection.project(current.lat, current.lon), projection.project(previous.lat, previous.lon)))
    return delta_t <= 0 or delta_t > parameters.max_time_gap_s or distance > parameters.max_space_gap_m


def _inside_distance(point: Point, gate: Gate) -> float:
    vector = _subtract(gate.second, gate.first)
    return gate.inside_sign * _cross(vector, _subtract(point, gate.first)) / _norm(vector)


def _intersection_fraction(start: Point, end: Point, gate: Gate) -> float | None:
    movement = _subtract(end, start)
    gate_vector = _subtract(gate.second, gate.first)
    denominator = _cross(movement, gate_vector)
    if abs(denominator) < 1e-12:
        return None
    offset = _subtract(gate.first, start)
    movement_fraction = _cross(offset, gate_vector) / denominator
    gate_fraction = _cross(offset, movement) / denominator
    if 0 <= movement_fraction <= 1 and 0 <= gate_fraction <= 1:
        return movement_fraction
    return None


def detect_crossings(
    observations: list[TrackObservation],
    sector: Sector,
    parameters: CrossingParameters,
) -> tuple[list[Crossing], int]:
    """Détecte les franchissements orientés de portes entre points consécutifs.

    La déduplication absorbe les oscillations très brèves autour d'une porte. Une
    rupture remet la continuité et la déduplication à zéro : aucune interpolation
    n'est jamais effectuée de part et d'autre de cette rupture.
    """
    parameters.validate()
    if not observations:
        return [], 0
    projection = LocalMetricProjection(sector.center_lat, sector.center_lon)
    events: list[Crossing] = []
    last_event_s: dict[str, float] = {}
    ruptures = 0
    continuity_id = 0
    previous = observations[0]
    for current in observations[1:]:
        if is_rupture(previous, current, projection, parameters):
            ruptures += 1
            continuity_id += 1
            last_event_s.clear()
            previous = current
            continue
        start = projection.project(previous.lat, previous.lon)
        end = projection.project(current.lat, current.lon)
        for gate in sector.gates:
            before = _inside_distance(start, gate)
            after = _inside_distance(end, gate)
            crosses_in_expected_direction = before < 0 <= after if gate.role == "entry" else before > 0 >= after
            if not crosses_in_expected_direction:
                continue
            fraction = _intersection_fraction(start, end, gate)
            if fraction is None:
                continue
            estimated = previous.time_s + fraction * (current.time_s - previous.time_s)
            if estimated - last_event_s.get(gate.identifier, -math.inf) < parameters.deduplication_s:
                continue
            events.append(
                Crossing(
                    previous.source_line,
                    previous.track_id,
                    previous.category,
                    gate.identifier,
                    gate.role,
                    continuity_id,
                    previous.group_index,
                    current.group_index,
                    estimated,
                    previous.time_s,
                    current.time_s,
                )
            )
            last_event_s[gate.identifier] = estimated
        previous = current
    return events, ruptures
