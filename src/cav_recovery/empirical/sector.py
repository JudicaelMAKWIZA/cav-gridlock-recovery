"""Géométrie métrique limitée au secteur CGR-E02 et décisions d'association."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


EARTH_RADIUS_M = 6_371_008.8
BranchRole = Literal["entry", "exit"]
AssociationStatus = Literal["accepted", "ambiguous", "outside", "insufficient"]


class SectorConfigurationError(ValueError):
    """Signale une géométrie ou une configuration de secteur inexploitable."""


@dataclass(frozen=True)
class Point:
    """Point dans le repère métrique local, en mètres."""

    x_m: float
    y_m: float


@dataclass(frozen=True)
class TrackObservation:
    """Observation CGR-E01 minimale nécessaire au profilage spatial."""

    source_line: int
    group_index: int
    track_id: str
    category: str
    lat: float
    lon: float
    time_s: float
    break_before: bool = False


@dataclass(frozen=True)
class Branch:
    """Branche radiale orientée autour du centre du secteur."""

    identifier: str
    role: BranchRole
    gate_center: Point
    outward_unit: Point


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
    branches: tuple[Branch, ...]
    gates: tuple[Gate, ...]
    source_document: dict


@dataclass(frozen=True)
class AssociationParameters:
    """Seuils métriques et directionnels, indépendants des valeurs de présélection."""

    max_distance_m: float
    max_branch_extent_m: float
    max_heading_difference_deg: float
    ambiguity_margin_m: float
    minimum_displacement_m: float
    gate_hysteresis_m: float
    gate_rearm_distance_m: float
    max_time_gap_s: float
    max_space_gap_m: float

    def validate(self) -> None:
        values = self.__dict__
        if any(not math.isfinite(value) or value <= 0 for value in values.values()):
            raise SectorConfigurationError("Tous les seuils CGR-E02 doivent être finis et strictement positifs.")
        if self.gate_rearm_distance_m <= self.gate_hysteresis_m:
            raise SectorConfigurationError("La distance de réarmement doit dépasser l'hystérésis de porte.")
        if self.max_heading_difference_deg >= 180:
            raise SectorConfigurationError("La tolérance directionnelle doit être inférieure à 180 degrés.")


@dataclass(frozen=True)
class Association:
    """Résultat explicite d'une association à une branche."""

    status: AssociationStatus
    branch_id: str | None
    distance_m: float | None
    reason: str


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
    """Projection équirectangulaire locale centrée sur C2.

    Cette approximation tangentielle est réservée aux distances de quelques dizaines
    de mètres du secteur. Aucun seuil métrique n'est appliqué aux degrés source.
    """

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


def _dot(first: Point, second: Point) -> float:
    return first.x_m * second.x_m + first.y_m * second.y_m


def _cross(first: Point, second: Point) -> float:
    return first.x_m * second.y_m - first.y_m * second.x_m


def _unit(vector: Point) -> Point:
    length = _norm(vector)
    if length == 0:
        raise SectorConfigurationError("Une direction de branche ne peut pas être nulle.")
    return Point(vector.x_m / length, vector.y_m / length)


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
    branches: list[Branch] = []
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
        role = branch_contract[identifier]
        gates.append(Gate(identifier, role, endpoint_points[0], endpoint_points[1], gate_center, 1 if center_side > 0 else -1))
        branches.append(Branch(identifier, role, gate_center, _unit(gate_center)))
    if set(branch_contract) != {gate.identifier for gate in gates}:
        raise SectorConfigurationError("Chaque branche doit disposer d'une porte unique.")
    if not any(branch.role == "entry" for branch in branches) or not any(branch.role == "exit" for branch in branches):
        raise SectorConfigurationError("Le secteur doit comporter au moins une entrée et une sortie.")
    return Sector(str(sector_data.get("id", "")), center_lat, center_lon, tuple(branches), tuple(gates), document)


def is_rupture(previous: TrackObservation, current: TrackObservation, projection: LocalMetricProjection, parameters: AssociationParameters) -> bool:
    """Détecte une lacune temporelle, un saut spatial ou un temps non croissant."""
    if current.break_before:
        return True
    delta_t = current.time_s - previous.time_s
    distance = _norm(_subtract(projection.project(current.lat, current.lon), projection.project(previous.lat, previous.lon)))
    return delta_t <= 0 or delta_t > parameters.max_time_gap_s or distance > parameters.max_space_gap_m


def associate_segment(previous: TrackObservation, current: TrackObservation, sector: Sector, parameters: AssociationParameters) -> Association:
    """Associe un déplacement à une branche par proximité, direction et marge d'ambiguïté."""
    projection = LocalMetricProjection(sector.center_lat, sector.center_lon)
    start = projection.project(previous.lat, previous.lon)
    end = projection.project(current.lat, current.lon)
    displacement = _subtract(end, start)
    length = _norm(displacement)
    if length < parameters.minimum_displacement_m:
        return Association("insufficient", None, None, "déplacement trop faible pour estimer la direction")
    heading = _unit(displacement)
    midpoint = Point((start.x_m + end.x_m) / 2, (start.y_m + end.y_m) / 2)
    candidates: list[tuple[float, Branch]] = []
    for branch in sector.branches:
        along = _dot(midpoint, branch.outward_unit)
        if along < 0 or along > parameters.max_branch_extent_m:
            continue
        distance = abs(_cross(branch.outward_unit, midpoint))
        expected_heading = Point(-branch.outward_unit.x_m, -branch.outward_unit.y_m) if branch.role == "entry" else branch.outward_unit
        cosine = max(-1.0, min(1.0, _dot(heading, expected_heading)))
        angle = math.degrees(math.acos(cosine))
        if distance <= parameters.max_distance_m and angle <= parameters.max_heading_difference_deg:
            candidates.append((distance, branch))
    if not candidates:
        return Association("outside", None, None, "aucune branche ne satisfait distance et direction")
    candidates.sort(key=lambda item: (item[0], item[1].identifier))
    if len(candidates) > 1 and candidates[1][0] - candidates[0][0] <= parameters.ambiguity_margin_m:
        return Association("ambiguous", None, candidates[0][0], "deux branches restent indissociables dans la marge configurée")
    return Association("accepted", candidates[0][1].identifier, candidates[0][0], "association distance-direction acceptée")


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


def detect_crossings(observations: list[TrackObservation], sector: Sector, parameters: AssociationParameters) -> tuple[list[Crossing], int]:
    """Détecte les franchissements attendus sans relier les ruptures.

    Le compteur retourné indique les ruptures qui ont empêché toute interpolation.
    """
    if not observations:
        return [], 0
    projection = LocalMetricProjection(sector.center_lat, sector.center_lon)
    states: dict[str, tuple[int, TrackObservation] | None] = {gate.identifier: None for gate in sector.gates}
    armed: dict[str, bool] = {gate.identifier: True for gate in sector.gates}
    events: list[Crossing] = []
    ruptures = 0
    continuity_id = 0
    previous = observations[0]
    first_point = projection.project(previous.lat, previous.lon)
    for gate in sector.gates:
        distance = _inside_distance(first_point, gate)
        side = 1 if distance >= parameters.gate_hysteresis_m else -1 if distance <= -parameters.gate_hysteresis_m else 0
        if side:
            states[gate.identifier] = (side, previous)
    for current in observations[1:]:
        if is_rupture(previous, current, projection, parameters):
            ruptures += 1
            continuity_id += 1
            states = {gate.identifier: None for gate in sector.gates}
            armed = {gate.identifier: True for gate in sector.gates}
            current_point = projection.project(current.lat, current.lon)
            for gate in sector.gates:
                distance = _inside_distance(current_point, gate)
                side = 1 if distance >= parameters.gate_hysteresis_m else -1 if distance <= -parameters.gate_hysteresis_m else 0
                if side:
                    states[gate.identifier] = (side, current)
            previous = current
            continue
        current_point = projection.project(current.lat, current.lon)
        for gate in sector.gates:
            inside_distance = _inside_distance(current_point, gate)
            side = 1 if inside_distance >= parameters.gate_hysteresis_m else -1 if inside_distance <= -parameters.gate_hysteresis_m else 0
            if side == 0:
                continue
            prior_state = states[gate.identifier]
            expected_start = -1 if gate.role == "entry" else 1
            expected_end = -expected_start
            if not armed[gate.identifier] and side == expected_start and abs(inside_distance) >= parameters.gate_rearm_distance_m:
                armed[gate.identifier] = True
            if prior_state is not None and prior_state[0] == expected_start and side == expected_end and armed[gate.identifier]:
                anchor = prior_state[1]
                anchor_point = projection.project(anchor.lat, anchor.lon)
                # La bande d'hystérésis ne doit pas prolonger implicitement la
                # fenêtre temporelle ou spatiale autorisée pour l'interpolation.
                fraction = None if is_rupture(anchor, current, projection, parameters) else _intersection_fraction(anchor_point, current_point, gate)
                if fraction is not None:
                    estimated = anchor.time_s + fraction * (current.time_s - anchor.time_s)
                    events.append(Crossing(anchor.source_line, anchor.track_id, anchor.category, gate.identifier, gate.role, continuity_id, anchor.group_index, current.group_index, estimated, anchor.time_s, current.time_s))
                    armed[gate.identifier] = False
            states[gate.identifier] = (side, current)
        previous = current
    return events, ruptures
