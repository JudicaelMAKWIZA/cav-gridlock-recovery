"""Profils de passages et de mouvements à partir des observations qualifiées."""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .sector import (
    CrossingParameters,
    Crossing,
    Sector,
    SectorConfigurationError,
    TrackObservation,
    detect_crossings,
    load_sector,
)


SCHEMA_VERSION = "CGR-E02-1"
EXPECTED_E01_SCHEMA = "CGR-E01-1"
E01_TRAJECTORY_FIELDS = {"source_line", "track_id", "type", "traveled_d_m", "avg_speed_kmh", "avg_speed_mps", "structural_status", "observation_count", "diagnostic_count"}
E01_OBSERVATION_FIELDS = {"source_line", "group_index", "track_id", "type", "lat", "lon", "speed_kmh", "speed_mps", "lon_acc_mps2", "lat_acc_mps2", "time_s", "all_numeric_finite", "coordinate_in_range"}
E01_ISSUE_FIELDS = {"source_line", "group_index", "field", "code", "severity", "source_value", "message"}
OUTPUTS = (
    "manifest.json",
    "sector.geojson",
    "sector_config.json",
    "crossings.csv",
    "partial_routes.csv",
    "flow_profile.csv",
    "movement_profile.csv",
    "quality_summary.json",
    "validation_reference.csv",
    "profile_report.md",
)


class ProfileInputError(ValueError):
    """Signale des données d'entrée incompatibles avec le profil de trafic."""


@dataclass(frozen=True)
class CoverageInterval:
    """Intervalle d'exposition d'une porte, en secondes, début inclus et fin exclue."""

    start_s: float
    end_s: float


CrossingKey = tuple[int, int, str, str, int, int]


@dataclass(frozen=True)
class Visit:
    """Visite reconstruite à partir des franchissements d'une même continuité.

    Une visite classifiable possède une entrée et une sortie. Une entrée absente
    donne ``censored_entry`` ; une sortie absente donne ``censored_exit``.
    Ces censures ne permettent pas d'attribuer un mouvement complet.
    """

    visit_id: str
    source_line: int
    track_id: str
    category: str
    continuity_id: int
    entry_gate: str | None
    exit_gate: str | None
    entry_time_s: float | None
    exit_time_s: float | None
    status: str
    route: tuple[str, ...]
    reason: str
    entry_crossing_key: CrossingKey | None = None
    exit_crossing_key: CrossingKey | None = None


@dataclass(frozen=True)
class RuntimeConfiguration:
    """Seuils de détection et exposition temporelle de chaque porte.

    Une couverture None est inconnue ; un tuple vide déclare une exposition
    nulle. Les intervalles connus suivent la convention [début, fin).
    """

    parameters: CrossingParameters
    coverage: dict[str, tuple[CoverageInterval, ...] | None]
    algorithm_justification: str
    coverage_evidence: dict[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_state() -> dict[str, object]:
    """Identifie le code de profilage et l'état Git sans chemin personnel."""
    digest = hashlib.sha256()
    for path in (Path(__file__).with_name("sector.py"), Path(__file__)):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    try:
        package_version = importlib.metadata.version("cav-gridlock-recovery")
    except importlib.metadata.PackageNotFoundError:
        package_version = "not-installed"
    repository = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, text=True, capture_output=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repository, text=True, capture_output=True, check=True).stdout.strip())
        git_state: dict[str, object] = {"commit": commit, "working_tree": "dirty" if dirty else "clean"}
    except (OSError, subprocess.CalledProcessError):
        git_state = {"commit": None, "working_tree": "unavailable"}
    return {"package_version": package_version, "cgr_e02_code_sha256": digest.hexdigest(), "git": git_state}


def _read_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileInputError(f"{label} absent ou JSON invalide : {path.name}") from error
    if not isinstance(value, dict):
        raise ProfileInputError(f"{label} doit contenir un objet JSON.")
    return value


def _csv_header(path: Path, compressed: bool = False) -> set[str]:
    opener = gzip.open if compressed else Path.open
    if compressed:
        handle = opener(path, "rt", encoding="utf-8", newline="")
    else:
        handle = opener(path, "r", encoding="utf-8", newline="")
    with handle:
        return set(csv.DictReader(handle).fieldnames or ())


def _count_gzip_csv_rows(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _validate_inputs(source: Path, e01_directory: Path, seed_path: Path, geometry_path: Path) -> tuple[dict, dict[str, str]]:
    """Vérifie la source, les exports qualifiés et la géométrie avant calcul."""
    if not source.is_file():
        raise ProfileInputError(f"Source pNEUMA absente : {source.name}")
    manifest_path = e01_directory / "manifest.json"
    manifest = _read_json(manifest_path, "Manifeste de qualification")
    if manifest.get("schema_version") != EXPECTED_E01_SCHEMA or manifest.get("status") != "complete":
        raise ProfileInputError("Le manifeste de qualification doit être complet et utiliser le schéma CGR-E01-1.")
    seed = _read_json(seed_path, "Configuration géométrique")
    source_contract = seed.get("source", {})
    geometry_contract = seed.get("geometry", {})
    actual_source = {"filename": source.name, "size_bytes": source.stat().st_size, "sha256": _sha256(source)}
    expected_source = {
        "filename": source_contract.get("pneuma_file"),
        "sha256": source_contract.get("pneuma_sha256"),
    }
    manifest_source = manifest.get("source", {})
    for key in ("filename", "sha256"):
        if actual_source[key] != expected_source[key] or actual_source[key] != manifest_source.get(key):
            raise ProfileInputError(f"Provenance incompatible pour la source pNEUMA ({key}).")
    if actual_source["size_bytes"] != manifest_source.get("size_bytes"):
        raise ProfileInputError("Provenance incompatible pour la source pNEUMA (taille).")
    if not geometry_path.is_file() or _sha256(geometry_path) != geometry_contract.get("osm_sha256"):
        raise ProfileInputError("Empreinte de la géométrie OSM incompatible avec la configuration du secteur.")
    required = ("trajectories.csv", "observations.csv.gz", "issues.csv", "quality_summary.json")
    hashes: dict[str, str] = {}
    for name in required:
        path = e01_directory / name
        if not path.is_file():
            raise ProfileInputError(f"Export qualifié obligatoire absent : {name}")
        hashes[name] = _sha256(path)
    summary = _read_json(e01_directory / "quality_summary.json", "Bilan de qualification")
    if summary.get("schema_version") != EXPECTED_E01_SCHEMA or summary.get("status") != "complete":
        raise ProfileInputError("Le bilan de qualification est incompatible ou incomplet.")
    headers = {
        "trajectories.csv": (_csv_header(e01_directory / "trajectories.csv"), E01_TRAJECTORY_FIELDS),
        "observations.csv.gz": (_csv_header(e01_directory / "observations.csv.gz", compressed=True), E01_OBSERVATION_FIELDS),
        "issues.csv": (_csv_header(e01_directory / "issues.csv"), E01_ISSUE_FIELDS),
    }
    for name, (actual, expected) in headers.items():
        if not expected.issubset(actual):
            missing = ", ".join(sorted(expected - actual))
            raise ProfileInputError(f"En-tête d'export incompatible pour {name} ; champs absents : {missing}.")
    counts = summary.get("counts")
    if not isinstance(counts, dict):
        raise ProfileInputError("Le bilan de qualification ne contient pas les comptages de réconciliation.")
    required_counts = (
        "candidate_lines",
        "decomposable_lines",
        "structure_excluded_lines",
        "trajectory_export_rows",
        "expected_groups_decomposable_lines",
        "observation_export_rows",
        "all_numeric_finite_groups",
        "numeric_invalid_groups",
        "distinct_nonempty_track_ids",
    )
    if any(type(counts.get(name)) is not int or counts[name] < 0 for name in required_counts):
        raise ProfileInputError("Le bilan de qualification contient un comptage absent ou invalide.")
    if not isinstance(counts.get("duplicate_track_ids"), list):
        raise ProfileInputError("Le bilan de qualification ne décrit pas correctement les identifiants dupliqués.")
    trajectory_rows = _count_csv_rows(e01_directory / "trajectories.csv")
    observation_rows = _count_gzip_csv_rows(e01_directory / "observations.csv.gz")
    reconciliations = (
        (trajectory_rows, counts.get("trajectory_export_rows"), "trajectoires exportées"),
        (observation_rows, counts.get("observation_export_rows"), "observations exportées"),
        (observation_rows, counts.get("expected_groups_decomposable_lines"), "groupes attendus"),
        (counts.get("candidate_lines"), trajectory_rows, "lignes candidates"),
        (counts.get("decomposable_lines", 0) + counts.get("structure_excluded_lines", 0), counts.get("candidate_lines"), "bilan structurel"),
        (counts.get("all_numeric_finite_groups", 0) + counts.get("numeric_invalid_groups", 0), observation_rows, "bilan numérique"),
        (counts.get("distinct_nonempty_track_ids") <= trajectory_rows, True, "identifiants distincts"),
    )
    for actual, expected, label in reconciliations:
        if actual != expected:
            raise ProfileInputError(f"Réconciliation de qualification incohérente : {label}.")
    categories = summary.get("categories")
    if not isinstance(categories, dict) or any(type(value) is not int or value < 0 for value in categories.values()):
        raise ProfileInputError("Le bilan de qualification ne décrit pas correctement les catégories.")
    if sum(categories.values()) != trajectory_rows:
        raise ProfileInputError("Réconciliation de qualification incohérente : catégories de trajectoires.")
    return manifest, hashes


def load_runtime_configuration(path: str | Path, sector: Sector) -> RuntimeConfiguration:
    """Charge les seuils déclarés et les intervalles d'exposition par porte."""
    document = _read_json(Path(path), "Configuration d'exécution")
    if document.get("schema") != "CGR-E02-runtime-1":
        raise SectorConfigurationError("Schéma de configuration d'exécution non pris en charge.")
    algorithm = document.get("algorithm")
    if not isinstance(algorithm, dict):
        raise SectorConfigurationError("La section algorithm est obligatoire.")
    try:
        parameters = CrossingParameters(
            deduplication_s=float(algorithm["deduplication_s"]),
            max_time_gap_s=float(algorithm["max_time_gap_s"]),
            max_space_gap_m=float(algorithm["max_space_gap_m"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SectorConfigurationError("Seuil algorithmique absent ou invalide.") from error
    parameters.validate()
    justification = document.get("algorithm_justification")
    if not isinstance(justification, str) or not justification.strip():
        raise SectorConfigurationError("La justification des seuils algorithmiques est obligatoire.")
    if document.get("validation_reference_used_for_tuning") is not False:
        raise SectorConfigurationError("La configuration doit déclarer que la référence de validation n'a pas servi au réglage.")
    coverage_document = document.get("coverage")
    coverage_evidence_document = document.get("coverage_evidence")
    if not isinstance(coverage_document, dict):
        raise SectorConfigurationError("La couverture doit être déclarée par porte, y compris comme inconnue.")
    if not isinstance(coverage_evidence_document, dict):
        raise SectorConfigurationError("La preuve ou l'absence de preuve de couverture doit être documentée par porte.")
    coverage: dict[str, tuple[CoverageInterval, ...] | None] = {}
    coverage_evidence: dict[str, str] = {}
    for gate in sector.gates:
        evidence = coverage_evidence_document.get(gate.identifier)
        if not isinstance(evidence, str) or not evidence.strip():
            raise SectorConfigurationError(f"Justification de couverture absente pour {gate.identifier}.")
        coverage_evidence[gate.identifier] = evidence.strip()
        value = coverage_document.get(gate.identifier, "missing")
        if value == "unknown":
            coverage[gate.identifier] = None
            continue
        if not isinstance(value, list):
            raise SectorConfigurationError(f"Couverture absente ou invalide pour {gate.identifier}.")
        intervals: list[CoverageInterval] = []
        for raw_interval in value:
            if not isinstance(raw_interval, list) or len(raw_interval) != 2:
                raise SectorConfigurationError(f"Intervalle de couverture invalide pour {gate.identifier}.")
            start, end = float(raw_interval[0]), float(raw_interval[1])
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                raise SectorConfigurationError(f"Bornes de couverture invalides pour {gate.identifier}.")
            intervals.append(CoverageInterval(start, end))
        intervals.sort(key=lambda item: (item.start_s, item.end_s))
        if any(current.start_s < previous.end_s for previous, current in zip(intervals, intervals[1:])):
            raise SectorConfigurationError(f"Les intervalles de couverture de {gate.identifier} se chevauchent.")
        coverage[gate.identifier] = tuple(intervals)
    return RuntimeConfiguration(parameters, coverage, justification.strip(), coverage_evidence)


def iter_tracks(observations_path: str | Path) -> Iterator[list[TrackObservation]]:
    """Lit les observations qualifiées une trajectoire à la fois."""
    current_key: tuple[int, str] | None = None
    current: list[TrackObservation] = []
    pending_break = False
    with gzip.open(observations_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source_line", "group_index", "track_id", "type", "lat", "lon", "time_s", "coordinate_in_range"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ProfileInputError("Schéma des observations qualifiées incompatible.")
        for row in reader:
            try:
                key = (int(row["source_line"]), row["track_id"])
            except ValueError as error:
                raise ProfileInputError("Identité de trajectoire illisible dans les observations qualifiées.") from error
            if current_key is not None and key != current_key:
                if current:
                    yield current
                current = []
                pending_break = False
            current_key = key
            if row["coordinate_in_range"] != "true" or not row["lat"] or not row["lon"] or not row["time_s"]:
                pending_break = True
                continue
            try:
                observation = TrackObservation(int(row["source_line"]), int(row["group_index"]), row["track_id"], row["type"], float(row["lat"]), float(row["lon"]), float(row["time_s"]), pending_break and bool(current))
            except ValueError as error:
                raise ProfileInputError("Valeur normalisée illisible dans les observations qualifiées.") from error
            current.append(observation)
            pending_break = False
    if current:
        yield current


def reconstruct_visits(crossings: list[Crossing]) -> list[Visit]:
    """Reconstruit des visites sans apparier des événements séparés par une rupture."""
    visits: list[Visit] = []
    by_continuity: dict[int, list[Crossing]] = defaultdict(list)
    for crossing in crossings:
        by_continuity[crossing.continuity_id].append(crossing)
    if not crossings:
        return visits
    sequence = 0
    for continuity_id in sorted(by_continuity):
        pending: Crossing | None = None
        for event in sorted(by_continuity[continuity_id], key=lambda item: (item.estimated_time_s, item.gate_id)):
            if event.role == "entry":
                if pending is not None:
                    sequence += 1
                    visits.append(_visit(sequence, pending, None, continuity_id, "censored_exit", "nouvelle entrée avant sortie observable"))
                pending = event
            elif pending is None:
                sequence += 1
                visits.append(_visit(sequence, None, event, continuity_id, "censored_entry", "sortie observée sans entrée observable"))
            else:
                sequence += 1
                visits.append(_visit(sequence, pending, event, continuity_id, "classifiable", "entrée et sortie observées dans la même continuité"))
                pending = None
        if pending is not None:
            sequence += 1
            visits.append(_visit(sequence, pending, None, continuity_id, "censored_exit", "fin de continuité sans sortie observable"))
    return visits


def _crossing_key(crossing: Crossing) -> CrossingKey:
    return (crossing.source_line, crossing.continuity_id, crossing.role, crossing.gate_id, crossing.from_group_index, crossing.to_group_index)


def _visit_ids_by_crossing(visits: list[Visit]) -> dict[CrossingKey, str]:
    """Relie chaque franchissement exact à la visite qui le contient."""
    result: dict[CrossingKey, str] = {}
    for visit in visits:
        for key in (visit.entry_crossing_key, visit.exit_crossing_key):
            if key is None:
                continue
            if key in result:
                raise RuntimeError("Un même franchissement a été affecté à plusieurs visites.")
            result[key] = visit.visit_id
    return result


def _visit(
    sequence: int,
    entry: Crossing | None,
    exit_event: Crossing | None,
    continuity_id: int,
    status: str,
    reason: str,
) -> Visit:
    event = entry or exit_event
    assert event is not None
    route = tuple(gate for gate in (entry.gate_id if entry else None, exit_event.gate_id if exit_event else None) if gate is not None)
    return Visit(
        f"{event.source_line}:{sequence}", event.source_line, event.track_id, event.category,
        continuity_id, entry.gate_id if entry else None, exit_event.gate_id if exit_event else None,
        entry.estimated_time_s if entry else None, exit_event.estimated_time_s if exit_event else None,
        status, route, reason, _crossing_key(entry) if entry else None,
        _crossing_key(exit_event) if exit_event else None,
    )


def _windows(origin_s: float, end_s: float, width_s: float) -> list[tuple[float, float]]:
    if width_s <= 0 or end_s <= origin_s:
        raise ProfileInputError("Fenêtre ou étendue temporelle invalide.")
    count = math.ceil((end_s - origin_s) / width_s)
    return [(origin_s + index * width_s, min(origin_s + (index + 1) * width_s, end_s)) for index in range(count)]


def _intersect_intervals(first: list[CoverageInterval], second: tuple[CoverageInterval, ...]) -> list[CoverageInterval]:
    intersections: list[CoverageInterval] = []
    for left in first:
        for right in second:
            start = max(left.start_s, right.start_s)
            end = min(left.end_s, right.end_s)
            if end > start:
                intersections.append(CoverageInterval(start, end))
    return intersections


def _common_exposure(coverage: dict[str, tuple[CoverageInterval, ...] | None]) -> float | None:
    """Calcule l'intersection temporelle commune, ou indique qu'elle est inconnue."""
    values = list(coverage.values())
    if any(intervals is None for intervals in values):
        return None
    if not values:
        return 0.0
    common = list(values[0] or ())
    for intervals in values[1:]:
        common = _intersect_intervals(common, intervals or ())
    return sum(interval.end_s - interval.start_s for interval in common)


def evaluate_admissibility(sector: Sector, runtime: RuntimeConfiguration, visits: list[Visit]) -> dict[str, object]:
    """Évalue les critères empiriques déclarés dans la configuration du secteur."""
    contract = sector.source_document.get("admissibility")
    if not isinstance(contract, dict):
        raise SectorConfigurationError("La section admissibility du seed est obligatoire.")
    required_keys = (
        "minimum_common_exposure_s",
        "minimum_complete_visits",
        "minimum_distinct_movements",
        "minimum_visits_per_movement",
    )
    try:
        required = {key: float(contract[key]) for key in required_keys}
    except (KeyError, TypeError, ValueError) as error:
        raise SectorConfigurationError("Les quatre critères d'admissibilité doivent être numériques.") from error
    if any(value < 0 for value in required.values()):
        raise SectorConfigurationError("Les critères d'admissibilité ne peuvent pas être négatifs.")
    classifiable = [visit for visit in visits if visit.status == "classifiable" and visit.entry_gate and visit.exit_gate]
    movements = Counter(f"{visit.entry_gate}->{visit.exit_gate}" for visit in classifiable)
    common_exposure = _common_exposure(runtime.coverage)
    qualifying_movements = sum(count >= required["minimum_visits_per_movement"] for count in movements.values())
    criteria: dict[str, dict[str, object]] = {
        "minimum_common_exposure_s": {
            "required": required["minimum_common_exposure_s"],
            "observed": common_exposure,
            "status": "unknown" if common_exposure is None else "pass" if common_exposure >= required["minimum_common_exposure_s"] else "fail",
        },
        "minimum_complete_visits": {
            "required": int(required["minimum_complete_visits"]),
            "observed": len(classifiable),
            "status": "pass" if len(classifiable) >= required["minimum_complete_visits"] else "fail",
        },
        "minimum_distinct_movements": {
            "required": int(required["minimum_distinct_movements"]),
            "observed": len(movements),
            "status": "pass" if len(movements) >= required["minimum_distinct_movements"] else "fail",
        },
        "minimum_visits_per_movement": {
            "required": int(required["minimum_visits_per_movement"]),
            "observed_counts": dict(sorted(movements.items())),
            "movements_meeting_threshold": qualifying_movements,
            "status": "pass" if qualifying_movements >= required["minimum_distinct_movements"] else "fail",
        },
    }
    statuses = {criterion["status"] for criterion in criteria.values()}
    status = "not_admissible" if "fail" in statuses else "undetermined" if "unknown" in statuses else "admissible"
    return {"status": status, "criteria": criteria}


def _exposure(intervals: tuple[CoverageInterval, ...] | None, start: float, end: float) -> tuple[str, float | None]:
    if intervals is None:
        return "unknown", None
    duration = sum(max(0.0, min(end, interval.end_s) - max(start, interval.start_s)) for interval in intervals)
    return ("known" if duration > 0 else "zero_exposure"), duration


def _belongs_to_exposure(time_s: float, intervals: tuple[CoverageInterval, ...]) -> bool:
    """Applique la convention temporelle semi-ouverte aux intervalles exposés."""
    return any(interval.start_s <= time_s < interval.end_s for interval in intervals)


def build_flow_rows(crossings: list[Crossing], sector: Sector, categories: Iterable[str], coverage: dict[str, tuple[CoverageInterval, ...] | None], windows: list[tuple[float, float]]) -> list[dict[str, object]]:
    """Produit les comptages et débits en distinguant zéro observé et couverture inconnue."""
    raw_counts = Counter((event.gate_id, event.category, _window_index(event.estimated_time_s, windows)) for event in crossings)
    covered_counts = Counter(
        (event.gate_id, event.category, _window_index(event.estimated_time_s, windows))
        for event in crossings
        if coverage[event.gate_id] is not None and _belongs_to_exposure(event.estimated_time_s, coverage[event.gate_id])
    )
    rows: list[dict[str, object]] = []
    for gate in sector.gates:
        for category in sorted(set(categories)):
            for index, (start, end) in enumerate(windows):
                status, duration = _exposure(coverage[gate.identifier], start, end)
                raw_count = raw_counts[(gate.identifier, category, index)]
                covered_count = None if duration is None else covered_counts[(gate.identifier, category, index)]
                rate = None if duration is None or duration <= 0 else 3600.0 * covered_count / duration
                rows.append({
                    "gate_id": gate.identifier,
                    "direction": gate.role,
                    "category": category,
                    "window_start_s": start,
                    "window_end_s": end,
                    "raw_passages": raw_count,
                    "passages_in_exposure": covered_count,
                    "exposure_s": duration,
                    "flow_veh_per_h": rate,
                    "coverage_status": status,
                    "coverage_assumption": "intervals_explicitly_provided" if status != "unknown" else "coverage_not_established",
                })
    return rows


def _window_index(time_s: float, windows: list[tuple[float, float]]) -> int | None:
    for index, (start, end) in enumerate(windows):
        if start <= time_s < end:
            return index
    return None


def build_movement_rows(visits: list[Visit], sector: Sector, categories: Iterable[str], windows: list[tuple[float, float]]) -> list[dict[str, object]]:
    """Calcule les proportions sur les seules visites classifiables.

    La fenêtre est celle de l'entrée. Pour une entrée censurée, la sortie observée
    situe la visite dans le temps, sans lui attribuer un mouvement complet.
    """
    entries = sorted(gate.identifier for gate in sector.gates if gate.role == "entry")
    exits = sorted(gate.identifier for gate in sector.gates if gate.role == "exit")
    categories = sorted(set(categories))
    movement_counts = Counter()
    denominators = Counter()
    status_counts = Counter()
    for visit in visits:
        if visit.entry_gate is None or visit.entry_time_s is None:
            exit_time_s = visit.exit_time_s if visit.exit_time_s is not None else -math.inf
            status_counts[("NO_OBSERVED_ENTRY", visit.category, _window_index(exit_time_s, windows), visit.status)] += 1
            continue
        window_index = _window_index(visit.entry_time_s, windows)
        if visit.status == "classifiable" and visit.exit_gate is not None:
            movement_counts[(visit.entry_gate, visit.exit_gate, visit.category, window_index)] += 1
            denominators[(visit.entry_gate, visit.category, window_index)] += 1
        else:
            status_counts[(visit.entry_gate, visit.category, window_index, visit.status)] += 1
    rows: list[dict[str, object]] = []
    for entry in entries:
        for exit_id in exits:
            for category in categories:
                for index, (start, end) in enumerate(windows):
                    count = movement_counts[(entry, exit_id, category, index)]
                    denominator = denominators[(entry, category, index)]
                    rows.append({
                        "record_type": "movement",
                        "entry_gate": entry,
                        "exit_gate": exit_id,
                        "category": category,
                        "window_start_s": start,
                        "window_end_s": end,
                        "count": count,
                        "denominator_classifiable": denominator,
                        "proportion": None if denominator == 0 else count / denominator,
                        "visit_status": "classifiable",
                    })
    for (entry, category, index, status), count in sorted(status_counts.items(), key=lambda item: str(item[0])):
        start, end = windows[index] if index is not None else (None, None)
        rows.append({
            "record_type": "unclassified_visit",
            "entry_gate": entry,
            "exit_gate": None,
            "category": category,
            "window_start_s": start,
            "window_end_s": end,
            "count": count,
            "denominator_classifiable": None,
            "proportion": None,
            "visit_status": status,
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]], headers: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(({key: "" if value is None else value for key, value in row.items()} for row in rows))


def _count_csv_rows(path: Path) -> int:
    """Compte les lignes de données d'un export CSV sans les conserver."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _sector_geojson(sector: Sector) -> dict:
    features = []
    for gate in sector.source_document["gates"]:
        features.append({"type": "Feature", "properties": {"kind": "gate", "id": gate["id"]}, "geometry": {"type": "LineString", "coordinates": [[point["lon"], point["lat"]] for point in gate["endpoints"]]}})
    for branch in sector.source_document["branches"]:
        gate = next(item for item in sector.source_document["gates"] if item["id"] == branch["id"])
        center = sector.source_document["sector"]["center"]
        features.append({"type": "Feature", "properties": {"kind": "branch_axis", "id": branch["id"], "role": branch["role"]}, "geometry": {"type": "LineString", "coordinates": [[center["lon"], center["lat"]], [gate["center"]["lon"], gate["center"]["lat"]]]}})
    return {"type": "FeatureCollection", "features": features}


def profile_pneuma(source_path: str | Path, e01_directory: str | Path, sector_seed_path: str | Path, geometry_path: str | Path, runtime_config_path: str | Path, output_directory: str | Path) -> dict:
    """Construit les passages, visites et mouvements à partir des exports qualifiés.

    Le fichier pNEUMA source sert au contrôle d'identité ; les franchissements
    sont calculés depuis les observations exportées, avec les seuils déclarés.
    """
    source = Path(source_path).resolve()
    e01_directory = Path(e01_directory).resolve()
    seed_path = Path(sector_seed_path).resolve()
    geometry_path = Path(geometry_path).resolve()
    runtime_path = Path(runtime_config_path).resolve()
    destination = Path(output_directory).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ProfileInputError("Le dossier de sortie existe déjà et n'est pas vide.")
    manifest_e01, export_hashes = _validate_inputs(source, e01_directory, seed_path, geometry_path)
    sector = load_sector(seed_path)
    runtime = load_runtime_configuration(runtime_path, sector)
    e01_summary = _read_json(e01_directory / "quality_summary.json", "Bilan de qualification")
    input_trajectory_rows = _count_csv_rows(e01_directory / "trajectories.csv")
    origin_s = float(sector.source_document["aggregation"]["origin_s"])
    window_s = float(sector.source_document["aggregation"]["window_s"])
    end_s = float(e01_summary["time_seconds"]["max"])
    windows = _windows(origin_s, end_s, window_s)

    all_crossings: list[Crossing] = []
    all_visits: list[Visit] = []
    categories: set[str] = set()
    trajectories = rupture_count = 0
    for observations in iter_tracks(e01_directory / "observations.csv.gz"):
        trajectories += 1
        categories.add(observations[0].category)
        crossings, ruptures = detect_crossings(observations, sector, runtime.parameters)
        rupture_count += ruptures
        all_crossings.extend(crossings)
        all_visits.extend(reconstruct_visits(crossings))

    flow_rows = build_flow_rows(all_crossings, sector, categories, runtime.coverage, windows)
    movement_rows = build_movement_rows(all_visits, sector, categories, windows)
    visit_counts = Counter(visit.status for visit in all_visits)
    crossings_by_gate = Counter(event.gate_id for event in all_crossings)
    crossings_by_category = Counter(event.category for event in all_crossings)
    movements = Counter(
        f"{visit.entry_gate}->{visit.exit_gate}"
        for visit in all_visits
        if visit.status == "classifiable" and visit.entry_gate and visit.exit_gate
    )
    admissibility = evaluate_admissibility(sector, runtime, all_visits)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "execution": {"status": "succeeded"},
        "empirical_admissibility": admissibility,
        "scientific_validation": {
            "status": "pending",
            "validation_reference": "pending",
            "sensitivity_analysis": "not_assessed_by_automatic_pipeline",
        },
        "method": "oriented_finite_virtual_gates",
        "counts": {
            "input_trajectory_rows": input_trajectory_rows,
            "trajectories_with_valid_observations": trajectories,
            "trajectories_without_valid_observations": input_trajectory_rows - trajectories,
            "crossings": len(all_crossings),
            "visits": len(all_visits),
            "ruptures": rupture_count,
            "visits_by_status": dict(sorted(visit_counts.items())),
        },
        "crossings_by_gate": dict(sorted(crossings_by_gate.items())),
        "crossings_by_category": dict(sorted(crossings_by_category.items())),
        "classifiable_movements": dict(sorted(movements.items())),
        "coverage": {
            gate: {"status": "unknown", "evidence": runtime.coverage_evidence[gate]}
            if intervals is None else {
                "status": "established" if intervals else "zero_exposure",
                "interval_count": len(intervals),
                "duration_s": sum(interval.end_s - interval.start_s for interval in intervals),
                "evidence": runtime.coverage_evidence[gate],
            }
            for gate, intervals in sorted(runtime.coverage.items())
        },
        "limitations": ["La méthode détecte des intersections avec des portes finies orientées ; elle ne réalise aucun map-matching de voie.", "Les débits ne sont définis que lorsque l'exposition de la porte est explicitement fournie ; seuls les événements inclus dans cette exposition alimentent leur numérateur.", "Les mouvements sont des couples entrée-sortie observés dans le secteur, pas des origines-destinations réelles."],
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".cgr-e02-", dir=destination.parent))
    try:
        runtime_copy = {
            "schema": "CGR-E02-runtime-used-1",
            "sector_seed": sector.source_document,
            "sector_seed_sha256": _sha256(seed_path),
            "metric_crs": {
                "name": "repère tangent équirectangulaire local",
                "origin_lat": sector.center_lat,
                "origin_lon": sector.center_lon,
                "units": "m",
            },
            "algorithm": asdict(runtime.parameters),
            "algorithm_justification": runtime.algorithm_justification,
            "validation_reference_used_for_tuning": False,
            "coverage": {
                gate: "unknown" if intervals is None else [[item.start_s, item.end_s] for item in intervals]
                for gate, intervals in sorted(runtime.coverage.items())
            },
            "coverage_evidence": dict(sorted(runtime.coverage_evidence.items())),
            "aggregation": {
                "origin_s": origin_s,
                "window_s": window_s,
                "interval_convention": "[a,b)",
                "terminal_s": end_s,
            },
        }
        (stage / "sector_config.json").write_text(json.dumps(runtime_copy, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (stage / "sector.geojson").write_text(json.dumps(_sector_geojson(sector), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        crossing_to_visit = _visit_ids_by_crossing(all_visits)
        crossing_rows = [
            {
                **asdict(event),
                "visit_id": crossing_to_visit.get(_crossing_key(event), ""),
                "status": "accepted",
                "uncertainty_reason": "interpolation locale entre observations encadrantes",
            }
            for event in all_crossings
        ]
        _write_csv(stage / "crossings.csv", crossing_rows, ["source_line", "track_id", "category", "gate_id", "role", "continuity_id", "from_group_index", "to_group_index", "estimated_time_s", "interval_start_s", "interval_end_s", "visit_id", "status", "uncertainty_reason"])
        route_rows = [
            {
                "visit_id": visit.visit_id,
                "source_line": visit.source_line,
                "track_id": visit.track_id,
                "category": visit.category,
                "continuity_id": visit.continuity_id,
                "branch_sequence": ">".join(visit.route),
                "entry_gate": visit.entry_gate,
                "exit_gate": visit.exit_gate,
                "entry_time_s": visit.entry_time_s,
                "exit_time_s": visit.exit_time_s,
                "status": visit.status,
                "censored_entry": str(visit.status == "censored_entry").lower(),
                "censored_exit": str(visit.status == "censored_exit").lower(),
                "reason": visit.reason,
            }
            for visit in all_visits
        ]
        _write_csv(stage / "partial_routes.csv", route_rows, ["visit_id", "source_line", "track_id", "category", "continuity_id", "branch_sequence", "entry_gate", "exit_gate", "entry_time_s", "exit_time_s", "status", "censored_entry", "censored_exit", "reason"])
        _write_csv(stage / "flow_profile.csv", flow_rows, ["gate_id", "direction", "category", "window_start_s", "window_end_s", "raw_passages", "passages_in_exposure", "exposure_s", "flow_veh_per_h", "coverage_status", "coverage_assumption"])
        _write_csv(stage / "movement_profile.csv", movement_rows, ["record_type", "entry_gate", "exit_gate", "category", "window_start_s", "window_end_s", "count", "denominator_classifiable", "proportion", "visit_status"])
        _write_csv(stage / "validation_reference.csv", [], ["source_line", "track_id", "review_set", "entry_gate", "exit_gate", "crossing_times_s", "censorship", "decision", "notes"])
        (stage / "quality_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report = "# Profil du secteur pNEUMA\n\n"
        report += f"Exécution technique : **{summary['execution']['status']}**. {trajectories} trajectoires, {len(all_crossings)} franchissements et {len(all_visits)} visites reconstruits.\n\n"
        report += "## Admissibilité et validation\n\n"
        report += f"- Admissibilité empirique : **{admissibility['status']}** ; critères : {admissibility['criteria']}.\n"
        report += "- Validation scientifique : **pending** ; la référence de validation et la sensibilité restent évaluées séparément.\n\n"
        report += "## Qualité et couverture\n\n"
        report += "- Méthode : intersections orientées avec des portes virtuelles finies.\n"
        report += f"- Franchissements par porte : {summary['crossings_by_gate']}.\n"
        report += f"- Franchissements par catégorie : {summary['crossings_by_category']}.\n"
        report += f"- Visites : {dict(sorted(visit_counts.items()))}.\n"
        report += f"- Mouvements classifiables : {summary['classifiable_movements']}.\n"
        report += f"- Couverture : {summary['coverage']}.\n\n"
        report += "## Configuration et reproduction\n\n"
        report += f"- Repère métrique : tangent équirectangulaire local centré en ({sector.center_lat}, {sector.center_lon}).\n"
        report += f"- Seuils utilisés : {asdict(runtime.parameters)}.\n"
        report += f"- Justification : {runtime.algorithm_justification}.\n"
        report += "- La géométrie de contrôle est exportée dans `sector.geojson` et la configuration complète dans `sector_config.json`.\n"
        report += "- Reproduction : relancer `scripts/profile_pneuma.py` avec les cinq entrées dont les empreintes figurent dans `manifest.json`.\n\n"
        report += "## Limites\n\n" + "\n".join(f"- {item}" for item in summary["limitations"]) + "\n\n"
        report += "## Référence de validation\n\nLa référence est fournie comme gabarit vide : elle doit rester indépendante du réglage automatique et être renseignée avant validation scientifique.\n"
        report += "\n## Sensibilité\n\nL'analyse bornée des seuils n'est pas exécutée automatiquement par ce pipeline ; son statut doit être documenté séparément avant validation scientifique.\n"
        (stage / "profile_report.md").write_text(report, encoding="utf-8")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "execution": summary["execution"],
            "empirical_admissibility": admissibility,
            "scientific_validation": summary["scientific_validation"],
            "source": manifest_e01["source"],
            "inputs": {
                "cgr_e01_manifest_sha256": _sha256(e01_directory / "manifest.json"),
                "locally_computed_cgr_e01_export_sha256": dict(sorted(export_hashes.items())),
                "sector_seed_sha256": _sha256(seed_path),
                "geometry_sha256": _sha256(geometry_path),
                "runtime_config_sha256": _sha256(runtime_path),
            },
            "configuration": runtime_copy,
            "software": _code_state(),
            "outputs": list(OUTPUTS),
            "validation_reference": "empty_template_requires_independent_annotation",
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        destination.mkdir(exist_ok=True)
        for name in [item for item in OUTPUTS if item != "manifest.json"] + ["manifest.json"]:
            os.replace(stage / name, destination / name)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return summary
