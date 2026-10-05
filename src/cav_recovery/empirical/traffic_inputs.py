"""Lecture et validation des profils de trafic utilisés par le contrat."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable
from types import MappingProxyType

from ..canonical_scenario import CANONICAL_SCENARIO, CanonicalScenario, Sector


EXPECTED_INPUT_SHA256 = MappingProxyType({item.filename: item.sha256 for item in CANONICAL_SCENARIO.empirical.profiles})
PROFILE_SCHEMA_VERSION = CANONICAL_SCENARIO.empirical.profile_schema
COVERAGE_SCHEMA_VERSION = CANONICAL_SCENARIO.empirical.coverage_schema
ALL_CATEGORIES = CANONICAL_SCENARIO.sector.categories
ENTRY_GATES = CANONICAL_SCENARIO.sector.entry_gates
EXIT_GATES = CANONICAL_SCENARIO.sector.exit_gates
ALL_GATES = CANONICAL_SCENARIO.sector.gates
FLOW_FIELDS = {
    "gate_id",
    "direction",
    "category",
    "window_start_s",
    "window_end_s",
    "raw_passages",
    "passages_in_exposure",
    "exposure_s",
    "coverage_status",
}
MOVEMENT_FIELDS = {
    "record_type",
    "entry_gate",
    "exit_gate",
    "category",
    "window_start_s",
    "window_end_s",
    "count",
    "denominator_classifiable",
    "proportion",
    "visit_status",
}


class ContractInputError(ValueError):
    """Signale des données d'entrée incompatibles avec le contrat."""


@dataclass(frozen=True)
class TrafficInputs:
    """Regroupe les profils contrôlés et leur provenance."""

    hashes: dict[str, str]
    manifest: dict
    coverage: dict
    summary: dict
    flows: dict[tuple[str, str, float], dict[str, object]]
    windows: list[tuple[float, float]]
    movements: dict[tuple[str, str, str, float], dict[str, int]]
    unclassified: list[dict[str, object]]


def same_number(first: float, second: float) -> bool:
    """Compare deux valeurs issues de fichiers décimaux avec une marge minimale."""

    return math.isclose(first, second, rel_tol=0.0, abs_tol=1e-9)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractInputError(f"{label} absent ou JSON invalide : {path.name}") from error
    if not isinstance(value, dict):
        raise ContractInputError(f"{label} doit contenir un objet JSON.")
    return value


def _read_csv(path: Path, required_fields: set[str], label: str) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            if not required_fields.issubset(fields):
                missing = ", ".join(sorted(required_fields - fields))
                raise ContractInputError(f"Colonnes absentes dans {label} : {missing}.")
            return list(reader)
    except OSError as error:
        raise ContractInputError(f"{label} absent ou illisible : {path.name}") from error


def _as_float(value: str, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ContractInputError(f"Valeur numérique invalide pour {label}.") from error
    if not math.isfinite(number):
        raise ContractInputError(f"Valeur non finie pour {label}.")
    return number


def _as_count(value: str, label: str) -> int:
    number = _as_float(value, label)
    if number < 0 or not number.is_integer():
        raise ContractInputError(f"Comptage invalide pour {label}.")
    return int(number)


def _verify_hashes(profile_dir: Path, coverage_path: Path, config: CanonicalScenario) -> dict[str, str]:
    actual: dict[str, str] = {}
    for identity in config.empirical.profiles:
        name = identity.filename
        path = coverage_path if name == "coverage.json" else profile_dir / name
        if not path.is_file():
            raise ContractInputError(f"Fichier d'entrée obligatoire absent : {name}")
        actual[name] = _sha256(path)
        if actual[name] != identity.sha256:
            raise ContractInputError(f"Empreinte SHA-256 inattendue pour {name}.")
    return actual


def _validate_profile_structure(manifest: dict, sector_config: dict, summary: dict) -> None:
    """Contrôle la cohérence des documents, indépendamment de l'identité C3."""

    if manifest.get("execution", {}).get("status") != "succeeded":
        raise ContractInputError("Le manifeste n'identifie pas une exécution réussie compatible.")
    if manifest.get("configuration") != sector_config:
        raise ContractInputError("La configuration ne correspond pas à celle du manifeste.")
    if summary.get("execution", {}).get("status") != "succeeded":
        raise ContractInputError("Le bilan d'entrée est incompatible ou incomplet.")
    upstream_inputs = manifest.get("inputs")
    required_provenance = {
        "cgr_e01_manifest_sha256",
        "geometry_sha256",
        "locally_computed_cgr_e01_export_sha256",
        "runtime_config_sha256",
        "sector_seed_sha256",
    }
    if not isinstance(upstream_inputs, dict) or not required_provenance.issubset(upstream_inputs):
        raise ContractInputError("La provenance des profils d'entrée est incomplète.")
    if not isinstance(manifest.get("software"), dict):
        raise ContractInputError("L'identité du code ayant produit les profils est absente du manifeste.")


def _validate_canonical_identity(
    manifest: dict, sector_config: dict, summary: dict, coverage: dict, config: CanonicalScenario,
) -> None:
    """Garde les identités figées même après le contrôle des empreintes."""
    if manifest.get("schema_version") != config.empirical.profile_schema:
        raise ContractInputError("Le manifeste n'identifie pas une exécution réussie compatible.")
    if summary.get("schema_version") != config.empirical.profile_schema:
        raise ContractInputError("Le bilan d'entrée est incompatible ou incomplet.")
    if summary.get("method") != config.empirical.extraction_method:
        raise ContractInputError("La méthode d'extraction n'est pas celle attendue.")
    source = manifest.get("source", {})
    expected_source = config.empirical.source.source_record()
    if source != expected_source:
        raise ContractInputError("La source pNEUMA déclarée n'est pas la source attendue.")

    seed = sector_config.get("sector_seed", {})
    sector = seed.get("sector", {})
    if sector.get("center_osm_node_id") != config.sector.node_id or sector.get("id") != config.sector.seed_id:
        raise ContractInputError("Le secteur n'est pas C3 au nœud OSM 250691665.")
    branches = seed.get("branches")
    if not isinstance(branches, list):
        raise ContractInputError("Les branches du secteur sont absentes.")
    roles = {row.get("id"): row.get("role") for row in branches if isinstance(row, dict)}
    expected_roles = {**{gate: "entry" for gate in config.sector.entry_gates},
                      **{gate: "exit" for gate in config.sector.exit_gates}}
    if roles != expected_roles:
        raise ContractInputError("Les quatre portes ou leurs rôles sont incompatibles.")

    aggregation = sector_config.get("aggregation", {})
    start, end = config.sector.observation_interval_s
    expected_aggregation = {"origin_s": start, "terminal_s": end, "window_s": config.sector.window_s,
                            "interval_convention": config.sector.interval_convention}
    if aggregation != expected_aggregation:
        raise ContractInputError("Le calendrier d'agrégation est incompatible.")
    expected_intervals = {gate: [[start, end]] for gate in config.sector.gates}
    if sector_config.get("coverage") != expected_intervals:
        raise ContractInputError("La couverture de la configuration ne correspond pas à [0,0 ; 802,8).")

    if coverage.get("schema") != config.empirical.coverage_schema or coverage.get("status") != "known":
        raise ContractInputError("La preuve de couverture n'est pas établie.")
    if coverage.get("intervals") != [[start, end]]:
        raise ContractInputError("L'intervalle de couverture est incompatible.")
    coverage_source = coverage.get("source_time_range", {})
    for key in ("filename", "sha256", "size_bytes"):
        if coverage_source.get(key) != source[key]:
            raise ContractInputError(f"La couverture et le manifeste divergent sur la source ({key}).")


def _window_key(row: dict[str, str], label: str) -> tuple[float, float]:
    start = _as_float(row["window_start_s"], f"{label}.window_start_s")
    end = _as_float(row["window_end_s"], f"{label}.window_end_s")
    if end <= start:
        raise ContractInputError(f"Fenêtre temporelle invalide dans {label}.")
    return start, end


def _validate_window_structure(windows: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Refuse les fenêtres invalides ou superposées, sans imposer une période."""
    ordered = sorted(set(windows))
    for index, (start, end) in enumerate(ordered):
        if not all(math.isfinite(value) for value in (start, end)) or end <= start:
            raise ContractInputError("Fenêtre temporelle invalide.")
        if index and start < ordered[index - 1][1]:
            raise ContractInputError("Les fenêtres temporelles se superposent.")
    return ordered


def _validate_windows(windows: Iterable[tuple[float, float]], sector: Sector) -> list[tuple[float, float]]:
    """Contrôle les treize minutes complètes et la dernière fenêtre partielle."""

    ordered = _validate_window_structure(windows)
    origin, terminal = sector.observation_interval_s
    expected_count = math.ceil((terminal - origin) / sector.window_s)
    if len(ordered) != expected_count:
        raise ContractInputError("Le contrat exige exactement 14 fenêtres.")
    for index, (start, end) in enumerate(ordered):
        expected_start = origin + index * sector.window_s
        expected_end = min(origin + (index + 1) * sector.window_s, terminal)
        if not same_number(start, expected_start) or not same_number(end, expected_end):
            raise ContractInputError("Les fenêtres ne suivent pas le calendrier [0,0 ; 802,8).")
    return ordered


def _parse_flow_rows(
    rows: list[dict[str, str]],
    sector: Sector,
) -> tuple[dict[tuple[str, str, float], dict[str, object]], list[tuple[float, float]]]:
    """Valide le profil de flux et construit son index porte/catégorie/fenêtre."""

    parsed: dict[tuple[str, str, float], dict[str, object]] = {}
    entry_gates, exit_gates = sector.entry_gates, sector.exit_gates
    gates, categories = sector.gates, sector.categories
    windows: set[tuple[float, float]] = set()
    roles = {**{gate: "entry" for gate in entry_gates}, **{gate: "exit" for gate in exit_gates}}
    for line_number, row in enumerate(rows, start=2):
        gate = row["gate_id"]
        category = row["category"]
        if gate not in roles or row["direction"] != roles[gate]:
            raise ContractInputError(f"Porte ou direction invalide dans flow_profile.csv, ligne {line_number}.")
        if category not in categories:
            raise ContractInputError(f"Catégorie inattendue dans flow_profile.csv : {category!r}.")
        start, end = _window_key(row, "flow_profile.csv")
        windows.add((start, end))
        key = (gate, category, start)
        if key in parsed:
            raise ContractInputError("Doublon porte/catégorie/fenêtre dans flow_profile.csv.")
        raw = _as_count(row["raw_passages"], "raw_passages")
        count = _as_count(row["passages_in_exposure"], "passages_in_exposure")
        exposure = _as_float(row["exposure_s"], "exposure_s")
        if row["coverage_status"] != "known" or not same_number(exposure, end - start):
            raise ContractInputError("Chaque ligne du profil doit avoir une exposition connue égale à sa fenêtre.")
        if raw != count:
            raise ContractInputError("Un passage brut se trouve hors de l'exposition retenue.")
        parsed[key] = {"start": start, "end": end, "count": count, "exposure": exposure}

    ordered = _validate_windows(windows, sector)
    expected = {(gate, category, start) for gate in gates for category in categories for start, _ in ordered}
    if set(parsed) != expected:
        raise ContractInputError("La grille porte/catégorie/fenêtre de flow_profile.csv est incomplète.")
    return parsed, ordered


def _parse_movement_rows(
    rows: list[dict[str, str]],
    windows: list[tuple[float, float]],
    sector: Sector,
) -> tuple[dict[tuple[str, str, str, float], dict[str, int]], list[dict[str, object]]]:
    """Valide les mouvements classifiables et conserve les visites censurées à part."""

    movements: dict[tuple[str, str, str, float], dict[str, int]] = {}
    entry_gates, exit_gates, categories = sector.entry_gates, sector.exit_gates, sector.categories
    unclassified: list[dict[str, object]] = []
    valid_starts = {start: end for start, end in windows}
    for line_number, row in enumerate(rows, start=2):
        start, end = _window_key(row, "movement_profile.csv")
        if start not in valid_starts or not same_number(end, valid_starts[start]):
            raise ContractInputError("Une fenêtre du profil de mouvements ne correspond pas au profil de flux.")
        category = row["category"]
        if category not in categories:
            raise ContractInputError(f"Catégorie inattendue dans movement_profile.csv : {category!r}.")
        count = _as_count(row["count"], "movement_profile.count")
        if row["record_type"] == "movement":
            entry = row["entry_gate"]
            exit_gate = row["exit_gate"]
            if entry not in entry_gates or exit_gate not in exit_gates or row["visit_status"] != "classifiable":
                raise ContractInputError(f"Mouvement invalide à la ligne {line_number}.")
            denominator = _as_count(row["denominator_classifiable"], "denominator_classifiable")
            proportion = row["proportion"]
            if denominator == 0:
                if proportion != "":
                    raise ContractInputError("Une probabilité est fournie avec un dénominateur nul.")
            elif not same_number(_as_float(proportion, "proportion"), count / denominator):
                raise ContractInputError("Une proportion de mouvement ne correspond pas à ses effectifs.")
            key = (entry, exit_gate, category, start)
            if key in movements:
                raise ContractInputError("Doublon de mouvement dans movement_profile.csv.")
            movements[key] = {"count": count, "denominator": denominator}
        elif row["record_type"] == "unclassified_visit":
            if row["visit_status"] not in {"censored_entry", "censored_exit"}:
                raise ContractInputError("Statut de visite non classifiable inattendu.")
            entry = row["entry_gate"]
            if row["visit_status"] == "censored_exit" and entry not in entry_gates:
                raise ContractInputError("Une censure de sortie doit conserver sa porte d'entrée.")
            if row["visit_status"] == "censored_entry" and entry != "NO_OBSERVED_ENTRY":
                raise ContractInputError("Une censure d'entrée ne doit pas inventer de porte d'entrée.")
            unclassified.append({
                "entry_gate": entry,
                "category": category,
                "start": start,
                "count": count,
                "status": row["visit_status"],
            })
        else:
            raise ContractInputError(f"Type d'enregistrement inconnu à la ligne {line_number}.")

    expected = {
        (entry, exit_gate, category, start)
        for entry in entry_gates
        for exit_gate in exit_gates
        for category in categories
        for start, _ in windows
    }
    if set(movements) != expected:
        raise ContractInputError("La grille des mouvements classifiables est incomplète.")
    # Le même dénominateur est répété pour chaque sortie dans le profil source.
    # Il ne doit donc être vérifié qu'une fois par entrée, catégorie et fenêtre.
    for entry in entry_gates:
        for category in categories:
            for start, _ in windows:
                values = [movements[(entry, exit_gate, category, start)] for exit_gate in exit_gates]
                denominators = {value["denominator"] for value in values}
                if len(denominators) != 1 or sum(value["count"] for value in values) != values[0]["denominator"]:
                    raise ContractInputError("Les mouvements ne se réconcilient pas avec leur dénominateur répété.")
    return movements, unclassified


def _validate_quality_reconciliations(
    flows: dict[tuple[str, str, float], dict[str, object]],
    windows: list[tuple[float, float]],
    summary: dict,
    sector: Sector,
) -> None:
    gates, categories = sector.gates, sector.categories
    by_gate = {
        gate: sum(int(flows[(gate, category, start)]["count"]) for category in categories for start, _ in windows)
        for gate in gates
    }
    by_category = {
        category: sum(int(flows[(gate, category, start)]["count"]) for gate in gates for start, _ in windows)
        for category in categories
    }
    if summary.get("crossings_by_gate") != by_gate or summary.get("crossings_by_category") != by_category:
        raise ContractInputError("Le profil de flux ne se réconcilie pas avec le bilan d'entrée.")
    if summary.get("counts", {}).get("crossings") != sum(by_gate.values()):
        raise ContractInputError("Le total de franchissements du bilan d'entrée est incohérent.")


def load_traffic_inputs(profile_dir: str | Path, coverage_path: str | Path) -> TrafficInputs:
    """Lit et valide les profils de flux, les mouvements et leur provenance."""
    return _load_traffic_inputs(Path(profile_dir).resolve(), Path(coverage_path).resolve(), CANONICAL_SCENARIO)


def _load_traffic_inputs(source_dir: Path, coverage_file: Path, config: CanonicalScenario) -> TrafficInputs:
    hashes = _verify_hashes(source_dir, coverage_file, config)
    manifest = _read_json(source_dir / "manifest.json", "Manifeste source")
    sector_config = _read_json(source_dir / "sector_config.json", "Configuration du secteur")
    summary = _read_json(source_dir / "quality_summary.json", "Bilan des profils")
    coverage = _read_json(coverage_file, "Couverture temporelle")
    _validate_profile_structure(manifest, sector_config, summary)
    _validate_canonical_identity(manifest, sector_config, summary, coverage, config)

    flow_rows = _read_csv(source_dir / "flow_profile.csv", FLOW_FIELDS, "flow_profile.csv")
    movement_rows = _read_csv(source_dir / "movement_profile.csv", MOVEMENT_FIELDS, "movement_profile.csv")
    flows, windows = _parse_flow_rows(flow_rows, config.sector)
    movements, unclassified = _parse_movement_rows(movement_rows, windows, config.sector)
    _validate_quality_reconciliations(flows, windows, summary, config.sector)
    return TrafficInputs(hashes, manifest, coverage, summary, flows, windows, movements, unclassified)
