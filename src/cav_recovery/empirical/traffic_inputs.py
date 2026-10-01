"""Lecture et validation des profils de trafic utilisés par le contrat."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable


EXPECTED_INPUT_SHA256 = {
    "manifest.json": "d6b659330d708c2d766088750a7e27ef44a25b435531a206d04d799e774da1af",
    "sector_config.json": "c62bd8543bfbe655e78ae067bf3fe8ae0f6bee14cc19ccda2deb50e8ca6300d2",
    "quality_summary.json": "22039ca750de97e74f0c6412f59f34514232d18fb7491c6e63686648986021b8",
    "flow_profile.csv": "b33be95fe52a1968a01f9b3e83e1423a8c883883a9e91287d610ee2edf56e2a8",
    "movement_profile.csv": "2fc9915f7eb4433873d8e7de37e413b6dfb962abf8769c57b27d50c45a442dda",
    "coverage.json": "c856992dc1733bd0aa131787edf8609d44a9928343b1919c1bec00285d02038d",
}

PROFILE_SCHEMA_VERSION = "CGR-E02-1"
COVERAGE_SCHEMA_VERSION = "CGR-E02-coverage-evidence-2"
ALL_CATEGORIES = ("Car", "Taxi", "Motorcycle", "Bus", "Medium Vehicle", "Heavy Vehicle")
ENTRY_GATES = ("W23183369_IN", "W284241336_IN")
EXIT_GATES = ("W23183369_OUT", "W284241336_OUT")
ALL_GATES = ENTRY_GATES + EXIT_GATES
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


def _verify_hashes(profile_dir: Path, coverage_path: Path) -> dict[str, str]:
    paths = {
        "manifest.json": profile_dir / "manifest.json",
        "sector_config.json": profile_dir / "sector_config.json",
        "quality_summary.json": profile_dir / "quality_summary.json",
        "flow_profile.csv": profile_dir / "flow_profile.csv",
        "movement_profile.csv": profile_dir / "movement_profile.csv",
        "coverage.json": coverage_path,
    }
    actual: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ContractInputError(f"Fichier d'entrée obligatoire absent : {name}")
        actual[name] = _sha256(path)
        if actual[name] != EXPECTED_INPUT_SHA256[name]:
            raise ContractInputError(f"Empreinte SHA-256 inattendue pour {name}.")
    return actual


def _validate_identity(manifest: dict, sector_config: dict, summary: dict, coverage: dict) -> None:
    """Vérifie que tous les fichiers décrivent la même source et le même secteur."""

    if (
        manifest.get("schema_version") != PROFILE_SCHEMA_VERSION
        or manifest.get("execution", {}).get("status") != "succeeded"
    ):
        raise ContractInputError("Le manifeste n'identifie pas une exécution réussie compatible.")
    if manifest.get("configuration") != sector_config:
        raise ContractInputError("La configuration ne correspond pas à celle du manifeste.")
    if (
        summary.get("schema_version") != PROFILE_SCHEMA_VERSION
        or summary.get("execution", {}).get("status") != "succeeded"
    ):
        raise ContractInputError("Le bilan d'entrée est incompatible ou incomplet.")
    if summary.get("method") != "oriented_finite_virtual_gates":
        raise ContractInputError("La méthode d'extraction n'est pas celle attendue.")
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

    source = manifest.get("source", {})
    expected_source = {
        "filename": "20181024_d3_0830_0900.csv",
        "sha256": "17970bd3f8e167df3ef54792e8fb7f874f89a9a0aceb14571321e9a48fbeea0d",
        "size_bytes": 199_512_534,
    }
    if source != expected_source:
        raise ContractInputError("La source pNEUMA déclarée n'est pas la source attendue.")

    seed = sector_config.get("sector_seed", {})
    sector = seed.get("sector", {})
    if sector.get("center_osm_node_id") != 250691665 or sector.get("id") != "PNEUMA_D3_NODE_250691665":
        raise ContractInputError("Le secteur n'est pas C3 au nœud OSM 250691665.")
    branches = seed.get("branches")
    if not isinstance(branches, list):
        raise ContractInputError("Les branches du secteur sont absentes.")
    roles = {row.get("id"): row.get("role") for row in branches if isinstance(row, dict)}
    expected_roles = {**{gate: "entry" for gate in ENTRY_GATES}, **{gate: "exit" for gate in EXIT_GATES}}
    if roles != expected_roles:
        raise ContractInputError("Les quatre portes ou leurs rôles sont incompatibles.")

    aggregation = sector_config.get("aggregation", {})
    expected_aggregation = {"origin_s": 0.0, "terminal_s": 802.8, "window_s": 60.0, "interval_convention": "[a,b)"}
    if aggregation != expected_aggregation:
        raise ContractInputError("Le calendrier d'agrégation est incompatible.")
    expected_intervals = {gate: [[0.0, 802.8]] for gate in ALL_GATES}
    if sector_config.get("coverage") != expected_intervals:
        raise ContractInputError("La couverture de la configuration ne correspond pas à [0,0 ; 802,8).")

    if coverage.get("schema") != COVERAGE_SCHEMA_VERSION or coverage.get("status") != "known":
        raise ContractInputError("La preuve de couverture n'est pas établie.")
    if coverage.get("intervals") != [[0.0, 802.8]]:
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


def _validate_windows(windows: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Contrôle les treize minutes complètes et la dernière fenêtre partielle."""

    ordered = sorted(set(windows))
    if len(ordered) != 14:
        raise ContractInputError("Le contrat exige exactement 14 fenêtres.")
    for index, (start, end) in enumerate(ordered):
        expected_start = index * 60.0
        expected_end = min((index + 1) * 60.0, 802.8)
        if not same_number(start, expected_start) or not same_number(end, expected_end):
            raise ContractInputError("Les fenêtres ne suivent pas le calendrier [0,0 ; 802,8).")
    return ordered


def _parse_flow_rows(
    rows: list[dict[str, str]],
) -> tuple[dict[tuple[str, str, float], dict[str, object]], list[tuple[float, float]]]:
    """Valide le profil de flux et construit son index porte/catégorie/fenêtre."""

    parsed: dict[tuple[str, str, float], dict[str, object]] = {}
    windows: set[tuple[float, float]] = set()
    roles = {**{gate: "entry" for gate in ENTRY_GATES}, **{gate: "exit" for gate in EXIT_GATES}}
    for line_number, row in enumerate(rows, start=2):
        gate = row["gate_id"]
        category = row["category"]
        if gate not in roles or row["direction"] != roles[gate]:
            raise ContractInputError(f"Porte ou direction invalide dans flow_profile.csv, ligne {line_number}.")
        if category not in ALL_CATEGORIES:
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

    ordered = _validate_windows(windows)
    expected = {(gate, category, start) for gate in ALL_GATES for category in ALL_CATEGORIES for start, _ in ordered}
    if set(parsed) != expected:
        raise ContractInputError("La grille porte/catégorie/fenêtre de flow_profile.csv est incomplète.")
    return parsed, ordered


def _parse_movement_rows(
    rows: list[dict[str, str]],
    windows: list[tuple[float, float]],
) -> tuple[dict[tuple[str, str, str, float], dict[str, int]], list[dict[str, object]]]:
    """Valide les mouvements classifiables et conserve les visites censurées à part."""

    movements: dict[tuple[str, str, str, float], dict[str, int]] = {}
    unclassified: list[dict[str, object]] = []
    valid_starts = {start: end for start, end in windows}
    for line_number, row in enumerate(rows, start=2):
        start, end = _window_key(row, "movement_profile.csv")
        if start not in valid_starts or not same_number(end, valid_starts[start]):
            raise ContractInputError("Une fenêtre du profil de mouvements ne correspond pas au profil de flux.")
        category = row["category"]
        if category not in ALL_CATEGORIES:
            raise ContractInputError(f"Catégorie inattendue dans movement_profile.csv : {category!r}.")
        count = _as_count(row["count"], "movement_profile.count")
        if row["record_type"] == "movement":
            entry = row["entry_gate"]
            exit_gate = row["exit_gate"]
            if entry not in ENTRY_GATES or exit_gate not in EXIT_GATES or row["visit_status"] != "classifiable":
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
            if row["visit_status"] == "censored_exit" and entry not in ENTRY_GATES:
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
        for entry in ENTRY_GATES
        for exit_gate in EXIT_GATES
        for category in ALL_CATEGORIES
        for start, _ in windows
    }
    if set(movements) != expected:
        raise ContractInputError("La grille des mouvements classifiables est incomplète.")
    # Le même dénominateur est répété pour chaque sortie dans le profil source.
    # Il ne doit donc être vérifié qu'une fois par entrée, catégorie et fenêtre.
    for entry in ENTRY_GATES:
        for category in ALL_CATEGORIES:
            for start, _ in windows:
                values = [movements[(entry, exit_gate, category, start)] for exit_gate in EXIT_GATES]
                denominators = {value["denominator"] for value in values}
                if len(denominators) != 1 or sum(value["count"] for value in values) != values[0]["denominator"]:
                    raise ContractInputError("Les mouvements ne se réconcilient pas avec leur dénominateur répété.")
    return movements, unclassified


def _validate_quality_reconciliations(
    flows: dict[tuple[str, str, float], dict[str, object]],
    windows: list[tuple[float, float]],
    summary: dict,
) -> None:
    by_gate = {
        gate: sum(int(flows[(gate, category, start)]["count"]) for category in ALL_CATEGORIES for start, _ in windows)
        for gate in ALL_GATES
    }
    by_category = {
        category: sum(int(flows[(gate, category, start)]["count"]) for gate in ALL_GATES for start, _ in windows)
        for category in ALL_CATEGORIES
    }
    if summary.get("crossings_by_gate") != by_gate or summary.get("crossings_by_category") != by_category:
        raise ContractInputError("Le profil de flux ne se réconcilie pas avec le bilan d'entrée.")
    if summary.get("counts", {}).get("crossings") != sum(by_gate.values()):
        raise ContractInputError("Le total de franchissements du bilan d'entrée est incohérent.")


def load_traffic_inputs(profile_dir: str | Path, coverage_path: str | Path) -> TrafficInputs:
    """Lit et valide les profils de flux, les mouvements et leur provenance."""

    source_dir = Path(profile_dir).resolve()
    coverage_file = Path(coverage_path).resolve()
    hashes = _verify_hashes(source_dir, coverage_file)
    manifest = _read_json(source_dir / "manifest.json", "Manifeste source")
    sector_config = _read_json(source_dir / "sector_config.json", "Configuration du secteur")
    summary = _read_json(source_dir / "quality_summary.json", "Bilan des profils")
    coverage = _read_json(coverage_file, "Couverture temporelle")
    _validate_identity(manifest, sector_config, summary, coverage)

    flow_rows = _read_csv(source_dir / "flow_profile.csv", FLOW_FIELDS, "flow_profile.csv")
    movement_rows = _read_csv(source_dir / "movement_profile.csv", MOVEMENT_FIELDS, "movement_profile.csv")
    flows, windows = _parse_flow_rows(flow_rows)
    movements, unclassified = _parse_movement_rows(movement_rows, windows)
    _validate_quality_reconciliations(flows, windows, summary)
    return TrafficInputs(hashes, manifest, coverage, summary, flows, windows, movements, unclassified)
