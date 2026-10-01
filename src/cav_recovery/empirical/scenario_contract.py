"""Construction du contrat empirique CGR-E03 depuis les agrégats CGR-E02."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "CGR-E03-1"
EXPECTED_INPUT_SHA256 = {
    "manifest.json": "d6b659330d708c2d766088750a7e27ef44a25b435531a206d04d799e774da1af",
    "sector_config.json": "c62bd8543bfbe655e78ae067bf3fe8ae0f6bee14cc19ccda2deb50e8ca6300d2",
    "quality_summary.json": "22039ca750de97e74f0c6412f59f34514232d18fb7491c6e63686648986021b8",
    "flow_profile.csv": "b33be95fe52a1968a01f9b3e83e1423a8c883883a9e91287d610ee2edf56e2a8",
    "movement_profile.csv": "2fc9915f7eb4433873d8e7de37e413b6dfb962abf8769c57b27d50c45a442dda",
    "coverage.json": "c856992dc1733bd0aa131787edf8609d44a9928343b1919c1bec00285d02038d",
}

SOURCE_CATEGORIES = ("Car", "Taxi")
ALL_CATEGORIES = ("Car", "Taxi", "Motorcycle", "Bus", "Medium Vehicle", "Heavy Vehicle")
ENTRY_GATES = ("W23183369_IN", "W284241336_IN")
EXIT_GATES = ("W23183369_OUT", "W284241336_OUT")
ALL_GATES = ENTRY_GATES + EXIT_GATES
LOAD_LEVEL_ORDER = ("LOW", "MID", "HIGH")
GROUP_SIZES = (4, 5, 4)
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
OUTPUTS = ("regime_profile.csv", "empirical_contract.json", "quality_summary.json", "regime_report.md")
PROVENANCE_LIMITATIONS = (
    "La référence TEST de CGR-E02 a été révisée après une première évaluation ; le score final ne provient donc pas d'un jeu de vérité resté totalement intact. Cette réserve de provenance n'invalide pas les agrégats E02 audités.",
    "Le profil nominal CGR-E02 a été produit avant le commit public final 2212be0 et n'a pas été rejoué sous ce commit ; la correction ultérieure du cas exit_time_s == 0.0 reste une réserve de reproductibilité, sans modifier les agrégats E02 audités.",
)


class ContractInputError(ValueError):
    """Signale une entrée CGR-E02 incompatible avec le contrat E03."""


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


def _same(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=0.0, abs_tol=1e-9)


def _verify_hashes(cgr_e02_dir: Path, coverage_path: Path) -> dict[str, str]:
    paths = {
        "manifest.json": cgr_e02_dir / "manifest.json",
        "sector_config.json": cgr_e02_dir / "sector_config.json",
        "quality_summary.json": cgr_e02_dir / "quality_summary.json",
        "flow_profile.csv": cgr_e02_dir / "flow_profile.csv",
        "movement_profile.csv": cgr_e02_dir / "movement_profile.csv",
        "coverage.json": coverage_path,
    }
    actual: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ContractInputError(f"Artefact CGR-E02 obligatoire absent : {name}")
        actual[name] = _sha256(path)
        if actual[name] != EXPECTED_INPUT_SHA256[name]:
            raise ContractInputError(f"Empreinte SHA-256 incompatible pour {name}.")
    return actual


def _validate_identity(manifest: dict, sector_config: dict, summary: dict, coverage: dict) -> None:
    if manifest.get("schema_version") != "CGR-E02-1" or manifest.get("execution", {}).get("status") != "succeeded":
        raise ContractInputError("Le manifeste CGR-E02 n'identifie pas une exécution réussie compatible.")
    if manifest.get("configuration") != sector_config:
        raise ContractInputError("La configuration CGR-E02 ne correspond pas à celle du manifeste.")
    if summary.get("schema_version") != "CGR-E02-1" or summary.get("execution", {}).get("status") != "succeeded":
        raise ContractInputError("Le bilan CGR-E02 est incompatible ou incomplet.")
    if summary.get("method") != "oriented_finite_virtual_gates":
        raise ContractInputError("La méthode CGR-E02 n'est pas celle attendue.")
    upstream_inputs = manifest.get("inputs")
    required_provenance = {
        "cgr_e01_manifest_sha256",
        "geometry_sha256",
        "locally_computed_cgr_e01_export_sha256",
        "runtime_config_sha256",
        "sector_seed_sha256",
    }
    if not isinstance(upstream_inputs, dict) or not required_provenance.issubset(upstream_inputs):
        raise ContractInputError("La provenance amont CGR-E01/CGR-E02 est incomplète.")
    if not isinstance(manifest.get("software"), dict):
        raise ContractInputError("L'identité du code CGR-E02 est absente du manifeste.")

    source = manifest.get("source", {})
    expected_source = {
        "filename": "20181024_d3_0830_0900.csv",
        "sha256": "17970bd3f8e167df3ef54792e8fb7f874f89a9a0aceb14571321e9a48fbeea0d",
        "size_bytes": 199_512_534,
    }
    if source != expected_source:
        raise ContractInputError("La source pNEUMA déclarée n'est pas la source canonique CGR-E03.")

    seed = sector_config.get("sector_seed", {})
    sector = seed.get("sector", {})
    if sector.get("center_osm_node_id") != 250691665 or sector.get("id") != "PNEUMA_D3_NODE_250691665":
        raise ContractInputError("Le secteur CGR-E02 n'est pas C3 au nœud OSM 250691665.")
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
        raise ContractInputError("Le calendrier d'agrégation CGR-E02 est incompatible.")
    expected_intervals = {gate: [[0.0, 802.8]] for gate in ALL_GATES}
    if sector_config.get("coverage") != expected_intervals:
        raise ContractInputError("La couverture de la configuration ne correspond pas à [0,0 ; 802,8).")

    if coverage.get("schema") != "CGR-E02-coverage-evidence-2" or coverage.get("status") != "known":
        raise ContractInputError("La preuve de couverture CGR-E02 n'est pas établie.")
    if coverage.get("intervals") != [[0.0, 802.8]]:
        raise ContractInputError("L'intervalle de couverture CGR-E02 est incompatible.")
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
    ordered = sorted(set(windows))
    if len(ordered) != 14:
        raise ContractInputError("CGR-E03 exige exactement 14 fenêtres CGR-E02.")
    for index, (start, end) in enumerate(ordered):
        expected_start = index * 60.0
        expected_end = min((index + 1) * 60.0, 802.8)
        if not _same(start, expected_start) or not _same(end, expected_end):
            raise ContractInputError("Les fenêtres CGR-E02 ne suivent pas le calendrier [0,0 ; 802,8).")
    return ordered


def _parse_flow_rows(rows: list[dict[str, str]]) -> tuple[dict[tuple[str, str, float], dict[str, object]], list[tuple[float, float]]]:
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
        if row["coverage_status"] != "known" or not _same(exposure, end - start):
            raise ContractInputError("Chaque ligne du profil doit avoir une exposition connue égale à sa fenêtre.")
        if raw != count:
            raise ContractInputError("Un passage brut se trouve hors de l'exposition canonique.")
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
    movements: dict[tuple[str, str, str, float], dict[str, int]] = {}
    unclassified: list[dict[str, object]] = []
    valid_starts = {start: end for start, end in windows}
    for line_number, row in enumerate(rows, start=2):
        start, end = _window_key(row, "movement_profile.csv")
        if start not in valid_starts or not _same(end, valid_starts[start]):
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
            else:
                if not _same(_as_float(proportion, "proportion"), count / denominator):
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
            unclassified.append({"entry_gate": entry, "category": category, "start": start, "count": count, "status": row["visit_status"]})
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
    for entry in ENTRY_GATES:
        for category in ALL_CATEGORIES:
            for start, _ in windows:
                values = [movements[(entry, exit_gate, category, start)] for exit_gate in EXIT_GATES]
                denominators = {value["denominator"] for value in values}
                if len(denominators) != 1 or sum(value["count"] for value in values) != values[0]["denominator"]:
                    raise ContractInputError("Les mouvements ne se réconcilient pas avec leur dénominateur répété.")
    return movements, unclassified


def _build_window_rows(
    flows: dict[tuple[str, str, float], dict[str, object]],
    windows: list[tuple[float, float]],
) -> list[dict[str, object]]:
    result = []
    for start, end in windows:
        by_gate = {
            gate: sum(int(flows[(gate, category, start)]["count"]) for category in SOURCE_CATEGORIES)
            for gate in ENTRY_GATES
        }
        duration = round(end - start, 10)
        result.append({
            "window_start_s": start,
            "window_end_s": end,
            "duration_s": duration,
            "eligible_for_stratification": _same(duration, 60.0),
            "passenger_in_w23183369": by_gate[ENTRY_GATES[0]],
            "passenger_in_w284241336": by_gate[ENTRY_GATES[1]],
            "passenger_in_total": sum(by_gate.values()),
            "regime_id": None,
        })
    return result


def _assign_load_levels(window_rows: list[dict[str, object]]) -> None:
    eligible = [row for row in window_rows if row["eligible_for_stratification"]]
    if len(eligible) != sum(GROUP_SIZES):
        raise ContractInputError("La règle 4/5/4 exige exactement 13 fenêtres complètes.")
    ranked = sorted(eligible, key=lambda row: (int(row["passenger_in_total"]), float(row["window_start_s"])))
    offset = 0
    for load_level, size in zip(LOAD_LEVEL_ORDER, GROUP_SIZES):
        for row in ranked[offset:offset + size]:
            row["regime_id"] = load_level
        offset += size


def _aggregate_movements(
    member_windows: set[float],
    movements: dict[tuple[str, str, str, float], dict[str, int]],
) -> tuple[dict[str, dict[str, int]], dict[str, int], dict[str, dict[str, float | None]]]:
    counts: dict[str, dict[str, int]] = {}
    denominators: dict[str, int] = {}
    probabilities: dict[str, dict[str, float | None]] = {}
    for entry in ENTRY_GATES:
        counts[entry] = {
            exit_gate: sum(
                movements[(entry, exit_gate, category, start)]["count"]
                for category in SOURCE_CATEGORIES
                for start in member_windows
            )
            for exit_gate in EXIT_GATES
        }
        denominator = sum(
            movements[(entry, EXIT_GATES[0], category, start)]["denominator"]
            for category in SOURCE_CATEGORIES
            for start in member_windows
        )
        if sum(counts[entry].values()) != denominator:
            raise ContractInputError(f"Les mouvements passenger_CAV ne se réconcilient pas pour {entry}.")
        denominators[entry] = denominator
        probabilities[entry] = {
            exit_gate: None if denominator == 0 else counts[entry][exit_gate] / denominator
            for exit_gate in EXIT_GATES
        }
    return counts, denominators, probabilities


def _build_load_levels(
    window_rows: list[dict[str, object]],
    movements: dict[tuple[str, str, str, float], dict[str, int]],
    unclassified: list[dict[str, object]],
) -> list[dict[str, object]]:
    load_levels = []
    for load_level in LOAD_LEVEL_ORDER:
        members = sorted((row for row in window_rows if row["regime_id"] == load_level), key=lambda row: float(row["window_start_s"]))
        starts = {float(row["window_start_s"]) for row in members}
        exposure = sum(float(row["duration_s"]) for row in members)
        entry_counts = {
            ENTRY_GATES[0]: sum(int(row["passenger_in_w23183369"]) for row in members),
            ENTRY_GATES[1]: sum(int(row["passenger_in_w284241336"]) for row in members),
        }
        total = sum(entry_counts.values())
        movement_counts, denominators, probabilities = _aggregate_movements(starts, movements)
        censored_exit = sum(
            int(row["count"])
            for row in unclassified
            if row["status"] == "censored_exit" and row["category"] in SOURCE_CATEGORIES and float(row["start"]) in starts
        )
        classifiable = sum(denominators.values())
        if total != classifiable + censored_exit:
            raise ContractInputError(f"Les entrées, mouvements et censures ne se réconcilient pas pour {load_level}.")
        load_levels.append({
            "regime_id": load_level,
            "member_windows": [[row["window_start_s"], row["window_end_s"]] for row in members],
            "exposure_s": exposure,
            "passenger_entry_counts": entry_counts,
            "passenger_entry_total": total,
            "passenger_entry_rates_veh_per_h": {gate: 3600.0 * count / exposure for gate, count in entry_counts.items()},
            "passenger_entry_shares": {gate: None if total == 0 else count / total for gate, count in entry_counts.items()},
            "combined_reference_flow_veh_per_h": 3600.0 * total / exposure,
            "passenger_movement_counts": movement_counts,
            "passenger_movement_denominators": denominators,
            "passenger_movement_probabilities": probabilities,
            "classifiable_visits": classifiable,
            "censored_exit": censored_exit,
        })
    return load_levels


def _build_composition(
    flows: dict[tuple[str, str, float], dict[str, object]],
    windows: list[tuple[float, float]],
    movements: dict[tuple[str, str, str, float], dict[str, int]],
    unclassified: list[dict[str, object]],
    summary: dict,
) -> dict[str, object]:
    complete_starts = [start for start, end in windows if _same(end - start, 60.0)]
    categories: dict[str, dict[str, object]] = {}
    for category in ALL_CATEGORIES:
        by_gate = {
            gate: sum(int(flows[(gate, category, start)]["count"]) for start, _ in windows)
            for gate in ENTRY_GATES
        }
        total = sum(by_gate.values())
        classifiable = sum(
            movements[(entry, EXIT_GATES[0], category, start)]["denominator"]
            for entry in ENTRY_GATES
            for start, _ in windows
        )
        censored_exit = sum(
            int(row["count"])
            for row in unclassified
            if row["category"] == category and row["status"] == "censored_exit"
        )
        if total != classifiable + censored_exit:
            raise ContractInputError(f"La composition ne se réconcilie pas pour {category}.")
        complete_counts = [sum(int(flows[(gate, category, start)]["count"]) for gate in ENTRY_GATES) for start in complete_starts]
        categories[category] = {
            "entry_counts": by_gate,
            "observed_entries": total,
            "classifiable_visits": classifiable,
            "censored_exit": censored_exit,
            "complete_window_entry_min": min(complete_counts),
            "complete_window_entry_max": max(complete_counts),
        }

    total_entries = sum(int(value["observed_entries"]) for value in categories.values())
    total_classifiable = sum(int(value["classifiable_visits"]) for value in categories.values())
    total_censored_exit = sum(int(value["censored_exit"]) for value in categories.values())
    censored_entry = sum(int(row["count"]) for row in unclassified if row["status"] == "censored_entry")
    expected_statuses = summary.get("counts", {}).get("visits_by_status", {})
    expected = {
        "classifiable": total_classifiable,
        "censored_exit": total_censored_exit,
        "censored_entry": censored_entry,
    }
    if expected_statuses != expected or total_entries != total_classifiable + total_censored_exit:
        raise ContractInputError("Les statuts de visite ne se réconcilient pas avec les entrées observées.")
    for value in categories.values():
        value["entry_share"] = None if total_entries == 0 else int(value["observed_entries"]) / total_entries
    return {
        "categories": categories,
        "total_observed_entries": total_entries,
        "classifiable_visits": total_classifiable,
        "censored_exit": total_censored_exit,
        "censored_entry": censored_entry,
        "passenger_entries": sum(int(categories[category]["observed_entries"]) for category in SOURCE_CATEGORIES),
    }


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
        raise ContractInputError("Le profil de flux ne se réconcilie pas avec le bilan CGR-E02.")
    if summary.get("counts", {}).get("crossings") != sum(by_gate.values()):
        raise ContractInputError("Le total de franchissements CGR-E02 est incohérent.")


def _contract_document(
    hashes: dict[str, str],
    manifest: dict,
    coverage: dict,
    window_rows: list[dict[str, object]],
    load_levels: list[dict[str, object]],
    composition: dict[str, object],
) -> dict[str, object]:
    complete = [row for row in window_rows if row["eligible_for_stratification"]]
    partial = [row for row in window_rows if not row["eligible_for_stratification"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "inputs": {
            "cgr_e02_artifacts_sha256": dict(sorted(hashes.items())),
            "upstream_provenance": manifest["inputs"],
            "upstream_software": manifest["software"],
        },
        "method": {
            "source_categories": list(SOURCE_CATEGORIES),
            "complete_window_s": 60,
            "eligible_window_count": len(complete),
            "partial_window_count": len(partial),
            "regime_order": list(LOAD_LEVEL_ORDER),
            "group_sizes": list(GROUP_SIZES),
            "primary_variable": "N_passenger_in",
            "tie_break": "window_start_s_ascending",
            "upstream_method": "oriented_finite_virtual_gates",
        },
        "empirical_context": {
            "source": manifest["source"],
            "sector": {"id": "C3", "osm_node_id": 250691665, "entry_gates": list(ENTRY_GATES), "exit_gates": list(EXIT_GATES)},
            "observation_interval_s": [0.0, 802.8],
            "interval_convention": "[a,b)",
            "coverage": {"status": coverage["status"], "intervals": coverage["intervals"]},
            "categories": list(ALL_CATEGORIES),
            "composition_observed": composition,
            "epistemic_status": {
                "observed_counts_and_censoring": "OBSERVÉ / issu du pipeline empirique validé",
                "sector_and_gates": "DÉRIVÉ / ESTIMÉ",
                "shares_and_rates": "DÉRIVÉ / ESTIMÉ",
            },
        },
        "passenger_cav_contract": {
            "population_id": "passenger_CAV",
            "source_categories": list(SOURCE_CATEGORIES),
            "epistemic_status": "DÉRIVÉ / ESTIMÉ",
            "regimes": load_levels,
            "complete_window_passenger_entries": sum(int(row["passenger_in_total"]) for row in complete),
            "all_window_passenger_entries": sum(int(row["passenger_in_total"]) for row in window_rows),
            "rates_interpretation": "empirical_passage_references_not_simulation_arrival_rates",
        },
        "assumptions": {
            "epistemic_status": "SUPPOSÉ / EXPÉRIMENTAL",
            "items_without_numeric_values": [
                "Les véhicules du futur scénario principal seront représentés comme des CAV.",
                "La dynamique longitudinale et le processus exact des arrivées restent à définir.",
                "Les feux, capacités, incidents et actions de récupération restent à définir.",
            ],
        },
        "limitations": [
            "LOW, MID et HIGH sont des niveaux relatifs de charge observée, pas des états de congestion ni des capacités.",
            "Les taux sont des références empiriques de passages, pas des taux d'insertion ou des lois d'arrivée SUMO.",
            "Le contrat passenger_CAV ne représente que Car et Taxi ; les quatre autres catégories restent décrites mais ne sont pas converties.",
            "Les mouvements sont locaux, conditionnels aux visites complètes et ne sont pas des origines-destinations réelles.",
            "La caractérisation repose sur une seule séquence courte et aucune période empirique indépendante n'est réservée.",
            "La couverture est inférée sans inspection du masque vidéo brut et aucun second secteur de transfert n'est validé.",
            *PROVENANCE_LIMITATIONS,
        ],
    }


def _quality_document(window_rows: list[dict[str, object]], load_levels: list[dict[str, object]], composition: dict[str, object], hashes: dict[str, str]) -> dict[str, object]:
    complete = [row for row in window_rows if row["eligible_for_stratification"]]
    partial = [row for row in window_rows if not row["eligible_for_stratification"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "inputs_sha256": dict(sorted(hashes.items())),
        "windows": {"total": len(window_rows), "complete": len(complete), "partial": len(partial), "group_sizes": {level: size for level, size in zip(LOAD_LEVEL_ORDER, GROUP_SIZES)}},
        "passenger_entries": {
            "complete_windows": sum(int(row["passenger_in_total"]) for row in complete),
            "partial_windows": sum(int(row["passenger_in_total"]) for row in partial),
            "all_windows": sum(int(row["passenger_in_total"]) for row in window_rows),
        },
        "regimes": {
            load_level["regime_id"]: {
                "passenger_entries": load_level["passenger_entry_total"],
                "combined_reference_flow_veh_per_h": load_level["combined_reference_flow_veh_per_h"],
                "classifiable_visits": load_level["classifiable_visits"],
                "censored_exit": load_level["censored_exit"],
            }
            for load_level in load_levels
        },
        "composition": {
            "observed_entries": composition["total_observed_entries"],
            "classifiable_visits": composition["classifiable_visits"],
            "censored_exit": composition["censored_exit"],
            "censored_entry": composition["censored_entry"],
            "categories": len(composition["categories"]),
        },
        "checks": {
            "canonical_inputs": "pass",
            "identity_and_coverage": "pass",
            "window_partition_4_5_4": "pass",
            "flow_reconciliation": "pass",
            "movement_denominators": "pass",
            "censoring_reconciliation": "pass",
            "excluded_categories_not_converted": "pass",
        },
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "window_start_s",
        "window_end_s",
        "duration_s",
        "eligible_for_stratification",
        "passenger_in_w23183369",
        "passenger_in_w284241336",
        "passenger_in_total",
        "regime_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["eligible_for_stratification"] = str(bool(row["eligible_for_stratification"])).lower()
            output["regime_id"] = row["regime_id"] or ""
            writer.writerow(output)


def _report(load_levels: list[dict[str, object]], composition: dict[str, object]) -> str:
    lines = [
        "# Contrat empirique des scénarios — CGR-E03",
        "",
        "## Méthode",
        "",
        "Les treize fenêtres complètes sont classées par entrées Car+Taxi croissantes, puis par temps de début. Les groupes LOW/MID/HIGH contiennent 4/5/4 fenêtres. La fenêtre partielle reste hors stratification.",
        "",
        "Ces groupes décrivent des niveaux relatifs de charge observée. Ils ne démontrent ni congestion, ni capacité, ni gridlock.",
        "",
        "## Contrat passenger_CAV",
        "",
        "| Niveau | Fenêtres | Entrées | Exposition par porte (s) | Référence de passages (véh/h) | Classifiables | Censurées en sortie |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for load_level in load_levels:
        lines.append(
            f"| {load_level['regime_id']} | {len(load_level['member_windows'])} | {load_level['passenger_entry_total']} | "
            f"{load_level['exposure_s']} | {load_level['combined_reference_flow_veh_per_h']} | "
            f"{load_level['classifiable_visits']} | {load_level['censored_exit']} |"
        )
    lines.extend([
        "",
        "Les mouvements sont calculés sur les seuls mouvements Car+Taxi complets, avec les dénominateurs poolés. Les censures ne deviennent jamais des mouvements.",
        "",
        "## Mouvements passenger_CAV",
        "",
        "| Niveau | Entrée | Vers W23183369_OUT | Vers W284241336_OUT | Dénominateur |",
        "|---|---|---:|---:|---:|",
    ])
    for load_level in load_levels:
        for entry in ENTRY_GATES:
            counts = load_level["passenger_movement_counts"][entry]
            denominator = load_level["passenger_movement_denominators"][entry]
            lines.append(
                f"| {load_level['regime_id']} | {entry} | {counts[EXIT_GATES[0]]} | "
                f"{counts[EXIT_GATES[1]]} | {denominator} |"
            )
    lines.extend([
        "",
        "## Contexte empirique",
        "",
        f"Les six catégories totalisent {composition['total_observed_entries']} entrées observées : {composition['classifiable_visits']} visites classifiables et {composition['censored_exit']} censurées en sortie. Les {composition['censored_entry']} censures d'entrée restent séparées.",
        "",
        "| Catégorie | Entrées observées | Classifiables | Censurées en sortie |",
        "|---|---:|---:|---:|",
    ])
    for category in ALL_CATEGORIES:
        values = composition["categories"][category]
        lines.append(
            f"| {category} | {values['observed_entries']} | {values['classifiable_visits']} | {values['censored_exit']} |"
        )
    lines.extend([
        "",
        "Motorcycle, Bus, Medium Vehicle et Heavy Vehicle restent documentés, sans conversion en voitures. Seuls Car et Taxi alimentent le contrat principal passenger_CAV.",
        "",
        "## Statut des informations",
        "",
        "### OBSERVÉ / issu du pipeline empirique",
        "",
        "Les catégories pNEUMA, les comptages de passages, les mouvements classifiables, les censures et l'exposition selon le contrat E02 proviennent du pipeline empirique validé. Les franchissements et mouvements sont extraits des trajectoires ; ils ne sont pas annotés directement sur une vidéo brute.",
        "",
        "### DÉRIVÉ / ESTIMÉ",
        "",
        "Les portes et la configuration C3, les taux en véh/h, les parts d'entrée, les probabilités de mouvements, N_passenger_in, les niveaux LOW/MID/HIGH et l'agrégation Car+Taxi sont dérivés des données validées.",
        "",
        "### SUPPOSÉ / EXPÉRIMENTAL",
        "",
        "La population simulée principale en CAV et ses paramètres restent différés : dynamique longitudinale, car-following, dimensions, accélération et décélération, processus d'arrivée, feux, capacités, incidents et actions de récupération. Aucune valeur nouvelle n'est fixée ici.",
        "",
        "## Limites",
        "",
        "Les taux sont des références empiriques de passages. Leur transformation en arrivées ou insertions SUMO n'est pas définie dans CGR-E03. La caractérisation porte sur un seul secteur et une seule séquence courte, avec une couverture inférée sans inspection vidéo brute.",
        "",
    ])
    lines.extend(f"- {limitation}" for limitation in PROVENANCE_LIMITATIONS)
    lines.append("")
    return "\n".join(lines)


def build_empirical_contract(cgr_e02_dir: str | Path, coverage_path: str | Path, output_dir: str | Path) -> dict[str, object]:
    """Valide CGR-E02 puis écrit le contrat descriptif CGR-E03.

    Le dossier de sortie doit être nouveau ou vide. Les empreintes canoniques,
    l'identité du secteur et toutes les réconciliations sont contrôlées avant
    qu'un artefact ne soit publié dans ce dossier.
    """
    source_dir = Path(cgr_e02_dir).resolve()
    coverage_file = Path(coverage_path).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        if not destination.is_dir():
            raise ContractInputError("Le chemin de sortie CGR-E03 existe et n'est pas un dossier.")
        if any(destination.iterdir()):
            raise ContractInputError("Le dossier de sortie CGR-E03 existe déjà et n'est pas vide.")

    hashes = _verify_hashes(source_dir, coverage_file)
    manifest = _read_json(source_dir / "manifest.json", "Manifeste CGR-E02")
    sector_config = _read_json(source_dir / "sector_config.json", "Configuration CGR-E02")
    summary_e02 = _read_json(source_dir / "quality_summary.json", "Bilan CGR-E02")
    coverage = _read_json(coverage_file, "Couverture CGR-E02")
    _validate_identity(manifest, sector_config, summary_e02, coverage)

    flow_rows = _read_csv(source_dir / "flow_profile.csv", FLOW_FIELDS, "flow_profile.csv")
    movement_rows = _read_csv(source_dir / "movement_profile.csv", MOVEMENT_FIELDS, "movement_profile.csv")
    flows, windows = _parse_flow_rows(flow_rows)
    movements, unclassified = _parse_movement_rows(movement_rows, windows)
    _validate_quality_reconciliations(flows, windows, summary_e02)

    window_rows = _build_window_rows(flows, windows)
    _assign_load_levels(window_rows)
    load_levels = _build_load_levels(window_rows, movements, unclassified)
    composition = _build_composition(flows, windows, movements, unclassified, summary_e02)
    contract = _contract_document(hashes, manifest, coverage, window_rows, load_levels, composition)
    quality = _quality_document(window_rows, load_levels, composition, hashes)

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".cgr-e03-", dir=destination.parent))
    destination_was_empty = destination.exists()
    try:
        _write_csv(stage / "regime_profile.csv", window_rows)
        (stage / "empirical_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (stage / "quality_summary.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (stage / "regime_report.md").write_text(_report(load_levels, composition), encoding="utf-8")
        if destination_was_empty:
            destination.rmdir()
        try:
            os.replace(stage, destination)
        except Exception:
            if destination_was_empty and not destination.exists():
                destination.mkdir()
            raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return quality
