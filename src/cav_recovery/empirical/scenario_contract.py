"""Construction d'un contrat de scénario à partir de profils de trafic validés."""

from __future__ import annotations

from pathlib import Path

from ..canonical_scenario import CANONICAL_SCENARIO, CanonicalScenario

from .contract_files import validate_output_directory, write_contract_files
from .traffic_inputs import (
    ContractInputError,
    load_traffic_inputs,
    same_number,
)


SCHEMA_VERSION = CANONICAL_SCENARIO.contract.schema_version
SOURCE_CATEGORIES = CANONICAL_SCENARIO.contract.source_categories
LOAD_LEVEL_ORDER = CANONICAL_SCENARIO.contract.load_levels
GROUP_SIZES = CANONICAL_SCENARIO.contract.group_sizes
PROVENANCE_LIMITATIONS = (
    "La référence TEST utilisée pour valider l'extraction a été révisée après une première évaluation ; "
    "le score final ne provient donc pas d'un jeu de vérité resté totalement intact. Cette réserve de "
    "provenance n'invalide pas les agrégats audités.",
    "Le profil nominal a été produit avant le commit public final 2212be0 et n'a pas été rejoué sous "
    "ce commit ; la correction ultérieure du cas exit_time_s == 0.0 reste une réserve de "
    "reproductibilité, sans modifier les agrégats audités.",
)


def _build_window_rows(
    flows: dict[tuple[str, str, float], dict[str, object]],
    windows: list[tuple[float, float]],
    config: CanonicalScenario,
) -> list[dict[str, object]]:
    """Calcule la charge Car+Taxi de chaque fenêtre aux deux portes d'entrée.

    La dernière fenêtre reste dans le profil, mais seule une fenêtre complète de
    60 secondes peut participer au classement des niveaux de charge.
    """
    entry_gates = config.sector.entry_gates
    source_categories = config.contract.source_categories

    result = []
    for start, end in windows:
        by_gate = {
            gate: sum(int(flows[(gate, category, start)]["count"]) for category in source_categories)
            for gate in entry_gates
        }
        duration = round(end - start, 10)
        result.append({
            "window_start_s": start,
            "window_end_s": end,
            "duration_s": duration,
            "eligible_for_stratification": same_number(duration, config.sector.window_s),
            "passenger_in_w23183369": by_gate[entry_gates[0]],
            "passenger_in_w284241336": by_gate[entry_gates[1]],
            "passenger_in_total": sum(by_gate.values()),
            "regime_id": None,
        })
    return result


def _assign_load_levels(window_rows: list[dict[str, object]], config: CanonicalScenario) -> None:
    """Classe les fenêtres complètes par charge croissante selon la règle 4/5/4.

    Le temps de début départage deux charges égales afin que le résultat reste
    déterministe.
    """

    eligible = [row for row in window_rows if row["eligible_for_stratification"]]
    if len(eligible) != sum(config.contract.group_sizes):
        raise ContractInputError("La règle 4/5/4 exige exactement 13 fenêtres complètes.")
    ranked = sorted(
        eligible,
        key=lambda row: (int(row["passenger_in_total"]), float(row["window_start_s"])),
    )
    offset = 0
    for load_level, size in zip(config.contract.load_levels, config.contract.group_sizes):
        for row in ranked[offset:offset + size]:
            row["regime_id"] = load_level
        offset += size


def _aggregate_movements(
    member_windows: set[float],
    movements: dict[tuple[str, str, str, float], dict[str, int]],
    config: CanonicalScenario,
) -> tuple[dict[str, dict[str, int]], dict[str, int], dict[str, dict[str, float | None]]]:
    """Agrège les mouvements Car+Taxi des fenêtres appartenant à un niveau."""

    entry_gates, exit_gates = config.sector.entry_gates, config.sector.exit_gates
    source_categories = config.contract.source_categories
    counts: dict[str, dict[str, int]] = {}
    denominators: dict[str, int] = {}
    probabilities: dict[str, dict[str, float | None]] = {}
    for entry in entry_gates:
        counts[entry] = {
            exit_gate: sum(
                movements[(entry, exit_gate, category, start)]["count"]
                for category in source_categories
                for start in member_windows
            )
            for exit_gate in exit_gates
        }
        # Le profil répète le même dénominateur sur les deux sorties. Une seule
        # sortie est donc lue pour éviter de compter deux fois les visites.
        denominator = sum(
            movements[(entry, exit_gates[0], category, start)]["denominator"]
            for category in source_categories
            for start in member_windows
        )
        if sum(counts[entry].values()) != denominator:
            raise ContractInputError(f"Les mouvements passenger_CAV ne se réconcilient pas pour {entry}.")
        denominators[entry] = denominator
        probabilities[entry] = {
            exit_gate: None if denominator == 0 else counts[entry][exit_gate] / denominator
            for exit_gate in exit_gates
        }
    return counts, denominators, probabilities


def _build_load_levels(
    window_rows: list[dict[str, object]],
    movements: dict[tuple[str, str, str, float], dict[str, int]],
    unclassified: list[dict[str, object]],
    config: CanonicalScenario,
) -> list[dict[str, object]]:
    """Construit les comptes, taux et mouvements de chaque niveau de charge."""

    entry_gates = config.sector.entry_gates
    source_categories = config.contract.source_categories
    load_levels = []
    for load_level in config.contract.load_levels:
        members = sorted(
            (row for row in window_rows if row["regime_id"] == load_level),
            key=lambda row: float(row["window_start_s"]),
        )
        starts = {float(row["window_start_s"]) for row in members}
        exposure = sum(float(row["duration_s"]) for row in members)
        entry_counts = {
            entry_gates[0]: sum(int(row["passenger_in_w23183369"]) for row in members),
            entry_gates[1]: sum(int(row["passenger_in_w284241336"]) for row in members),
        }
        total = sum(entry_counts.values())
        movement_counts, denominators, probabilities = _aggregate_movements(starts, movements, config)
        # Une sortie censurée reste une entrée observée, mais ne devient jamais
        # un mouvement certain dans les probabilités ci-dessus.
        censored_exit = sum(
            int(row["count"])
            for row in unclassified
            if row["status"] == "censored_exit"
            and row["category"] in source_categories
            and float(row["start"]) in starts
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
            "passenger_entry_rates_veh_per_h": {
                gate: 3600.0 * count / exposure for gate, count in entry_counts.items()
            },
            "passenger_entry_shares": {
                gate: None if total == 0 else count / total for gate, count in entry_counts.items()
            },
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
    config: CanonicalScenario,
) -> dict[str, object]:
    """Résume les six catégories sans convertir les véhicules non retenus."""

    entry_gates, exit_gates = config.sector.entry_gates, config.sector.exit_gates
    source_categories = config.contract.source_categories
    complete_starts = [start for start, end in windows if same_number(end - start, config.sector.window_s)]
    categories: dict[str, dict[str, object]] = {}
    for category in config.sector.categories:
        by_gate = {
            gate: sum(int(flows[(gate, category, start)]["count"]) for start, _ in windows)
            for gate in entry_gates
        }
        total = sum(by_gate.values())
        classifiable = sum(
            movements[(entry, exit_gates[0], category, start)]["denominator"]
            for entry in entry_gates
            for start, _ in windows
        )
        censored_exit = sum(
            int(row["count"])
            for row in unclassified
            if row["category"] == category and row["status"] == "censored_exit"
        )
        if total != classifiable + censored_exit:
            raise ContractInputError(f"La composition ne se réconcilie pas pour {category}.")
        complete_counts = [
            sum(int(flows[(gate, category, start)]["count"]) for gate in entry_gates)
            for start in complete_starts
        ]
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
        "passenger_entries": sum(int(categories[category]["observed_entries"]) for category in source_categories),
    }


def _contract_document(
    hashes: dict[str, str],
    manifest: dict,
    coverage: dict,
    window_rows: list[dict[str, object]],
    load_levels: list[dict[str, object]],
    composition: dict[str, object],
    config: CanonicalScenario,
) -> dict[str, object]:
    complete = [row for row in window_rows if row["eligible_for_stratification"]]
    partial = [row for row in window_rows if not row["eligible_for_stratification"]]
    return {
        "schema_version": config.contract.schema_version,
        "status": "complete",
        "inputs": {
            "input_artifacts_sha256": dict(sorted(hashes.items())),
            "upstream_provenance": manifest["inputs"],
            "upstream_software": manifest["software"],
        },
        "method": {
            "source_categories": list(config.contract.source_categories),
            "complete_window_s": config.sector.window_s,
            "eligible_window_count": len(complete),
            "partial_window_count": len(partial),
            "regime_order": list(config.contract.load_levels),
            "group_sizes": list(config.contract.group_sizes),
            "primary_variable": "N_passenger_in",
            "tie_break": "window_start_s_ascending",
            "upstream_method": config.empirical.extraction_method,
        },
        "empirical_context": {
            "source": manifest["source"],
            "sector": {
                "id": config.sector.name,
                "osm_node_id": config.sector.node_id,
                "entry_gates": list(config.sector.entry_gates),
                "exit_gates": list(config.sector.exit_gates),
            },
            "observation_interval_s": list(config.sector.observation_interval_s),
            "interval_convention": config.sector.interval_convention,
            "coverage": {"status": coverage["status"], "intervals": coverage["intervals"]},
            "categories": list(config.sector.categories),
            "composition_observed": composition,
            "epistemic_status": {
                "observed_counts_and_censoring": "OBSERVÉ / issu du pipeline empirique validé",
                "sector_and_gates": "DÉRIVÉ / ESTIMÉ",
                "shares_and_rates": "DÉRIVÉ / ESTIMÉ",
            },
        },
        "passenger_cav_contract": {
            "population_id": config.simulation.vehicle_type["id"],
            "source_categories": list(config.contract.source_categories),
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
            "LOW, MID et HIGH sont des niveaux relatifs de charge observée, pas des états de "
            "congestion ni des capacités.",
            "Les taux sont des références empiriques de passages, pas des taux d'insertion ou des "
            "lois d'arrivée SUMO.",
            "Le contrat passenger_CAV ne représente que Car et Taxi ; les quatre autres catégories "
            "restent décrites mais ne sont pas converties.",
            "Les mouvements sont locaux, conditionnels aux visites complètes et ne sont pas des "
            "origines-destinations réelles.",
            "La caractérisation repose sur une seule séquence courte et aucune période empirique "
            "indépendante n'est réservée.",
            "La couverture est inférée sans inspection du masque vidéo brut et aucun second secteur "
            "de transfert n'est validé.",
            *PROVENANCE_LIMITATIONS,
        ],
    }


def _quality_document(
    window_rows: list[dict[str, object]],
    load_levels: list[dict[str, object]],
    composition: dict[str, object],
    hashes: dict[str, str],
    config: CanonicalScenario,
) -> dict[str, object]:
    complete = [row for row in window_rows if row["eligible_for_stratification"]]
    partial = [row for row in window_rows if not row["eligible_for_stratification"]]
    return {
        "schema_version": config.contract.schema_version,
        "status": "complete",
        "inputs_sha256": dict(sorted(hashes.items())),
        "windows": {
            "total": len(window_rows),
            "complete": len(complete),
            "partial": len(partial),
            "group_sizes": {level: size for level, size in zip(config.contract.load_levels, config.contract.group_sizes)},
        },
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


def build_empirical_contract(
    profile_dir: str | Path,
    coverage_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Construit un contrat de trafic à partir de profils validés.

    Les fichiers d'entrée sont contrôlés avant le calcul. Le dossier de sortie
    doit être absent ou vide et n'est publié qu'après génération complète.
    """

    validate_output_directory(output_dir)
    inputs = load_traffic_inputs(profile_dir, coverage_path)

    config = CANONICAL_SCENARIO
    window_rows = _build_window_rows(inputs.flows, inputs.windows, config)
    _assign_load_levels(window_rows, config)
    load_levels = _build_load_levels(window_rows, inputs.movements, inputs.unclassified, config)
    composition = _build_composition(
        inputs.flows,
        inputs.windows,
        inputs.movements,
        inputs.unclassified,
        inputs.summary,
        config,
    )
    contract = _contract_document(
        inputs.hashes,
        inputs.manifest,
        inputs.coverage,
        window_rows,
        load_levels,
        composition,
        config,
    )
    quality = _quality_document(window_rows, load_levels, composition, inputs.hashes, config)
    write_contract_files(
        output_dir,
        window_rows,
        contract,
        quality,
        load_levels,
        composition,
        PROVENANCE_LIMITATIONS,
    )
    return quality
