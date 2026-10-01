"""Construction d'un contrat de scénario à partir de profils de trafic validés."""

from __future__ import annotations

from pathlib import Path

from .contract_files import validate_output_directory, write_contract_files
from .traffic_inputs import (
    ALL_CATEGORIES,
    ENTRY_GATES,
    EXIT_GATES,
    ContractInputError,
    load_traffic_inputs,
    same_number,
)


SCHEMA_VERSION = "CGR-E03-1"
SOURCE_CATEGORIES = ("Car", "Taxi")
LOAD_LEVEL_ORDER = ("LOW", "MID", "HIGH")
GROUP_SIZES = (4, 5, 4)
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
) -> list[dict[str, object]]:
    """Calcule la charge Car+Taxi de chaque fenêtre aux deux portes d'entrée.

    La dernière fenêtre reste dans le profil, mais seule une fenêtre complète de
    60 secondes peut participer au classement des niveaux de charge.
    """

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
            "eligible_for_stratification": same_number(duration, 60.0),
            "passenger_in_w23183369": by_gate[ENTRY_GATES[0]],
            "passenger_in_w284241336": by_gate[ENTRY_GATES[1]],
            "passenger_in_total": sum(by_gate.values()),
            "regime_id": None,
        })
    return result


def _assign_load_levels(window_rows: list[dict[str, object]]) -> None:
    """Classe les fenêtres complètes par charge croissante selon la règle 4/5/4.

    Le temps de début départage deux charges égales afin que le résultat reste
    déterministe.
    """

    eligible = [row for row in window_rows if row["eligible_for_stratification"]]
    if len(eligible) != sum(GROUP_SIZES):
        raise ContractInputError("La règle 4/5/4 exige exactement 13 fenêtres complètes.")
    ranked = sorted(
        eligible,
        key=lambda row: (int(row["passenger_in_total"]), float(row["window_start_s"])),
    )
    offset = 0
    for load_level, size in zip(LOAD_LEVEL_ORDER, GROUP_SIZES):
        for row in ranked[offset:offset + size]:
            row["regime_id"] = load_level
        offset += size


def _aggregate_movements(
    member_windows: set[float],
    movements: dict[tuple[str, str, str, float], dict[str, int]],
) -> tuple[dict[str, dict[str, int]], dict[str, int], dict[str, dict[str, float | None]]]:
    """Agrège les mouvements Car+Taxi des fenêtres appartenant à un niveau."""

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
        # Le profil répète le même dénominateur sur les deux sorties. Une seule
        # sortie est donc lue pour éviter de compter deux fois les visites.
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
    """Construit les comptes, taux et mouvements de chaque niveau de charge."""

    load_levels = []
    for load_level in LOAD_LEVEL_ORDER:
        members = sorted(
            (row for row in window_rows if row["regime_id"] == load_level),
            key=lambda row: float(row["window_start_s"]),
        )
        starts = {float(row["window_start_s"]) for row in members}
        exposure = sum(float(row["duration_s"]) for row in members)
        entry_counts = {
            ENTRY_GATES[0]: sum(int(row["passenger_in_w23183369"]) for row in members),
            ENTRY_GATES[1]: sum(int(row["passenger_in_w284241336"]) for row in members),
        }
        total = sum(entry_counts.values())
        movement_counts, denominators, probabilities = _aggregate_movements(starts, movements)
        # Une sortie censurée reste une entrée observée, mais ne devient jamais
        # un mouvement certain dans les probabilités ci-dessus.
        censored_exit = sum(
            int(row["count"])
            for row in unclassified
            if row["status"] == "censored_exit"
            and row["category"] in SOURCE_CATEGORIES
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
) -> dict[str, object]:
    """Résume les six catégories sans convertir les véhicules non retenus."""

    complete_starts = [start for start, end in windows if same_number(end - start, 60.0)]
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
        complete_counts = [
            sum(int(flows[(gate, category, start)]["count"]) for gate in ENTRY_GATES)
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
        "passenger_entries": sum(int(categories[category]["observed_entries"]) for category in SOURCE_CATEGORIES),
    }


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
            "input_artifacts_sha256": dict(sorted(hashes.items())),
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
            "sector": {
                "id": "C3",
                "osm_node_id": 250691665,
                "entry_gates": list(ENTRY_GATES),
                "exit_gates": list(EXIT_GATES),
            },
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
) -> dict[str, object]:
    complete = [row for row in window_rows if row["eligible_for_stratification"]]
    partial = [row for row in window_rows if not row["eligible_for_stratification"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "inputs_sha256": dict(sorted(hashes.items())),
        "windows": {
            "total": len(window_rows),
            "complete": len(complete),
            "partial": len(partial),
            "group_sizes": {level: size for level, size in zip(LOAD_LEVEL_ORDER, GROUP_SIZES)},
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

    window_rows = _build_window_rows(inputs.flows, inputs.windows)
    _assign_load_levels(window_rows)
    load_levels = _build_load_levels(window_rows, inputs.movements, inputs.unclassified)
    composition = _build_composition(
        inputs.flows,
        inputs.windows,
        inputs.movements,
        inputs.unclassified,
        inputs.summary,
    )
    contract = _contract_document(
        inputs.hashes,
        inputs.manifest,
        inputs.coverage,
        window_rows,
        load_levels,
        composition,
    )
    quality = _quality_document(window_rows, load_levels, composition, inputs.hashes)
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
