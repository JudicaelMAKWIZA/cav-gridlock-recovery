"""Transformation de passages observés en nouvelles missions déterministes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from ..c3_reference import (
    C3_NODE_ID, C3_SECTOR_ID, ENTRY_GATES, EXIT_GATES, CONTRACT_SCHEMA,
    SOURCE_CATEGORIES, CAV_POPULATION_ID, LOAD_LEVELS, OBSERVATION_INTERVAL_S,
)


class TrafficInputError(ValueError):
    """Signale une entrée incompatible avec le scénario nominal."""


CONTRACT_IDENTITY = (13055, "409564667c1ea1466bc41b70e4bdff3ef4922dbb648308fc526dabeb5c7f980a")
STEP_S = 0.5
EXPECTED_DEMAND = {
    "LOW": {
        "injection_s": 240,
        "entries": {"W23183369_IN": 23, "W284241336_IN": 72},
        "allocation": {
            "W23183369_IN": {"W23183369_OUT": 18, "W284241336_OUT": 5},
            "W284241336_IN": {"W23183369_OUT": 3, "W284241336_OUT": 69},
        },
    },
    "MID": {
        "injection_s": 300,
        "entries": {"W23183369_IN": 22, "W284241336_IN": 144},
        "allocation": {
            "W23183369_IN": {"W23183369_OUT": 16, "W284241336_OUT": 6},
            "W284241336_IN": {"W23183369_OUT": 3, "W284241336_OUT": 141},
        },
    },
    "HIGH": {
        "injection_s": 240,
        "entries": {"W23183369_IN": 15, "W284241336_IN": 141},
        "allocation": {
            "W23183369_IN": {"W23183369_OUT": 7, "W284241336_OUT": 8},
            "W284241336_IN": {"W23183369_OUT": 8, "W284241336_OUT": 133},
        },
        "classifiable_visits": 154,
        "censored_exit": 2,
    },
}


def file_hash(path: str | Path) -> str:
    """Calcule l'empreinte des octets, sans charger une grosse source en mémoire."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_identity(path: str | Path, identity: tuple[int, str]) -> dict:
    path = Path(path)
    if not path.is_file():
        raise TrafficInputError(f"Fichier d'entrée absent : {path.name}")
    size, digest = identity
    if path.stat().st_size != size or file_hash(path) != digest:
        raise TrafficInputError(f"Taille ou SHA-256 inattendu : {path.name}")
    return {"filename": path.name, "size_bytes": size, "sha256": digest}


def read_contract(path: str | Path) -> dict:
    """Vérifie la sortie validée avant de réutiliser ses comptes et censures."""
    verify_identity(path, CONTRACT_IDENTITY)
    try:
        contract = json.loads(Path(path).read_text(encoding="utf-8"))
        plans = demand_plans(contract)
        _validate_c3_contract(contract, plans)
        return contract
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise TrafficInputError(f"Structure du contrat invalide : {error}") from error


def _validate_c3_contract(contract: dict, plans: dict) -> None:
    """Vérifie l'identité C3 et les effectifs attendus."""
    try:
        context = contract["empirical_context"]
        sector = context["sector"]
        passenger = contract["passenger_cav_contract"]
        if (contract["schema_version"] != CONTRACT_SCHEMA or contract["status"] != "complete"
                or sector["id"] != C3_SECTOR_ID or sector["osm_node_id"] != C3_NODE_ID
                or sector["entry_gates"] != list(ENTRY_GATES)
                or sector["exit_gates"] != list(EXIT_GATES)
                or context["coverage"] != {"status": "known", "intervals": [list(OBSERVATION_INTERVAL_S)]}
                or passenger["source_categories"] != list(SOURCE_CATEGORIES)
                or passenger["population_id"] != CAV_POPULATION_ID):
            raise TrafficInputError("Identité du secteur, couverture ou population incompatible.")
        if set(plans) != set(LOAD_LEVELS):
            raise TrafficInputError("Les trois niveaux de charge sont requis.")
        for name, expected in EXPECTED_DEMAND.items():
            plan = plans[name]
            if (plan["injection_s"] != expected["injection_s"] or plan["entries"] != expected["entries"]
                    or plan["allocation"] != expected["allocation"]):
                raise TrafficInputError(f"Effectifs ou durée incompatibles pour {name}.")
        expected_high = EXPECTED_DEMAND["HIGH"]
        high = next(r for r in passenger["regimes"] if r["regime_id"] == "HIGH")
        if high["classifiable_visits"] != expected_high["classifiable_visits"] or high["censored_exit"] != expected_high["censored_exit"]:
            raise TrafficInputError("La censure du niveau HIGH doit rester distincte des missions.")
    except (KeyError, TypeError) as error:
        raise TrafficInputError(f"Structure du contrat invalide : {error}") from error


def allocate_counts(total: int, observed: dict[str, int]) -> dict[str, int]:
    """Applique les plus forts restes aux proportions des mouvements complets.

    L'allocation décrit de nouvelles missions ; elle ne change pas les visites
    censurées. Les calculs entiers évitent un départage dû aux flottants.
    """
    if type(total) is not int or total < 0 or any(type(n) is not int or n < 0 for n in observed.values()):
        raise TrafficInputError("Les effectifs doivent être des entiers positifs ou nuls.")
    denominator = sum(observed.values())
    if denominator == 0:
        if total:
            raise TrafficInputError("Aucune proportion définie pour programmer cette entrée.")
        return {key: 0 for key in sorted(observed)}
    counts = {key: total * observed[key] // denominator for key in sorted(observed)}
    order = sorted(observed, key=lambda key: (-(total * observed[key] % denominator), key))
    for key in order[:total - sum(counts.values())]:
        counts[key] += 1
    return counts


def demand_plans(contract: dict) -> dict:
    """Alloue de nouvelles missions selon les mouvements complets observés.

    Les effectifs observés et les censures restent dans le plan, séparément de
    l'allocation simulée. Une censure ne devient pas un mouvement observé.
    """
    plans = {}
    for row in contract["passenger_cav_contract"]["regimes"]:
        name = row["regime_id"]
        if name in plans:
            raise TrafficInputError("Niveau de charge répété.")
        entries = {gate: row["passenger_entry_counts"][gate] for gate in ENTRY_GATES}
        observed = row["passenger_movement_counts"]
        for gate in ENTRY_GATES:
            if set(observed[gate]) != set(EXIT_GATES):
                raise TrafficInputError("La grille des mouvements est incomplète.")
            if sum(observed[gate].values()) != row["passenger_movement_denominators"][gate]:
                raise TrafficInputError("Le dénominateur des mouvements est incohérent.")
        if (sum(row["passenger_movement_denominators"].values()) != row["classifiable_visits"]
                or sum(entries.values()) != row["classifiable_visits"] + row["censored_exit"]):
            raise TrafficInputError("Les entrées ne se réconcilient pas avec les visites et censures.")
        plans[name] = {
            "injection_s": row["exposure_s"],
            "entries": entries,
            "observed_movements": observed,
            "observed_denominators": row["passenger_movement_denominators"],
            "censored_exit": row["censored_exit"],
            "allocation": {gate: allocate_counts(entries[gate], observed[gate]) for gate in ENTRY_GATES},
        }
    return plans


def departure_times(count: int, duration_s: float) -> list[float]:
    """Répartit les départs sur la grille de demi-secondes, sans doublon par entrée."""
    if (not math.isfinite(duration_s) or duration_s <= 0 or duration_s % STEP_S
            or type(count) is not int or count < 0 or count > duration_s / STEP_S):
        raise TrafficInputError("Durée ou effectif incompatible avec la grille de départs.")
    slots = int(duration_s / STEP_S)
    return [(i * slots // count) * STEP_S for i in range(count)]


def interleave_movements(counts: dict[str, int]) -> list[str]:
    """Choisit à chaque départ la sortie la plus en retard sur sa part cumulée.

    En cas d'égalité, l'ordre des identifiants de sortie départage les mouvements.
    """
    total = sum(counts.values())
    used = dict.fromkeys(counts, 0)
    sequence = []
    for index in range(total):
        remaining = [key for key in sorted(counts) if used[key] < counts[key]]
        key = max(remaining, key=lambda k: (index + 1) * counts[k] - total * used[k])
        used[key] += 1
        sequence.append(key)
    return sequence


@dataclass(frozen=True)
class Mission:
    """Mission assignée avant insertion, indépendante de la trajectoire humaine."""

    vehicle_id: str
    regime: str
    entry_gate: str
    exit_gate: str
    route_id: str
    route: tuple[str, ...]
    destination: str
    scheduled_s: float


def build_missions(regime: str, plan: dict, routes: dict[str, list[str]]) -> list[Mission]:
    """Construit les missions SUMO à partir du plan de trafic.

    Les sorties sont entrelacées et les départs régulièrement espacés. Chaque
    destination est la dernière arête de la route assignée ; les missions sont
    triées par horaire puis identifiant pour garder un ordre déterministe.
    """
    missions = []
    for gate in ENTRY_GATES:
        times = departure_times(plan["entries"][gate], plan["injection_s"])
        sequence = interleave_movements(plan["allocation"][gate])
        for index, (time_s, exit_gate) in enumerate(zip(times, sequence)):
            route_id = f"{gate}__{exit_gate}"
            route = tuple(routes[route_id])
            if not route:
                raise TrafficInputError("Une mission ne peut pas avoir une route vide.")
            missions.append(Mission(f"{regime}-{gate}-{index:04d}", regime, gate, exit_gate,
                                    route_id, route, route[-1], time_s))
    return sorted(missions, key=lambda m: (m.scheduled_s, m.vehicle_id))


def mission_records(missions: list[Mission]) -> list[dict]:
    return [{**asdict(mission), "route": list(mission.route)} for mission in missions]
