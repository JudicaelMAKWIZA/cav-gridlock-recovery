"""Transformation de passages observés en nouvelles missions déterministes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path


class TrafficInputError(ValueError):
    """Signale une entrée incompatible avec le scénario nominal."""


CONTRACT_IDENTITY = (13055, "409564667c1ea1466bc41b70e4bdff3ef4922dbb648308fc526dabeb5c7f980a")
OSM_IDENTITY = (76486, "c5c2105c28807e8bcb2743ca53079bacef23ccca0bdf18d7314d218f51b1f2bf")
ENTRY_GATES = ("W23183369_IN", "W284241336_IN")
EXIT_GATES = ("W23183369_OUT", "W284241336_OUT")
STEP_S = 0.5
VEHICLE_TYPE = {
    "id": "passenger_CAV", "vClass": "passenger", "carFollowModel": "Krauss",
    "length": "5.0", "minGap": "2.5", "accel": "2.6", "decel": "4.5",
    "tau": "1.0", "sigma": "0", "speedFactor": "1.0", "guiShape": "passenger/sedan",
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
        context = contract["empirical_context"]
        sector = context["sector"]
        passenger = contract["passenger_cav_contract"]
        if (contract["schema_version"] != "CGR-E03-1" or contract["status"] != "complete"
                or sector["id"] != "C3" or sector["osm_node_id"] != 250691665
                or sector["entry_gates"] != list(ENTRY_GATES)
                or sector["exit_gates"] != list(EXIT_GATES)
                or context["coverage"] != {"status": "known", "intervals": [[0.0, 802.8]]}
                or passenger["source_categories"] != ["Car", "Taxi"]
                or passenger["population_id"] != "passenger_CAV"):
            raise TrafficInputError("Identité du secteur, couverture ou population incompatible.")
        plans = demand_plans(contract)
        expected = {"LOW": (240, [23, 72], [18, 5, 3, 69]),
                    "MID": (300, [22, 144], [16, 6, 3, 141]),
                    "HIGH": (240, [15, 141], [7, 8, 8, 133])}
        if set(plans) != set(expected):
            raise TrafficInputError("Les trois niveaux de charge sont requis.")
        for name, (duration, entries, movements) in expected.items():
            plan = plans[name]
            if (plan["injection_s"] != duration or list(plan["entries"].values()) != entries
                    or [n for counts in plan["allocation"].values() for n in counts.values()] != movements):
                raise TrafficInputError(f"Effectifs ou durée incompatibles pour {name}.")
        high = next(r for r in passenger["regimes"] if r["regime_id"] == "HIGH")
        if high["classifiable_visits"] != 154 or high["censored_exit"] != 2:
            raise TrafficInputError("La censure du niveau HIGH doit rester distincte des missions.")
        return contract
    except (KeyError, TypeError, json.JSONDecodeError) as error:
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
    """Conserve la distribution observée à côté de l'allocation simulée."""
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
        plans[name] = {"injection_s": row["exposure_s"], "entries": entries,
                       "observed_movements": observed,
                       "observed_denominators": row["passenger_movement_denominators"],
                       "censored_exit": row["censored_exit"],
                       "allocation": {gate: allocate_counts(entries[gate], observed[gate]) for gate in ENTRY_GATES}}
    return plans


def departure_times(count: int, duration_s: float) -> list[float]:
    """Répartit les départs sur la grille de demi-secondes, sans doublon par entrée."""
    if (not math.isfinite(duration_s) or duration_s <= 0 or duration_s % STEP_S
            or type(count) is not int or count < 0 or count > duration_s / STEP_S):
        raise TrafficInputError("Durée ou effectif incompatible avec la grille de départs.")
    slots = int(duration_s / STEP_S)
    return [(i * slots // count) * STEP_S for i in range(count)]


def interleave_movements(counts: dict[str, int]) -> list[str]:
    """Choisit à chaque départ la sortie la plus en retard sur sa part cumulée."""
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
