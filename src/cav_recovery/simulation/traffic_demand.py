"""Missions synthétiques avec arrivées de Poisson explicites et reproductibles."""

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class Mission:
    """Route et destination assignées avant l'insertion du véhicule."""

    vehicle_id: str
    regime: str
    entry_gate: str
    exit_gate: str
    route_id: str
    route: tuple[str, ...]
    destination: str
    scheduled_s: float
    sampled_s: float | None = None


def poisson_missions(routes: dict, weights: dict, rates: dict, duration_s: float,
                     seed: int, demand: str) -> list[Mission]:
    """Tire les intervalles exponentiels puis la destination de chaque mission.

    Les intensités sont en véhicules/heure/entrée. Un seul RNG local consomme
    les entrées et destinations dans un ordre trié. Les temps continus sont
    conservés dans sampled_s. La programmation scheduled_s est arrondie vers
    le haut à la milliseconde, résolution interne de SUMO, afin de ne jamais
    insérer avant le tirage. L'insertion réelle reste mesurée séparément.
    """
    if not math.isfinite(duration_s) or duration_s <= 0 or type(seed) is not int or seed < 0:
        raise ValueError("Durée ou seed invalide.")
    rng = random.Random(seed)
    missions = []
    for entry in sorted(rates):
        rate = rates[entry]
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("Chaque intensité doit être finie et positive.")
        destinations = sorted(weights[entry])
        values = [weights[entry][exit_gate] for exit_gate in destinations]
        if not destinations or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Les poids des destinations doivent être positifs.")
        for exit_gate in destinations:
            if entry == exit_gate or f"{entry}__{exit_gate}" not in routes:
                raise ValueError("Destination ou route absente pour la demande.")
        time_s = rng.expovariate(rate / 3600)
        sequence = 0
        while time_s < duration_s:
            exit_gate = rng.choices(destinations, weights=values, k=1)[0]
            route_id = f"{entry}__{exit_gate}"
            route = tuple(routes[route_id])
            if len(route) < 2:
                raise ValueError("Une mission doit traverser au moins deux edges.")
            missions.append(Mission(f"{entry}_{sequence:06d}", demand, entry, exit_gate,
                                    route_id, route, route[-1],
                                    math.ceil(time_s * 1000) / 1000, time_s))
            sequence += 1
            time_s += rng.expovariate(rate / 3600)
    return sorted(missions, key=lambda mission: (mission.scheduled_s, mission.vehicle_id))
