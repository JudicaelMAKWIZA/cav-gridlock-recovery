"""Recettes publiques compactes, développées par le générateur déjà utilisé."""

import json
import math
from pathlib import Path
import random

from ..dataset.contract import digest
from .catalog import specification


def _driving_parameters(spec, rules):
    """Fixe les paramètres propres de conduite avant la création des véhicules."""
    if not isinstance(rules, list):
        raise ValueError("agent overrides must be a list")
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != {"origin", "values"}:
            raise ValueError("invalid agent override")
        origin, values = rule["origin"], rule["values"]
        if origin is not None and (
            type(origin) is not int or not 0 <= origin < spec["topology"]["arms"]
        ):
            raise ValueError("invalid origin selector")
        if not isinstance(values, dict) or not set(values) <= {
            "reaction_s",
            "clearance_m",
            "target_speed_mps",
            "departure_offset_s",
        }:
            raise ValueError("unsupported driving parameter")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= (120 if key == "departure_offset_s" else 10)
            for key, value in values.items()
        ):
            raise ValueError("invalid driving parameter")
        for agent in spec["agents"]:
            if origin is None or agent["origin"] == origin:
                agent.update(
                    {
                        key: value
                        for key, value in values.items()
                        if key != "departure_offset_s"
                    }
                )
                agent["departure_s"] += values.get("departure_offset_s", 0)


def _right_turn_pulse(spec, pulse):
    """Construit un flux fini sur des connexions à droite réellement présentes."""
    if pulse is not None:
        if not isinstance(pulse, dict) or set(pulse) != {
            "seed",
            "headway_s",
            "jitter_s",
        }:
            raise ValueError("invalid arrival pulse")
        if (
            type(pulse["seed"]) is not int
            or not 0 < pulse["headway_s"] <= 10
            or not 0 <= pulse["jitter_s"] <= 1
        ):
            raise ValueError("invalid arrival pulse parameters")
        rng = random.Random(pulse["seed"])
        lane = spec["topology"]["lane_num"] - 1
        for index, agent in enumerate(spec["agents"]):
            choices = [
                road
                for road in spec["roads"]
                if road["start"] == f"a{agent['origin']}_entry"
                and road["kind"] == "right"
            ]
            if not choices:
                raise ValueError("native right-turn movement missing")
            road = rng.choice(choices)
            destination = int(road["end"].split("_")[0][1:])
            agent.update(
                destination=destination,
                lane=lane,
                movement_id=f"a{agent['origin']}_a{destination}_l{lane}",
                movement_kind="right",
                departure_s=round(
                    index * pulse["headway_s"] + rng.uniform(0, pulse["jitter_s"]), 3
                ),
            )


def load_specification(path):
    """Lit une configuration complète ou une recette dont l'identité est gelée.

    La recette ne contient ni mesure ni étiquette. Elle réutilise la géométrie du
    catalogue, puis fixe les paramètres de trafic avant la première simulation.
    Son empreinte attendue empêche une modification silencieuse de l'expérience.
    """
    recipe = json.loads(Path(path).read_text("utf-8"))
    if not isinstance(recipe, dict):
        raise ValueError("demonstration configuration must be an object")
    if recipe.get("schema") != "cgr-demo-recipe-1":
        return recipe
    fields = {
        "schema",
        "base_scenario",
        "seed",
        "overrides",
        "agent_overrides",
        "right_turn_pulse",
        "expected_spec_id",
    }
    if set(recipe) != fields or type(recipe["seed"]) is not int or recipe["seed"] < 0:
        raise ValueError("invalid demonstration recipe fields")
    spec = specification(recipe["base_scenario"], seed=recipe["seed"])
    updates = recipe["overrides"]
    if not isinstance(updates, dict) or not set(updates) <= {
        "experiment_name",
        "movement_profile",
        "demand",
        "active_limit",
        "density_level",
        "horizon_s",
    }:
        raise ValueError("unsupported demonstration override")
    spec.update(updates)
    if type(spec["active_limit"]) is not int or not 1 <= spec["active_limit"] <= 30:
        raise ValueError("invalid active CAV limit")
    if (
        not isinstance(spec["horizon_s"], (int, float))
        or not math.isfinite(spec["horizon_s"])
        or not 0 < spec["horizon_s"] <= 600
    ):
        raise ValueError("invalid demonstration horizon")
    _driving_parameters(spec, recipe["agent_overrides"])
    _right_turn_pulse(spec, recipe["right_turn_pulse"])
    spec.pop("id")
    spec["id"] = digest(spec)[:24]
    if spec["id"] != recipe["expected_spec_id"]:
        raise ValueError("expanded demonstration differs from its frozen specification")
    return spec
