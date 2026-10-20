"""Les recettes publiques restent reproductibles sans recopier les géométries."""

import json
from pathlib import Path

import pytest

from cav_recovery.experiments.catalog import specification
from cav_recovery.experiments.configuration import load_specification
from cav_recovery.experiments.configuration import _driving_parameters

RECIPES = Path(__file__).parents[2] / "configs/metadrive-demonstrations"


@pytest.mark.parametrize(
    "path", sorted(RECIPES.glob("*.json")), ids=lambda path: path.stem
)
def test_published_recipe_expands_to_its_frozen_physical_configuration(path):
    recipe = json.loads(path.read_text("utf-8"))
    spec = load_specification(path)
    assert spec["id"] == recipe["expected_spec_id"]
    assert spec["intersection_count"] == 1 and spec["development_only"]
    assert 1 <= spec["active_limit"] <= 30
    assert all(agent["target_speed_mps"] > 0.15 for agent in spec["agents"])


@pytest.mark.parametrize("alteration", ["reference_label", "changed_identity"])
def test_recipe_does_not_accept_opaque_fields_or_a_changed_physical_identity(
    tmp_path, alteration
):
    recipe = json.loads(next(RECIPES.glob("*.json")).read_text("utf-8"))
    if alteration == "reference_label":
        recipe["reference_label"] = "POSITIVE_CONFIRMED"
    else:
        recipe["expected_spec_id"] = "0" * 24
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(recipe), encoding="utf-8")
    with pytest.raises(ValueError):
        load_specification(path)


def test_complete_specification_format_remains_supported(tmp_path):
    spec = specification("nominal-1")
    path = tmp_path / "complete.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert load_specification(path) == spec


def test_departure_offset_preserves_staggered_missions_and_other_approaches():
    spec = specification("natural_release-1")
    before = [
        (agent["origin"], agent["departure_s"], agent["movement_id"])
        for agent in spec["agents"]
    ]
    _driving_parameters(spec, [dict(origin=0, values=dict(departure_offset_s=22))])
    for agent, (origin, departure, movement) in zip(spec["agents"], before):
        assert agent["departure_s"] == departure + (22 if origin == 0 else 0)
        assert agent["movement_id"] == movement


@pytest.mark.parametrize("offset", [-1, 121, float("nan")])
def test_departure_offset_rejects_invalid_or_unbounded_admissions(offset):
    with pytest.raises(ValueError):
        _driving_parameters(
            specification("natural_release-1"),
            [dict(origin=0, values=dict(departure_offset_s=offset))],
        )
