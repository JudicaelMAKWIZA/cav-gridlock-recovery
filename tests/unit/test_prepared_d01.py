"""Le chemin rapide conserve les mêmes décisions et la même frontière locale."""

from dataclasses import FrozenInstanceError

import pytest

from cav_recovery.d01.inputs import LocalInputFactory, InputError
from cav_recovery.d01.observer import EgoObserver
from tests.support.d01_fixtures import fixture_a, packet


def dynamic(row):
    return {
        key: value
        for key, value in row.items()
        if key not in {"schema_version", "contract_id", "mission", "map"}
    }


def test_prepared_and_public_inputs_have_identical_k04_decisions():
    rows = fixture_a()
    factory = LocalInputFactory(rows[0])
    raw, prepared = EgoObserver(), EgoObserver()
    for row in rows:
        assert raw.update(row) == prepared.update(factory.build(dynamic(row)))


def test_prepared_static_context_is_detached_immutable_and_ego_bound():
    row = packet()
    factory = LocalInputFactory(row)
    row["map"]["drivable_polygons_xy_m"][0][0][0] = 999.0
    assert factory.map["drivable_polygons_xy_m"][0][0][0] == -20.0
    with pytest.raises(FrozenInstanceError):
        factory.map = {}
    changed = dynamic(packet())
    changed["ego_id"] = "another-car"
    with pytest.raises(InputError):
        factory.build(changed)


def test_dynamic_fields_and_future_times_still_receive_full_validation():
    factory = LocalInputFactory(packet())
    for key in ("hidden_global", "guard_blockers"):
        wrong = dynamic(packet())
        wrong[key] = []
        with pytest.raises(InputError):
            factory.build(wrong)
    wrong = dynamic(packet())
    wrong["ego"]["observed_at_s"] = 2.0
    with pytest.raises(InputError):
        factory.build(wrong)
    wrong = dynamic(packet())
    wrong["own_controls"]["planned_path_xyh_m"] = [[0, 0, 0], [float("nan"), 0, 0]]
    with pytest.raises(InputError):
        factory.build(wrong)


def test_compact_output_changes_only_diagnostic_detail_not_conclusions():
    detailed, compact = EgoObserver(), EgoObserver(detailed=False)
    for row in fixture_a():
        first, second = detailed.update(row), compact.update(row)
        for key in (
            "status",
            "reason",
            "effect_window",
            "current_occupants",
            "historical_confirmations",
        ):
            assert first[key] == second[key]
    assert not second["own_effect_samples"]
    assert second["details_omitted"]
    assert (
        second["coverage"]["persistence_window"]
        == first["coverage"]["persistence_window"]
    )
