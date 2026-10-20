"""Counts must represent distinct affected cars and known confirmation times."""

from collections import Counter
from scripts.describe_dense_traffic import effect_changes, summary


def event(
    actor, start, end, *, confirmed=True, confirmation=30.0, resource="junction:central"
):
    return dict(
        requester=actor,
        resource_id=resource,
        confirmed=confirmed,
        effect_intervals=[
            dict(
                start_s=start,
                end_s=end,
                confirmed=confirmed,
                confirmation_at_s=confirmation if confirmed else None,
            )
        ],
    )


def actors_at(changes, t):
    counts = Counter()
    for moment, delta in changes:
        if moment <= t + 1e-7:
            counts.update(delta)
    return {actor for (actor, _), n in counts.items() if n > 0}


def test_multiple_relations_do_not_duplicate_a_vehicle():
    changes = effect_changes(
        [event("V1", 0, 40), event("V1", 0, 40), event("V2", 0, 40)]
    )
    assert actors_at(changes, 30) == {"V1", "V2"}


def test_confirmation_is_unavailable_before_its_actual_instant():
    changes = effect_changes([event("V1", 0, 40)], asof=True)
    assert not actors_at(changes, 29.8)
    assert actors_at(changes, 30) == {"V1"}


def test_retroactive_effect_includes_only_its_observed_interval():
    changes = effect_changes([event("V1", 5, 40)])
    assert not actors_at(changes, 4.8)
    assert actors_at(changes, 5) == {"V1"}
    assert actors_at(changes, 40) == {"V1"}
    assert not actors_at(changes, 40.2)


def test_transient_constraints_are_not_confirmed_obstructions():
    events = [event("V1", 0, 8, confirmed=False)]
    assert not actors_at(effect_changes(events), 5)
    assert actors_at(effect_changes(events, confirmed_only=False), 5) == {"V1"}


def test_sequential_region_changes_do_not_overlap():
    changes = effect_changes(
        [event("V1", 0, 30, resource="arm:0:lane:0:approach"), event("V1", 30.2, 65)]
    )
    counts = Counter()
    for t, delta in changes:
        counts.update(delta)
        assert sum(n > 0 for n in counts.values()) <= 1


def test_histogram_quantiles_preserve_observation_counts():
    result = summary(Counter({0: 5, 10: 4, 30: 1}))
    assert result["n"] == 10
    assert result["median"] == 0
    assert result["q95"] == 30
    assert result["mean"] == 7
