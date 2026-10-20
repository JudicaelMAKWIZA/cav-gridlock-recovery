"""Préfixes indépendants du futur et fenêtres sans mélange de véhicules."""

import pytest

from cav_recovery.experiments.temporal import causal_windows


def feature(ego, at_s, value):
    return dict(
        ego_id=ego,
        movement_id=ego + ":forward",
        observed_at_s=at_s,
        available_at_s=at_s,
        own_state=dict(mission_progress_m=value),
    )


def test_each_movement_has_its_own_immutable_past_and_missing_prefix_mask():
    rows = [feature("A", 0.0, 1), feature("B", 0.0, 2), feature("A", 0.2, 3)]
    windows = list(causal_windows(rows, history_samples=3))
    assert windows[-1]["observed_mask"] == (False, True, True)
    assert [
        row["own_state"]["mission_progress_m"] for row in windows[-1]["samples"]
    ] == [1, 3]
    rows[0]["own_state"]["mission_progress_m"] = 999
    assert windows[0]["samples"][0]["own_state"]["mission_progress_m"] == 1
    with pytest.raises(TypeError):
        windows[0]["samples"][0]["own_state"]["mission_progress_m"] = 999


def test_future_suffix_and_observation_gap_never_fill_a_past_window():
    prefix = [feature("A", 0.0, 1), feature("A", 0.2, 2)]
    first = list(causal_windows(prefix))
    second = list(causal_windows(prefix + [feature("A", 0.8, 999)]))
    assert first == second[:2]
    assert len(second[-1]["samples"]) == 1
    assert sum(second[-1]["observed_mask"]) == 1


def test_reference_annotations_are_rejected_from_feature_windows():
    row = feature("A", 0.0, 1)
    row["reference_label"] = "POSITIVE_CONFIRMED"
    with pytest.raises(ValueError):
        list(causal_windows([row]))
