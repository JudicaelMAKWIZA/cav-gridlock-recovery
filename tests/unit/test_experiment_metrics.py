"""Références indépendantes, unités statistiques et absence de scores artificiels."""

import gzip
import json

import pytest

from shapely.geometry import box

from cav_recovery.experiments import evaluation


def write_rows(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def test_empty_prediction_denominator_is_not_perfect_precision():
    metrics = evaluation.metrics(dict(tp=0, fp=0, fn=3))
    assert metrics["precision"] is None and metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


@pytest.mark.parametrize("effect_established", [False, True])
def test_natural_recovery_requires_an_established_effect_before_progress(
    tmp_path, monkeypatch, effect_established
):
    directory = tmp_path / "diagnostic"
    directory.mkdir()
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "spec.json").write_text(
        json.dumps(dict(family="four_arm", requested_situation="natural_release")),
        encoding="utf-8",
    )
    event = dict(
        requester="V0",
        confirmed=False,
        start_s=0.4,
        end_s=6.4,
        end_kind="release_observed",
        movement_effect_duration_s=6.0 if effect_established else 0.0,
        effect_intervals=[
            dict(start_s=0.4, end_s=6.4, confirmed=False, confirmation_at_s=None)
        ]
        if effect_established
        else [],
    )
    frames = [
        dict(time_s=t, states={"V0": dict(progress_m=progress)}, witnesses=[])
        for t, progress in ((6.4, 0.0), (40.0, 20.0))
    ]
    reference = dict(label="NEGATIVE_CONFIRMED", events=[event])
    native = dict(
        label="NEGATIVE_CONFIRMED",
        integrity=dict(complete_measurements=True, errors=[]),
        traffic={},
    )
    monkeypatch.setattr(
        evaluation, "physical_reference", lambda folder: (reference, frames, native)
    )
    write_rows(
        directory / "inputs.jsonl.gz",
        [
            dict(at_s=frame["time_s"], measurements={"V0": dict(v2v=None)})
            for frame in frames
        ],
    )
    write_rows(directory / "reports.jsonl.gz", [])
    write_rows(
        directory / "decisions.jsonl.gz",
        [
            dict(
                at_s=frame["time_s"],
                vehicles={
                    "V0": dict(
                        A=dict(status="NO_ALERT", reason="PROGRESSING"),
                        B=dict(status="NO_ALERT", reason="PROGRESSING"),
                        C=dict(
                            simple_count_baseline=False,
                            simple_count_requesters=[],
                            groups=[],
                            edges=[],
                        ),
                    )
                },
            )
            for frame in frames
        ],
    )
    result = evaluation.evaluate_episode(tmp_path)
    assert (
        result["scenario_acceptance"]["physical_rule_satisfied"] is effect_established
    )
    assert (
        result["observed_releases_with_resumed_progress"][0]["subsequent_progress_m"]
        == 20.0
    )


def test_independent_groups_require_physically_connected_requesters():
    def edge(requester, blocker, x):
        return dict(
            requester=requester,
            blocker=blocker,
            resource=dict(polygon_wkt=box(x, 0, x + 1, 1).wkt),
        )

    assert evaluation.reference_groups([edge("A", "B", 0), edge("B", "C", 5)]) == [
        ["A", "B"]
    ]
    assert evaluation.reference_groups([edge("A", "B", 0), edge("C", "D", 5)]) == []


def test_event_metrics_use_the_qualified_end_s_and_keep_unknown_misses(
    tmp_path, monkeypatch
):
    directory = tmp_path / "diagnostic"
    directory.mkdir()
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "spec.json").write_text(
        json.dumps(dict(family="four_arm")), encoding="utf-8"
    )
    event = dict(
        requester="V0",
        confirmed=True,
        confirmation_at_s=30.4,
        end_s=30.4,
        end_kind="last_observed_right_censored",
        effect_intervals=[
            dict(confirmed=True, start_s=0.4, end_s=30.4, confirmation_at_s=30.4)
        ],
    )
    edge = dict(
        requester="V0",
        blocker="V1",
        road_admissible=True,
        resource=dict(polygon_wkt=box(0, 0, 1, 1).wkt),
    )
    reference = dict(label="POSITIVE_CONFIRMED", events=[event])
    native = dict(
        label="POSITIVE_CONFIRMED",
        integrity=dict(complete_measurements=True, errors=[]),
        traffic={},
    )
    monkeypatch.setattr(
        evaluation,
        "physical_reference",
        lambda folder: (reference, [dict(time_s=30.4, witnesses=[edge])], native),
    )
    write_rows(
        directory / "inputs.jsonl.gz",
        [dict(at_s=30.4, measurements={"V0": dict(v2v=None)})],
    )
    write_rows(directory / "reports.jsonl.gz", [])
    write_rows(
        directory / "decisions.jsonl.gz",
        [
            dict(
                at_s=30.4,
                vehicles={
                    "V0": dict(
                        A=dict(status="UNKNOWN", reason="MISSING_INFORMATION"),
                        B=dict(
                            status="CONFIRMED_OBSTRUCTION",
                            reason="PERSISTENT_MOVEMENT_OBSTRUCTION",
                        ),
                        C=dict(
                            simple_count_baseline=False,
                            simple_count_requesters=[],
                            groups=[],
                            edges=[],
                        ),
                    )
                },
            )
        ],
    )
    result = evaluation.evaluate_episode(tmp_path)
    assert result["metrics"]["B"]["recall"] == 1.0
    assert result["metrics"]["A"]["recall"] == 0.0
    assert result["event_results"][0]["missed"]
    assert result["event_results"][1]["delay_s"] == 0.0
