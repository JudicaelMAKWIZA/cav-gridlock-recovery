"""Frontière causale et invariances des variables du laboratoire."""
# ruff: noqa: E402

import math
import pytest

np = pytest.importorskip("numpy", reason="install the laboratory extra")
from cav_recovery.laboratory.features import vectorize, received_ages, BASE_FEATURES
from cav_recovery.laboratory.metrics import binary_metrics, choose_threshold


def state(t=1.0, heading=0.0):
    return dict(
        observed_at_s=t,
        available_at_s=t,
        heading_rad=heading,
        signed_speed_mps=2.0,
        acceleration_xy_mps2=[1.0, 0.0],
        mission_progress_m=3.0,
    )


def test_coordinate_and_identifier_invariance_and_causal_differences():
    a = state()
    a.update(vehicle_id="V99", position_xy_m=[1000, 3000], global_label=True)
    b = state(heading=math.pi / 2)
    b["acceleration_xy_mps2"] = [0.0, 1.0]
    np.testing.assert_allclose(
        vectorize(a, None, [1.0] * 16, []),
        vectorize(b, None, [1.0] * 16, []),
        atol=1e-7,
    )
    assert len(vectorize(a, None, [1.0] * 16, [])) == len(BASE_FEATURES)
    a["available_at_s"] = 1.1
    with pytest.raises(ValueError):
        vectorize(a, None, [1.0] * 16, [])


def test_future_mailbox_and_missing_supervision_are_not_negatives():
    box = dict(
        received=dict(
            peer=dict(
                received_at_s=1.2, message=dict(observed_at_s=1.1, valid_until_s=2.0)
            )
        )
    )
    with pytest.raises(ValueError):
        received_ages(box, 1.0)
    result = binary_metrics([1, 0, -1], [True, True, True], [True, True, False])
    assert (result["tp"], result["fp"], result["masked"]) == (1, 1, 1)
    assert choose_threshold([0, 1], [0.1, 0.9])["score"]["f1"] == 1.0


def test_unknown_oracle_fields_do_not_enter_feature_names():
    assert not any(
        any(
            term in key
            for term in [
                "label",
                "episode",
                "topology",
                "diagnostic",
                "graph",
                "vehicle_id",
            ]
        )
        for key in BASE_FEATURES
    )
