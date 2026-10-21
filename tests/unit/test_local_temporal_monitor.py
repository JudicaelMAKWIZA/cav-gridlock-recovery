"""L'estimation apprise reste locale, causale et distincte d'une preuve K04."""

import pytest

pytest.importorskip("torch", reason="install the laboratory extra")

from cav_recovery.laboratory.checkpoints import CheckpointStore
from cav_recovery.laboratory.features import BASE_FEATURES
from cav_recovery.laboratory.inference import LocalTemporalMonitor
from cav_recovery.laboratory.model import create_model
from cav_recovery.laboratory.storage import atomic_json


def artifact(path):
    path.mkdir()
    config = dict(hidden=3, dilations=[1, 2], kernel_size=3, architecture="tcn")
    count = len(BASE_FEATURES)
    compatibility = dict(features=count, test=True)
    model = create_model(config, count)
    CheckpointStore(path / "checkpoints", compatibility).save(
        dict(global_step=0, model=model.state_dict()), best=True
    )
    atomic_json(path / "complete.json", dict(compatibility=compatibility))
    atomic_json(path / "config.json", config)
    atomic_json(path / "calibration.json", dict(threshold=0.5))
    atomic_json(
        path / "normalization.json", dict(mean=[0.0] * count, std=[1.0] * count)
    )
    return path


def row(at):
    return dict(
        ego_id="ego",
        movement_id="own-forward",
        observed_at_s=at,
        available_at_s=at,
        own_state=dict(
            observed_at_s=at,
            available_at_s=at,
            heading_rad=0.0,
            signed_speed_mps=1.0,
            mission_progress_m=at,
            acceleration_xy_mps2=[0.0, 0.0],
        ),
        requested_speed_mps=2.0,
        accepted_speed_mps=2.0,
        own_native_action=[0.0, 0.1],
        lidar=[1.0] * 16,
        received_message_ages_s=[],
        lidar_available=True,
        v2v_box_available=True,
    )


def test_monitor_warmup_gap_and_future_boundary(tmp_path):
    monitor = LocalTemporalMonitor(artifact(tmp_path / "model"))
    for i in range(26):
        result = monitor.observe(row(round(i * 0.2, 7)))
    assert result["probability"] is not None
    assert not result["changes_K04_status"] and result["action"] is None
    assert result["status"] in ["STATISTICAL_ALERT", "NO_STATISTICAL_ALERT"]
    gap = monitor.observe(row(7.0))
    assert gap["status"] == "ABSTAIN_INSUFFICIENT_HISTORY"
    future = row(7.2)
    future["own_state"]["observed_at_s"] = 8.0
    with pytest.raises(ValueError, match="même instant"):
        monitor.observe(future)
