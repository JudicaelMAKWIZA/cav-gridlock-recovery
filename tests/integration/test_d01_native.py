import copy
import json
import random
import pytest

pytest.importorskip(
    "shapely", reason="install the d01 extra for observer integration tests"
)

from cav_recovery.d01.native import local_packet, project_mailbox
from cav_recovery.d01.inputs import adapt, InputError
from cav_recovery.d01.observer import EgoObserver
from tests.support.d01_fixtures import packet


def test_projection_does_not_admit_transport_global_or_mutate_mailbox():
    p = packet()
    entry = p["v2v"]["received"][0]
    knowledge = dict(
        own={"secret": "not used"},
        received={"V1": copy.deepcopy(entry)},
        expired={},
        last_sent={},
        last_sequence={},
    )
    a = project_mailbox(knowledge, 0.4)
    knowledge["own"] = {"global_world": "changed"}
    b = project_mailbox(knowledge, 0.4)
    assert a == b
    knowledge["received"]["V1"]["message"]["position_xy_m"][0] = 999
    assert a["received"][0]["message"]["position_xy_m"] == [5.0, 0.0]
    assert project_mailbox(None, 0.4) is None


def test_composed_live_input_is_exactly_one_ego_and_core_uses_no_rng():
    p = packet()
    measurement = {
        k: v
        for k, v in p["ego"].items()
        if k not in ("observed_at_s", "available_at_s")
    }
    controls = {
        k: v
        for k, v in p["own_controls"].items()
        if k not in ("observed_at_s", "available_at_s")
    }
    knowledge = dict(received={"V1": p["v2v"]["received"][0]}, expired={})
    out = local_packet(
        "V0",
        0.4,
        measurement,
        p["mission"],
        p["map"],
        p["lidar"]["normalized_ranges"],
        controls,
        knowledge,
    )
    assert adapt(out).ego_id == "V0"
    before = random.getstate()
    EgoObserver().update(out)
    assert before == random.getstate()
    measurement["guard_blockers"] = ["secret"]
    with pytest.raises(InputError):
        adapt(
            local_packet(
                "V0",
                0.4,
                measurement,
                p["mission"],
                p["map"],
                p["lidar"]["normalized_ranges"],
                controls,
                knowledge,
            )
        )


def test_native_passive_on_off_same_physics_commands_missions_and_integrity(tmp_path):
    pytest.importorskip("metadrive", reason="requires pinned native MetaDrive")
    from cav_recovery.d01.native import run_native

    off = run_native(tmp_path / "off", vehicles=4, horizon_s=0.4)
    on = run_native(tmp_path / "on", observer=True, vehicles=4, horizon_s=0.4)
    for key in (
        "logical_sha256",
        "physics_sha256",
        "missions",
        "arrivals",
        "departures",
        "integrity",
    ):
        assert off[key] == on[key], key
    assert on["observer_updates"] == 3 and off["observer_updates"] == 0
    assert on["observer_ms"]["p95"] is not None
    assert not any(on["integrity"].values())


def test_existing_dataset_or_experiment_cannot_be_overwritten(tmp_path):
    from cav_recovery.d01.native import run_native

    existing = tmp_path / "frozen"
    existing.mkdir()
    (existing / "campaign.json").write_text("{}")
    with pytest.raises(ValueError, match="dataset"):
        run_native(existing / "new")
    with pytest.raises(ValueError, match="new experimental"):
        run_native(existing)
