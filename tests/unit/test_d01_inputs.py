import copy
import pytest

from cav_recovery.d01.inputs import adapt, InputError, message_age, plain
from tests.support.d01_fixtures import packet


def test_detached_immutable_input_and_no_hidden_world_dependency():
    p = packet()
    before = copy.deepcopy(p)
    a = adapt(p)
    assert plain(a.data) == before
    p["ego"]["position_xy_m"][0] = 999
    assert a.data["ego"]["position_xy_m"] == (0.0, 0.0)
    with pytest.raises(TypeError):
        a.data["ego"]["position_xy_m"] = None
    assert plain(adapt(before).data) == plain(a.data)


@pytest.mark.parametrize(
    "section", [None, "ego", "mission", "map", "lidar", "own_controls", "v2v"]
)
def test_privileged_fields_rejected_at_every_boundary(section):
    p = packet()
    target = p if section is None else p[section]
    target["witnesses"] = {"guard_blockers": ["secret"]}
    with pytest.raises(InputError):
        adapt(p)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["ego"].update(signed_speed_mps=float("nan")),
        lambda p: p["ego"].update(length_m=True),
        lambda p: p["ego"].update(position_xy_m=[0]),
        lambda p: p["ego"].update(available_at_s=0.6),
        lambda p: p["map"].update(coordinate_frame="sumo_xy_m"),
        lambda p: p["lidar"].update(num_lasers=72),
        lambda p: p["lidar"].update(normalized_ranges=[1.0] * 15),
        lambda p: p["lidar"]["normalized_ranges"].__setitem__(0, -0.1),
        lambda p: p["own_controls"].update(native_action=[2.0, 0.0]),
        lambda p: p["v2v"]["received"][0]["message"].update(dependencies=[]),
        lambda p: p["v2v"]["received"][0]["message"].update(observed_at_s=0.5),
        lambda p: p["v2v"]["received"][0]["message"].update(confidence="oracle"),
        lambda p: p["v2v"]["received"].append(copy.deepcopy(p["v2v"]["received"][0])),
    ],
)
def test_invalid_provenance_shape_future_and_privilege_are_not_sanitized(change):
    p = packet()
    change(p)
    with pytest.raises(InputError):
        adapt(p)


def test_valid_historical_expired_and_absent_are_different():
    entry = adapt(packet()).data["v2v"]["received"][0]
    assert message_age(entry, 0.4) == "current"
    assert message_age(entry, 0.6) == "historical"
    assert message_age(entry, 1.6) == "historical"
    assert message_age(entry, 1.8) == "expired"
    p = packet()
    p["v2v"]["received"] = []
    p["lidar"]["normalized_ranges"][0] = None
    assert adapt(p).data["lidar"]["normalized_ranges"][0] is None
    assert adapt(p).data["v2v"]["received"] == ()
    p["lidar"] = None
    p["v2v"] = None
    assert adapt(p).data["v2v"] is None


def test_no_file_or_engine_access_required(monkeypatch):
    import builtins

    def forbidden(*a, **k):
        raise AssertionError("forbidden file access")

    monkeypatch.setattr(builtins, "open", forbidden)
    assert adapt(packet()).ego_id == "V0"
