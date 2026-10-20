"""Native mission completion, constrained paths, retreat and honest counterexamples."""

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip(
    "metadrive", reason="Physical experiments require the pinned MetaDrive environment."
)

from shapely.geometry import box

from cav_recovery.metadrive.maneuvers import FeasibilityController
from cav_recovery.metadrive.native import geometry, make_env, missions_for
from cav_recovery.metadrive.observation import observe, rectangle
from cav_recovery.metadrive.physical import passage_certificate, probe
from cav_recovery.metadrive.run import run_metadrive
from cav_recovery.metadrive.scenes import PHYSICAL_SCENES


def lines(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def exact_passage_surface():
    return box(0, -1.7, 70, 1.7).union(box(0, 1.3, 22, 4.7))


@pytest.mark.parametrize("case", PHYSICAL_SCENES)
def test_predeclared_missions_and_distinct_native_geometry(case):
    states = []
    for _ in range(2):
        env = make_env(case, sensors=False)
        try:
            env.reset(seed=1)
            missions = missions_for(env)
            lanes, surface = geometry(env)
            rows = observe(env, 0, missions)
            assert set(rows) == set(PHYSICAL_SCENES[case]["agents"])
            for key, mission in missions.items():
                configured = PHYSICAL_SCENES[case]["agents"][key]
                assert mission.origin_lane == configured["spawn_lane_index"]
                assert mission.destination == configured["destination"]
                assert mission.checkpoints[-1] == configured["destination"]
                assert surface.covers(
                    rectangle(
                        rows[key]["position_xy_m"],
                        rows[key]["heading_rad"],
                        rows[key]["length_m"],
                        rows[key]["width_m"],
                    )
                )
            if case == "lane_merge":
                assert any(lane["index"][:2] == ["1y0_0_", "1y0_1_"] for lane in lanes)
                assert len(env.current_map.road_network.graph["1y0_0_"]["1y0_1_"]) == 1
            if case == "shared_passage":
                result = passage_certificate(rows, exact_passage_surface())
                assert (
                    result["status"] == "westward_retreat_required_for_joint_completion"
                )
                assert all(result["checks"].values())
                assert (
                    result["reverse_gear_necessity"]
                    == "conditional_on_nonholonomic_no_sideslip_model"
                )
            states.append((lanes, rows))
        finally:
            env.close()
    assert states[0] == states[1]


@pytest.mark.parametrize(
    "alteration",
    [
        "extra_east_bay",
        "wide_strip",
        "already_in_bay",
        "same_direction",
        "same_destination",
    ],
)
def test_order_argument_does_not_label_unproved_configurations(alteration):
    env = make_env("shared_passage", sensors=False)
    try:
        env.reset(seed=1)
        rows = observe(env, 0, missions_for(env))
    finally:
        env.close()
    surface = exact_passage_surface()
    options = {}
    if alteration == "extra_east_bay":
        surface = surface.union(box(40, 1.3, 65, 4.7))
    elif alteration == "wide_strip":
        surface = box(0, -2.1, 70, 2.1).union(box(0, 0.9, 22, 5.1))
        options["width"] = 4.2
    elif alteration == "already_in_bay":
        rows["agent0"]["body_xy_m"] = list(
            rectangle((10, 3), 0, 4.515, 1.852).exterior.coords
        )
    elif alteration == "same_direction":
        rows["agent1"]["heading_rad"] = 0.0
    else:
        rows["agent1"]["mission"]["destination"] = "east"
    assert passage_certificate(rows, surface, **options)["status"] == "indeterminate"


def test_frozen_path_witness_is_independent_of_guard_and_preserves_free_advance():
    poses = [(x / 20, 0, 0) for x in range(81)]
    result = probe(poses, 4.0, 2.0, box(-5, -3, 20, 3), {"other": box(3, -1, 7, 1)})
    assert result["status"] == "sampled_path_obstructed"
    assert result["blockers"] == ["other"]
    assert result["clear_sample_distance_m"] == pytest.approx(1.0)
    assert result["tested_distance_m"] == pytest.approx(1.05)
    assert not result["outside_surface"]
    free = probe(poses, 4.0, 2.0, box(-5, -3, 20, 3), {})
    assert free["status"] == "geometric_candidate_dynamics_unverified"


def test_probe_rejects_road_exit_without_inventing_a_vehicle_blocker():
    result = probe([(0, 0, 0), (0, 2, 0)], 4, 2, box(-5, -2, 5, 2), {})
    assert result["outside_surface"] and not result["blockers"]


def test_absent_path_is_unknown():
    result = probe([], 4, 2, box(-5, -2, 5, 2), {})
    assert result["status"] == "indeterminate_no_path_samples"
    assert (
        result["clear_sample_distance_m"] is None and result["outside_surface"] is None
    )


def test_physical_halt_evidence_does_not_require_crdg_or_v2v(tmp_path):
    result = run_metadrive(
        tmp_path / "without-graph",
        case="shared_passage",
        horizon_s=10,
        crdg=False,
        v2v=False,
        sensors=False,
    )
    assert result["status"] == "horizon_reached" and result["crdg"]["observations"] == 0
    assert all(
        episode["first_halted_s"] is not None
        for episode in result["physical_episodes_open_censored"]
    )
    assert (
        result["recovery_admissibility"]
        == "partial_execution_joint_completion_unverified"
    )


def test_bay_controller_stops_extrapolating_curvature_past_the_curve():
    vehicle = SimpleNamespace(
        position=(22, 0.5), heading_theta=0, config={"max_steering": 40}
    )
    # Past the S curve, positive y must command a turn back toward the road center.
    assert FeasibilityController.bay_tracking(vehicle, 8, 12, False, 3) < 0
    vehicle.position = (22, 0)
    assert FeasibilityController.bay_tracking(vehicle, 8, 12, False, 3) == 0


@pytest.fixture(scope="module")
def experiments(tmp_path_factory):
    root = tmp_path_factory.mktemp("physical-scenes")
    results = {}
    for case, horizon in (
        ("intersection_approaches", 240),
        ("lane_merge", 400),
        ("shared_passage", 180),
    ):
        for policy in ("road_following", "global_clearance"):
            folder = root / f"{case}-{policy}"
            result = run_metadrive(
                folder, case=case, policy=policy, seed=1, horizon_s=horizon
            )
            results[(case, policy)] = (folder, result)
    return results


@pytest.mark.parametrize(
    "case,policy",
    [
        (case, policy)
        for case in PHYSICAL_SCENES
        for policy in ("road_following", "global_clearance")
    ],
)
def test_complete_native_experiments_keep_all_missions_and_integrity(
    experiments, case, policy
):
    folder, result = experiments[(case, policy)]
    waiting = case == "shared_passage" and policy == "road_following"
    assert result["status"] == ("horizon_reached" if waiting else "completed"), result[
        "errors"
    ]
    assert not result["errors"] and not result["collision_contacts"]
    assert not result["footprint_overlaps"] and not result["outside_surface"]
    assert result["minimum_body_gap_m"] >= 0.7
    assert (
        result["scheduled"]
        == result["departed"]
        == len(PHYSICAL_SCENES[case]["agents"])
    )
    if waiting:
        assert not result["arrived"] and len(result["active"]) == 2
        assert result["simulation_time_s"] == result["horizon_s"]
        assert len(result["physical_episodes_open_censored"]) == 2
    else:
        assert len(result["arrived"]) == result["scheduled"] and not result["active"]
        assert all(
            event["native_arrive_dest"] and event["corridor_verified"]
            for event in result["events"]
        )
    manifest = json.loads((folder / "manifest.json").read_text())
    assert not manifest["code_changed_during_run"]
    assert "metadrive/physical.py" in manifest["code"]
    assert all(
        hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
        for name, digest in manifest["outputs"].items()
    )
    previous = {}
    for frame in lines(folder / "physics.jsonl"):
        assert (
            not frame["contacts"]
            and not frame["overlaps"]
            and not frame["outside_surface"]
        )
        for key, row in frame["vehicles"].items():
            assert (
                row["mission"]["destination"]
                == PHYSICAL_SCENES[case]["agents"][key]["destination"]
            )
            assert math.hypot(*row["velocity_xy_mps"]) < 1.6
            if key in previous:
                assert (
                    math.dist(previous[key], row["position_xy_m"]) < 0.04
                )  # 0.02 s native motion, no pose jumps.
            previous[key] = row["position_xy_m"]


def test_prepared_passage_forms_path_constraints_then_recovers_by_real_reverse(
    experiments,
):
    folder, waiting = experiments[("shared_passage", "road_following")]
    references = list(lines(folder / "physical_reference.jsonl"))
    assert all(
        row["probes"]["assigned_forward_4m"]["status"]
        == "geometric_candidate_dynamics_unverified"
        for row in references[0]["vehicles"].values()
    )
    last = references[-1]
    assert last["resources"] and any(
        resource["occupants"] for resource in last["resources"]
    )
    assert all(
        row["probes"]["assigned_forward_4m"]["status"] == "sampled_path_obstructed"
        for row in last["vehicles"].values()
    )
    assert all(
        row["probes"]["reverse_straight_4m"]["status"]
        == "geometric_candidate_dynamics_unverified"
        for row in last["vehicles"].values()
    )
    assert all(metric["halted_s"] > 150 for metric in waiting["progress"].values())
    recovered_folder, recovered = experiments[("shared_passage", "global_clearance")]
    assert recovered["progress"]["agent0"]["reverse_distance_m"] > 20
    assert len(recovered["arrived"]) == 2 and recovered["controller_stage"] == "finish"
    velocities = [
        frame["vehicles"]["agent0"]["signed_speed_mps"]
        for frame in lines(recovered_folder / "physics.jsonl")
        if "agent0" in frame["vehicles"]
    ]
    assert min(velocities) < -0.9 and max(velocities) > 1.1


@pytest.mark.parametrize("case", PHYSICAL_SCENES)
def test_existing_2d_view_uses_native_geometry_and_stable_labels(case):
    from cav_recovery.metadrive.view import TopDownView

    env = make_env(case, sensors=False)
    try:
        env.reset(seed=1)
        _, surface = geometry(env)
        xmin, ymin, xmax, ymax = surface.bounds
        view = TopDownView(
            window=False,
            camera_position=((xmin + xmax) / 2, (ymin + ymax) / 2),
            scaling=4,
        )
        env.engine.top_down_renderer = view
        image = env.render(mode="topdown")
        assert image.shape == (760, 1000, 3)
        assert view.labels == {key: f"V{i}" for i, key in enumerate(sorted(env.agents))}
        assert all(
            len(view.vehicle_pixels(vehicle)) == 4 for vehicle in env.agents.values()
        )
    finally:
        env.close()


def test_rendered_recovery_matches_headless_physics(experiments, monkeypatch, tmp_path):
    import cav_recovery.metadrive.view as view_module

    native_view = view_module.TopDownView
    monkeypatch.setattr(
        view_module, "TopDownView", lambda **kwargs: native_view(window=False, **kwargs)
    )
    folder = tmp_path / "rendered-recovery"
    result = run_metadrive(
        folder,
        case="shared_passage",
        policy="global_clearance",
        horizon_s=180,
        gui=True,
        gui_delay_ms=0,
    )
    reference_folder, reference = experiments[("shared_passage", "global_clearance")]
    assert result["status"] == "completed" and result["arrived"] == reference["arrived"]
    assert (folder / "physics.jsonl").read_bytes() == (
        reference_folder / "physics.jsonl"
    ).read_bytes()
    assert len(list(folder.glob("topdown_stage_*.png"))) >= 5


@pytest.mark.parametrize(
    "case,policy",
    [("shared_passage", "scheduled_passage"), ("crossing", "global_clearance")],
)
def test_incompatible_policies_are_rejected_before_outputs(case, policy, tmp_path):
    folder = tmp_path / "invalid"
    with pytest.raises(ValueError):
        run_metadrive(folder, case=case, policy=policy)
    assert not folder.exists()
