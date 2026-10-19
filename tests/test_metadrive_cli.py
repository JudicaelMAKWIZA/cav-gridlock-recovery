"""Le nouveau lanceur ne change pas les commandes SUMO."""

from unittest.mock import Mock

import pytest

from cav_recovery import cli


def result(status="completed"):
    return dict(status=status, scheduled=2, departed=2, arrived={"a": 1, "b": 2}, active=[], simulation_time_s=2)


def test_metadrive_launch_options_are_forwarded(tmp_path):
    runner = Mock(return_value=result())
    assert cli.main(["metadrive", "run", "crossing", "--gui", "--seed", "2", "--horizon-s", "20",
                     "--output-dir", str(tmp_path), "--no-v2v"], runner=runner) == 0
    options = runner.call_args.kwargs
    assert options["gui"] and options["seed"] == 2 and not options["v2v"] and options["sensors"]
    assert options["horizon_s"] == 20 and options["policy"] == "scheduled_passage"


def test_headless_and_scenario_listing(capsys):
    runner = Mock(return_value=result("horizon_reached"))
    assert cli.main(["metadrive", "run", "convergence", "--headless"], runner=runner) == 3
    assert not runner.call_args.kwargs["gui"]
    runner.reset_mock()
    assert cli.main(["metadrive", "scenarios"], runner=runner) == 0
    runner.assert_not_called()
    assert "crossing_three" in capsys.readouterr().out


def test_metadrive_no_scene_or_incompatible_display_flags_rejected():
    with pytest.raises(SystemExit):
        cli.main(["metadrive", "run", "crossing", "--gui", "--headless"])


def test_old_advanced_run_keeps_sumo_options():
    runner = Mock(return_value={"status": "completed", "counts": {}})
    assert cli.main(["run", "clearance"], runner=runner) == 0
    assert runner.call_args.kwargs["scenario"] == "kintambo"


def test_help_does_not_import_metadrive_native(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "cav_recovery.metadrive.native", None)
    assert cli.main(["metadrive", "scenarios"]) == 0


@pytest.mark.parametrize("scene", ["intersection_approaches", "lane_merge", "shared_passage"])
def test_physical_scenes_default_to_driving_and_support_global_intervention(scene):
    runner = Mock(return_value=result())
    assert cli.main(["metadrive", "run", scene], runner=runner) == 0
    assert runner.call_args.kwargs["policy"] == "road_following"
    assert cli.main(["metadrive", "run", scene, "--policy", "global_clearance"], runner=runner) == 0
    assert runner.call_args.kwargs["policy"] == "global_clearance"
