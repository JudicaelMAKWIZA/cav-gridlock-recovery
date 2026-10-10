"""Protège la pause, les fermetures explicites et la commande commune."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cav_recovery import cli
from cav_recovery.simulation.crdg_live import CrdgPanelLink, atomic_json
from cav_recovery.simulation.sumo_view import visible_labels
from test_crdg_live import missions, current, native, graph_of, camera_link


def test_all_labels_include_ordinary_cars_and_restore_custom_labels(tmp_path):
    link, connection = CrdgPanelLink(tmp_path, missions()), native()
    connection.vehicle.getParameter.return_value = "original"
    link.native_view = SimpleNamespace(size=(1000, 1000), redraw=Mock())
    connection.gui = SimpleNamespace(getBoundary=lambda _: ((0, 0), (100, 100)))
    atomic_json(link.directory / "control.json", {"labels": "all"})
    readings = {m.vehicle_id: {"position": (10 + i * 10, 10)} for i, m in enumerate(missions())}
    link.observe(connection, graph_of(current(edge=False)), current(edge=False), readings)
    assert set(link.labels.values()) == {"V001", "V002", "V003"}
    link.observe(connection, graph_of(current()), current(), readings)
    link.restore(connection, readings)
    assert all(call.args[2] == "original" for call in connection.vehicle.setParameter.call_args_list[-3:])


def test_panel_exit_without_close_request_is_an_error(tmp_path):
    link = CrdgPanelLink(tmp_path, missions())
    link.process = Mock()
    link.process.poll.return_value = 0
    with pytest.raises(RuntimeError, match="sans demande explicite"):
        link.control()
    assert not link.stats["closed_by_user"]


def test_terminal_view_holds_without_a_terminal_and_never_steps(tmp_path, monkeypatch):
    link, connection, readings = camera_link(tmp_path)
    monkeypatch.setattr("cav_recovery.simulation.crdg_live.sys.stdin", SimpleNamespace(isatty=lambda: False))
    calls = []
    def close_after_inspection(seconds):
        calls.append(seconds)
        assert json.loads((link.directory / "clock.json").read_text())["status"] == "horizon_reached"
        assert json.loads((link.directory / "frozen.json").read_text())["time_s"] == 9
        atomic_json(link.directory / "control.json", {"close_requested": True})
    monkeypatch.setattr("cav_recovery.simulation.crdg_live.time.sleep", close_after_inspection)
    link.hold_view(connection, readings, "horizon_reached", 9)
    assert calls == [.1] and link.stats["closed_by_user"]
    assert not hasattr(connection, "simulationStep")


def test_failed_sumo_leaves_local_inspection_without_native_calls(tmp_path, monkeypatch):
    link, connection, readings = camera_link(tmp_path)
    link.sumo_process = Mock()
    link.sumo_process.poll.return_value = 1
    monkeypatch.setattr("cav_recovery.simulation.crdg_live.sys.stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr("cav_recovery.simulation.crdg_live.time.sleep",
                        lambda _: atomic_json(link.directory / "control.json", {"close_requested": True}))
    link.hold_view(connection, readings, "failed", 9, reason="Erreur testée")
    assert not connection.gui.method_calls and not connection.vehicle.method_calls
    assert json.loads((link.directory / "clock.json").read_text())["reason"] == "Erreur testée"


def test_labels_all_declutter_secondary_cars_and_prioritize_selection():
    aliases = {"a": "V001", "b": "V002", "c": "V003"}
    readings = {item: {"position": (50, 50)} for item in aliases}
    assert visible_labels(aliases, readings, {"b"}, "all", ((0, 0), (100, 100)), (1000, 1000)) == {"b"}
    assert visible_labels(aliases, readings, set(), "all", ((0, 0), (100, 100)), (1000, 1000)) == {"a"}
    readings["c"]["position"] = (80, 80)
    assert visible_labels(aliases, readings, {"b"}, "all", ((0, 0), (100, 100)), (1000, 1000)) == {"b", "c"}


def test_labels_none_keep_selection_data_without_displaying_it():
    aliases = {"a": "V001"}
    readings = {"a": {"position": (50, 50)}}
    assert visible_labels(aliases, readings, {"a"}, "selected") == {"a"}
    assert visible_labels(aliases, readings, {"a"}, "none") == set()
    assert aliases == {"a": "V001"} and readings["a"]["position"] == (50, 50)


def test_official_command_opens_both_views_on_one_pipeline(tmp_path):
    runner = Mock(return_value={"status": "completed", "counts": {}})
    assert cli.main(["run", "crossing", "--gui", "--output-dir", str(tmp_path)], runner=runner) == 0
    args = runner.call_args.kwargs
    assert args["gui"] and args["crdg_live"] and args["crdg"]
    assert args["kintambo_case"] == "crossing" and not args["close_on_end"]


def test_only_available_commands_and_cases_are_advertised(capsys):
    assert cli.main(["scenarios"]) == 0
    output = capsys.readouterr().out
    assert "crossing" in output and "junction_adverse" in output and "Poisson" in output
    with pytest.raises(SystemExit):
        cli.main(["evaluate"])


def test_retired_interface_gives_migration_instead_of_silent_remapping():
    with pytest.raises(SystemExit):
        cli.main(["--scenario", "mutual_yield"], legacy=True)


def test_interactive_mode_is_not_silently_full(monkeypatch, tmp_path):
    runner = Mock(return_value={"status": "completed", "counts": {}})
    assert cli.main(["run", "clearance", "--output-mode", "interactive", "--output-dir", str(tmp_path)], runner=runner) == 0
    assert runner.call_args.kwargs["output_mode"] == "interactive"


def test_callback_error_stays_visible_and_is_reported_to_the_run():
    from cav_recovery.simulation.crdg_panel import InfoPanel
    panel = InfoPanel.__new__(InfoPanel)
    panel.end_reason, panel.command = Mock(), Mock()
    panel.callback_error(ValueError, ValueError("Erreur testée"), None)
    assert "Erreur testée" in panel.end_reason.set.call_args.args[0]
    panel.command.assert_called_once_with(ui_error="Erreur testée")


@pytest.mark.parametrize("mode", ["selected", "all"])
def test_labels_near_view_boundary_are_hidden_not_partially_drawn(mode):
    aliases = {"a": "V001", "b": "V002", "c": "V003"}
    readings = {"a": {"position": (1, 50)}, "b": {"position": (50, 50)}, "c": {"position": (110, 50)}}
    assert visible_labels(aliases, readings, set(aliases), mode, ((0, 0), (100, 100)), (1000, 1000)) == {"b"}
    assert len(aliases) == len(readings) == 3


def test_installed_package_provenance_does_not_require_git(monkeypatch):
    from cav_recovery.simulation.traffic_run import code_provenance
    monkeypatch.setattr("cav_recovery.simulation.traffic_run.subprocess.run", Mock(side_effect=FileNotFoundError("git")))
    result = code_provenance()
    assert len(result["sha256"]) == 64 and result["git_commit"] is None and result["git_state"] is None
