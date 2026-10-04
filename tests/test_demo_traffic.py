"""Contrôles du cycle de vie d'une démonstration, sans ouvrir d'interface graphique."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest


def load_cli(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def demo(monkeypatch, tmp_path):
    cli = load_cli("demo_traffic")
    mkdtemp = cli.tempfile.mkdtemp
    monkeypatch.setattr(cli.tempfile, "mkdtemp", Mock(side_effect=lambda **kwargs: mkdtemp(dir=tmp_path, **kwargs)))

    def prepare(osm, contract, output):
        output.mkdir()
        (output / "scenario.json").write_text("{}", encoding="utf-8")

    def run(scenario, regime, output, **kwargs):
        assert (scenario / "scenario.json").exists()
        output.mkdir()
        result = {"status": "passed", "reason": None, "connection_closed": True,
                  "process_stopped": True, "process_returncode": 0, "forced_process_stop": False,
                  "cleanup_errors": []}
        (output / "summary.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr(cli, "prepare_traffic", Mock(side_effect=prepare))
    monkeypatch.setattr(cli, "run_traffic", Mock(side_effect=run))
    monkeypatch.setattr("sys.argv", ["demo_traffic", "--osm", "synthetic.osm", "--contract", "synthetic.json",
                                     "--regime", "LOW"])
    return cli


@pytest.mark.parametrize("regime", ["LOW", "MID", "HIGH"])
def test_success_cleans_workspace_after_gui_run(demo, monkeypatch, capsys, regime):
    monkeypatch.setattr("sys.argv", ["demo_traffic", "--osm", "synthetic.osm", "--contract", "synthetic.json",
                                     "--regime", regime, "--gui-delay-ms", "7", "--drain-horizon-s", "300"])
    assert demo.main() == 0
    demo.tempfile.mkdtemp.assert_called_once_with(prefix="traffic-demo-")
    osm, contract, scenario = demo.prepare_traffic.call_args.args
    assert (osm, contract) == ("synthetic.osm", "synthetic.json")
    demo.run_traffic.assert_called_once_with(scenario, regime, scenario.parent / "result", gui=True,
                                             gui_delay_ms=7, drain_horizon_s=300)
    assert not scenario.parent.exists()
    assert "nettoyés" in capsys.readouterr().out


@pytest.mark.parametrize("field,value", [
    ("status", "failed"), ("connection_closed", False), ("process_stopped", False),
    ("process_returncode", 1), ("forced_process_stop", True), ("cleanup_errors", ["Erreur de fermeture"]),
    ("connection_closed", None),
])
def test_failure_or_unconfirmed_closure_preserves_workspace(demo, capsys, field, value):
    normal_run = demo.run_traffic.side_effect

    def run(*args, **kwargs):
        result = normal_run(*args, **kwargs)
        result[field] = value
        return result

    demo.run_traffic.side_effect = run
    assert demo.main() == 1
    workspace = demo.prepare_traffic.call_args.args[2].parent
    assert (workspace / "scenario/scenario.json").exists()
    assert (workspace / "result/summary.json").exists()
    captured = capsys.readouterr()
    assert str(workspace) in captured.err and "diagnostic" in captured.err
    assert "nettoyés" not in captured.out
    if field == "cleanup_errors":
        assert "Erreur de fermeture" in captured.err


def test_execution_exception_preserves_diagnostics(demo, capsys):
    def fail(scenario, regime, output, **kwargs):
        output.mkdir()
        (output / "sumo.log").write_text("Erreur TraCI", encoding="utf-8")
        raise OSError("Connexion interrompue")

    demo.run_traffic.side_effect = fail
    assert demo.main() == 1
    workspace = demo.prepare_traffic.call_args.args[2].parent
    assert (workspace / "result/sumo.log").read_text(encoding="utf-8") == "Erreur TraCI"
    error = capsys.readouterr().err
    assert str(workspace) in error and "Connexion interrompue" in error


@pytest.mark.parametrize("exists", [False, True])
def test_keep_artifacts_never_uses_or_cleans_temporary_workspace(demo, monkeypatch, tmp_path, capsys, exists):
    workspace = tmp_path / "kept"
    if exists:
        workspace.mkdir()
    monkeypatch.setattr("sys.argv", ["demo_traffic", "--osm", "synthetic.osm", "--contract", "synthetic.json",
                                     "--regime", "LOW", "--keep-artifacts", str(workspace)])
    monkeypatch.setattr(demo.shutil, "rmtree", Mock(side_effect=AssertionError("Suppression interdite")))
    assert demo.main() == 0
    demo.tempfile.mkdtemp.assert_not_called()
    demo.shutil.rmtree.assert_not_called()
    assert (workspace / "scenario/scenario.json").exists()
    assert (workspace / "result/summary.json").exists()
    assert str(workspace) in capsys.readouterr().out
    assert demo.run_traffic.call_args.kwargs == {"gui": True, "gui_delay_ms": 100, "drain_horizon_s": 600}


def test_nonempty_keep_directory_is_refused_without_changes(demo, monkeypatch, tmp_path):
    protected = tmp_path / "existing.txt"
    protected.write_text("à conserver", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["demo_traffic", "--osm", "synthetic.osm", "--contract", "synthetic.json",
                                     "--regime", "LOW", "--keep-artifacts", str(tmp_path)])
    assert demo.main() == 2
    demo.prepare_traffic.assert_not_called()
    demo.run_traffic.assert_not_called()
    demo.tempfile.mkdtemp.assert_not_called()
    assert list(tmp_path.iterdir()) == [protected]
    assert protected.read_text(encoding="utf-8") == "à conserver"


@pytest.mark.parametrize("input_error", [False, True])
def test_preparation_failure_preserves_diagnostics_without_running(demo, capsys, input_error):
    def fail(osm, contract, output):
        output.mkdir()
        (output / "conversion.log").write_text("Conversion interrompue", encoding="utf-8")
        error = demo.TrafficInputError if input_error else RuntimeError
        raise error("Préparation refusée")

    demo.prepare_traffic.side_effect = fail
    assert demo.main() == (2 if input_error else 1)
    demo.run_traffic.assert_not_called()
    workspace = demo.prepare_traffic.call_args.args[2].parent
    assert (workspace / "scenario/conversion.log").exists()
    error = capsys.readouterr().err
    assert str(workspace) in error and "Préparation refusée" in error


def test_cleanup_error_is_not_reported_as_success(demo, monkeypatch, capsys):
    monkeypatch.setattr(demo.shutil, "rmtree", Mock(side_effect=OSError("Nettoyage refusé")))
    assert demo.main() == 1
    workspace = demo.prepare_traffic.call_args.args[2].parent
    assert (workspace / "result/summary.json").exists()
    captured = capsys.readouterr()
    assert "Nettoyage refusé" in captured.err and str(workspace) in captured.err
    assert "Démonstration terminée" not in captured.out


@pytest.mark.parametrize("name,args", [
    ("prepare_traffic", ["--osm", "synthetic", "--contract", "synthetic"]),
    ("run_traffic", ["--scenario-dir", "synthetic", "--regime", "LOW"]),
])
def test_scientific_scripts_still_require_explicit_output_directory(monkeypatch, name, args):
    cli = load_cli(name)
    operation = Mock()
    monkeypatch.setattr(cli, name, operation)
    monkeypatch.setattr("sys.argv", [name, *args])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    operation.assert_not_called()
