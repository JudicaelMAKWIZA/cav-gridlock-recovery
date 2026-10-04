"""Contrôles du cycle de vie d'une démonstration, sans ouvrir d'interface graphique."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from cav_recovery.simulation import traffic_scenario as scenario


def load_cli(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def demo(monkeypatch, tmp_path):
    cli = load_cli("demo_traffic")
    mkdtemp = cli.tempfile.mkdtemp

    def temporary_workspace(suffix=None, prefix=None, dir=None):
        return mkdtemp(suffix=suffix, prefix=prefix, dir=tmp_path if dir is None else dir)

    monkeypatch.setattr(cli.tempfile, "mkdtemp", Mock(side_effect=temporary_workspace))

    def prepare(osm, contract, output, *, failure_diagnostics_dir=None):
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
    assert demo.prepare_traffic.call_args.kwargs == {"failure_diagnostics_dir": scenario.parent / "preparation-failure"}
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


@pytest.fixture
def preparation(monkeypatch):
    contract_path = Path(__file__).parent / "fixtures/traffic/demand.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(scenario, "read_contract", Mock(return_value=contract))
    monkeypatch.setattr(scenario, "check_environment", Mock(return_value={}))
    return scenario


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_real_preparation_failure_copies_stage_only_when_requested(preparation, monkeypatch, tmp_path,
                                                                  preserve, error_type):
    error = error_type("Conversion interrompue")
    stages = []

    def convert(osm, stage):
        stages.append(stage)
        (stage / "conversion.log").write_text("Conversion interrompue", encoding="utf-8")
        (stage / "LOW").mkdir()
        (stage / "LOW/traffic.rou.xml").write_text("<routes/>", encoding="utf-8")
        raise error

    monkeypatch.setattr(preparation, "convert_network", convert)
    output = tmp_path / "scenario"
    diagnostics = tmp_path / "preparation-failure"
    options = {"failure_diagnostics_dir": diagnostics} if preserve else {}
    with pytest.raises(error_type) as caught:
        preparation.prepare_traffic("synthetic", "synthetic", output, **options)
    assert caught.value is error
    assert len(stages) == 1 and not stages[0].exists()
    assert not output.exists()
    assert not list(tmp_path.glob(".traffic-*"))
    if preserve:
        assert (diagnostics / "conversion.log").read_text(encoding="utf-8") == "Conversion interrompue"
        assert (diagnostics / "LOW/traffic.rou.xml").read_text(encoding="utf-8") == "<routes/>"
        assert not (diagnostics / "scenario.json").exists()
    else:
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("copy_fails", [False, True])
def test_demo_real_preparation_failure_keeps_stage_diagnostics(demo, preparation, monkeypatch, capsys, copy_fails):
    error = RuntimeError("Conversion SUMO en échec")
    stages = []

    def convert(osm, stage):
        stages.append(stage)
        (stage / "conversion.log").write_text("Conversion interrompue", encoding="utf-8")
        raise error

    monkeypatch.setattr(preparation, "convert_network", convert)
    monkeypatch.setattr(demo, "prepare_traffic", preparation.prepare_traffic)
    if copy_fails:
        monkeypatch.setattr(preparation.shutil, "copytree", Mock(side_effect=OSError("Copie refusée")))
    assert demo.main() == 1
    demo.run_traffic.assert_not_called()
    workspace = stages[0].parent
    assert workspace.exists() and not stages[0].exists()
    assert not (workspace / "scenario").exists()
    captured = capsys.readouterr()
    assert str(workspace) in captured.err and "Conversion SUMO en échec" in captured.err
    if copy_fails:
        assert "Copie refusée" in captured.err
        assert "Copie refusée" in error.__notes__[0]
    else:
        assert (workspace / "preparation-failure/conversion.log").read_text(encoding="utf-8") == "Conversion interrompue"
        assert not (workspace / "preparation-failure/scenario.json").exists()


def test_demo_interrupt_preserves_real_preparation_diagnostics(demo, preparation, monkeypatch, capsys):
    stages = []

    def convert(osm, stage):
        stages.append(stage)
        (stage / "conversion.log").write_text("Interruption", encoding="utf-8")
        raise KeyboardInterrupt

    monkeypatch.setattr(preparation, "convert_network", convert)
    monkeypatch.setattr(demo, "prepare_traffic", preparation.prepare_traffic)
    assert demo.main() == 130
    demo.run_traffic.assert_not_called()
    workspace = stages[0].parent
    assert not stages[0].exists() and not (workspace / "scenario").exists()
    assert (workspace / "preparation-failure/conversion.log").read_text(encoding="utf-8") == "Interruption"
    assert str(workspace) in capsys.readouterr().err


def test_diagnostic_copy_error_preserves_original_exception(preparation, monkeypatch, tmp_path):
    error = RuntimeError("Erreur de préparation originale")

    def convert(osm, stage):
        (stage / "conversion.log").write_text("Erreur originale", encoding="utf-8")
        raise error

    monkeypatch.setattr(preparation, "convert_network", convert)
    monkeypatch.setattr(preparation.shutil, "copytree", Mock(side_effect=OSError("Copie refusée")))
    with pytest.raises(RuntimeError) as caught:
        preparation.prepare_traffic("synthetic", "synthetic", tmp_path / "scenario",
                                    failure_diagnostics_dir=tmp_path / "diagnostics")
    assert caught.value is error
    assert "Copie refusée" in error.__notes__[0]
    assert not (tmp_path / "scenario").exists()
    assert not list(tmp_path.glob(".traffic-*"))


def test_nonempty_diagnostics_refused_before_preparation(preparation, tmp_path):
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    protected = diagnostics / "conversion.log"
    protected.write_text("à conserver", encoding="utf-8")
    with pytest.raises(preparation.TrafficInputError):
        preparation.prepare_traffic("synthetic", "synthetic", tmp_path / "scenario",
                                    failure_diagnostics_dir=diagnostics)
    preparation.read_contract.assert_not_called()
    assert protected.read_text(encoding="utf-8") == "à conserver"
    assert not (tmp_path / "scenario").exists()


def test_empty_diagnostics_directory_accepts_failure_copy(preparation, monkeypatch, tmp_path):
    output = tmp_path / "scenario"
    output.mkdir()
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()

    def convert(osm, stage):
        (stage / "conversion.log").write_text("Conversion interrompue", encoding="utf-8")
        raise RuntimeError("Conversion interrompue")

    monkeypatch.setattr(preparation, "convert_network", convert)
    with pytest.raises(RuntimeError, match="Conversion interrompue"):
        preparation.prepare_traffic("synthetic", "synthetic", output, failure_diagnostics_dir=diagnostics)
    assert not list(output.iterdir())
    assert (diagnostics / "conversion.log").read_text(encoding="utf-8") == "Conversion interrompue"
    assert not list(tmp_path.glob(".traffic-*"))


@pytest.mark.parametrize("location", ["same", "inside", "parent"])
def test_diagnostics_cannot_be_published_as_partial_scenario(preparation, tmp_path, location):
    output = tmp_path / "scenario"
    diagnostics = {"same": output, "inside": output / "diagnostics", "parent": tmp_path}[location]
    with pytest.raises(preparation.TrafficInputError, match="séparés"):
        preparation.prepare_traffic("synthetic", "synthetic", output, failure_diagnostics_dir=diagnostics)
    preparation.read_contract.assert_not_called()
    assert not output.exists()


def test_failure_before_staging_creates_no_diagnostics(demo, preparation, monkeypatch, tmp_path, capsys):
    preparation.read_contract.side_effect = preparation.TrafficInputError("Contrat invalide")
    monkeypatch.setattr(demo, "prepare_traffic", preparation.prepare_traffic)
    assert demo.main() == 2
    demo.run_traffic.assert_not_called()
    preparation.check_environment.assert_not_called()
    workspace, = tmp_path.iterdir()
    assert not list(workspace.iterdir())
    captured = capsys.readouterr()
    assert "Contrat invalide" in captured.err and str(workspace) in captured.err


def test_success_keeps_atomic_publication_without_creating_diagnostics(preparation, monkeypatch, tmp_path):
    def convert(osm, stage):
        (stage / "network.net.xml").write_text("<net/>", encoding="utf-8")
        return {}

    monkeypatch.setattr(preparation, "convert_network", convert)
    monkeypatch.setattr(preparation, "inspect_network", lambda *args: {
        "center": {"x": "0", "y": "0"}, "routes": preparation.ROUTES,
        "gate_mapping": {**preparation.GATE_EDGES, "entry_connector": ":2725672310_0"}})
    monkeypatch.setattr(preparation, "build_scenery", lambda *args: {})
    diagnostics = tmp_path / "diagnostics"
    baseline = preparation.prepare_traffic("synthetic", "synthetic", tmp_path / "baseline")
    output = tmp_path / "scenario"
    output.mkdir()
    result = preparation.prepare_traffic("synthetic", "synthetic", output, failure_diagnostics_dir=diagnostics)
    assert result == baseline
    assert preparation.read_scenario(output) == result
    assert not diagnostics.exists()
    assert not list(tmp_path.glob(".traffic-*"))
