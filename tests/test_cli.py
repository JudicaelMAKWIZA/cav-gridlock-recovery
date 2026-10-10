"""Le menu, les raccourcis et le lanceur utilisent la même exécution."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cav_recovery import cli


@pytest.mark.parametrize("name", ["crossing", "junction_adverse", "clearance"])
def test_shortcut_opens_current_gui_with_explicit_defaults(name, tmp_path):
    runner = Mock(return_value={"status": "completed", "counts": {}})
    assert cli.main([name, "--output-dir", str(tmp_path)], runner=runner) == 0
    args = runner.call_args.kwargs
    assert args["kintambo_case"] == name and args["gui"] and args["crdg_live"]
    assert args["seed"] == 1 and args["gui_delay_ms"] == 300 and not args["close_on_end"]


def test_shortcut_can_override_gui_seed_and_delay(tmp_path):
    runner = Mock(return_value={"status": "horizon_reached", "counts": {}})
    assert cli.main(["crossing", "--no-gui", "--seed", "2", "--gui-delay-ms", "10", "--output-dir", str(tmp_path)], runner=runner) == 3
    args = runner.call_args.kwargs
    assert not args["gui"] and not args["crdg_live"] and args["seed"] == 2 and args["gui_delay_ms"] == 10


def test_no_arguments_without_a_terminal_never_waits_for_input(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    blocked = Mock(side_effect=AssertionError("input interdit"))
    monkeypatch.setattr("builtins.input", blocked)
    assert cli.main([]) == 0
    assert "terminal interactif" in capsys.readouterr().out
    blocked.assert_not_called()


def test_menu_lists_real_cases_and_runs_the_same_pipeline(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    names = ("kintambo", *cli.case_names())
    values = iter(["2", "", "3", "", "1", str(names.index("crossing") + 1), "", "", "", "", "0"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(values))
    runner = Mock(return_value={"status": "completed", "counts": {}})
    assert cli.main([], runner=runner) == 0
    runner.assert_called_once()
    args = runner.call_args.kwargs
    assert args["kintambo_case"] == "crossing" and args["gui"] and args["seed"] == 1
    assert args["gui_delay_ms"] == 300 and args["output_mode"] == "full"
    text = capsys.readouterr().out
    assert all(name in text for name in names) and "non implémentés" in text


@pytest.mark.parametrize("choice,title,content", [
    ("2", "SCÉNARIOS DISPONIBLES", "junction_adverse"),
    ("3", "AIDE CAV GRIDLOCK RECOVERY", "cgr run crossing --help"),
])
def test_information_remains_displayed_until_explicit_return(monkeypatch, capsys, choice, title, content):
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    prompts = []
    def reply(prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return choice
        if len(prompts) == 2:
            assert "Entrée pour revenir" in prompt
            text = capsys.readouterr().out
            assert title in text and content in text
            assert text.count("1. Lancer SUMO") == 1
            return ""
        return "0"
    monkeypatch.setattr("builtins.input", reply)
    runner = Mock()
    assert cli.main([], runner=runner) == 0
    assert len(prompts) == 3 and "Votre choix" in prompts[-1]
    runner.assert_not_called()


@pytest.mark.parametrize("error", [EOFError(), KeyboardInterrupt()])
def test_menu_eof_or_interrupt_exits_cleanly(monkeypatch, error):
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", Mock(side_effect=error))
    assert cli.main([]) == 0


def test_menu_unknown_choice_and_return_do_not_launch(monkeypatch):
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    values = iter(["99", "1", "0", "0"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(values))
    runner = Mock()
    assert cli.main([], runner=runner) == 0
    runner.assert_not_called()


def test_integer_prompt_retries_invalid_input(monkeypatch):
    values = iter(["-1", "abc", "2"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(values))
    assert cli.prompt_integer("Seed", 1) == 2


def project(tmp_path):
    root = tmp_path / "projet avec espaces"
    (root / ".venv/bin").mkdir(parents=True)
    (root / ".venv/bin/python").write_text("python")
    (root / "src/cav_recovery").mkdir(parents=True)
    (root / "src/cav_recovery/cli.py").write_text("")
    (root / "pyproject.toml").write_text("")
    return root


@pytest.mark.skipif(cli.os.name != "posix", reason="Lanceur Ubuntu uniquement.")
def test_launcher_does_not_change_shell_configuration_by_default(tmp_path, monkeypatch):
    root, home = project(tmp_path), tmp_path / "home"
    home.mkdir()
    startup = home / ".bashrc"
    startup.write_bytes(b"# configuration existante\n")
    monkeypatch.setattr(cli.shutil, "which", lambda _: None)
    assert cli.install_launcher(root, home=home) == 0
    script = (home / ".local/bin/cgr").read_text()
    assert str(root / ".venv/bin/python") in script and "-m cav_recovery.cli" in script
    assert "exec" in script and "setStop" not in script
    assert startup.read_bytes() == b"# configuration existante\n"


@pytest.mark.skipif(cli.os.name != "posix", reason="Lanceur Ubuntu uniquement.")
def test_explicit_shell_path_is_idempotent_and_preserves_existing_content(tmp_path, monkeypatch):
    root, home = project(tmp_path), tmp_path / "home"
    home.mkdir()
    (home / ".bashrc").write_bytes(b"# existant\n")
    monkeypatch.setattr(cli.shutil, "which", lambda _: None)
    cli.install_launcher(root, home=home, shell_path=True)
    first = (home / ".bashrc").read_bytes()
    cli.install_launcher(root, home=home, shell_path=True)
    assert (home / ".bashrc").read_bytes() == first
    assert first.startswith(b"# existant\n") and first.count(b'export PATH="$HOME/.local/bin:$PATH"') == 1
    assert (home / ".profile").read_bytes().count(b'export PATH="$HOME/.local/bin:$PATH"') == 1


@pytest.mark.skipif(cli.os.name != "posix", reason="Lanceur Ubuntu uniquement.")
def test_shell_setup_uses_existing_login_profile_without_creating_a_competing_one(tmp_path, monkeypatch):
    root, home = project(tmp_path), tmp_path / "home"
    home.mkdir()
    (home / ".bash_profile").write_bytes(b"# profil personnel\n")
    monkeypatch.setattr(cli.shutil, "which", lambda _: None)
    cli.install_launcher(root, home=home, shell_path=True)
    assert (home / ".bash_profile").read_bytes().startswith(b"# profil personnel\n")
    assert not (home / ".profile").exists()


@pytest.mark.skipif(cli.os.name != "posix", reason="Lanceur Ubuntu uniquement.")
@pytest.mark.parametrize("foreign", ["file", "path"])
def test_installer_never_overwrites_or_shadows_another_command(tmp_path, monkeypatch, foreign):
    root, home = project(tmp_path), tmp_path / "home"
    target = home / ".local/bin/cgr"
    target.parent.mkdir(parents=True)
    if foreign == "file":
        target.write_bytes(b"autre commande")
    monkeypatch.setattr(cli.shutil, "which", lambda _: None if foreign == "file" else "/usr/bin/cgr")
    with pytest.raises(ValueError, match="existe|remplacée"):
        cli.install_launcher(root, home=home, shell_path=True)
    assert not (home / ".bashrc").exists()
    if foreign == "file":
        assert target.read_bytes() == b"autre commande"
    else:
        assert not target.exists()
