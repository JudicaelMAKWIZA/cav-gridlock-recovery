"""Tests du trajet SUMO et de ses erreurs, sans données de trafic réelles."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cav_recovery.simulation import sumo_smoke
from cav_recovery.simulation.sumo_process import close_sumo


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/sumo_smoke"
VEHICLE = "smoke_car"
ROUTE = ["approach", "destination"]


class TripConnection:
    """Simule un départ, un passage à destination puis une arrivée."""

    def __init__(self):
        self.frames = [
            {"time": 0.5, "active": [VEHICLE], "departed": [VEHICLE], "road": "approach", "index": 0},
            {"time": 1.0, "active": [VEHICLE], "road": "destination", "index": 1},
            {"time": 1.5, "active": [], "arrived": [VEHICLE]},
        ]
        self.frame = {}
        self.next_frame = 0
        self.simulationStep = Mock(side_effect=self.advance)
        self.getVersion = Mock(return_value=(1, "SUMO test"))
        self.close = Mock()
        self.route = SimpleNamespace(getEdges=Mock(return_value=ROUTE))
        self.simulation = SimpleNamespace(
            getTime=lambda: self.frame["time"],
            getDepartedIDList=Mock(side_effect=lambda: self.frame.get("departed", [])),
            getArrivedIDList=Mock(side_effect=lambda: self.frame.get("arrived", [])),
            getStartingTeleportIDList=Mock(side_effect=lambda: self.frame.get("teleport_starts", [])),
            getEndingTeleportIDList=Mock(side_effect=lambda: self.frame.get("teleport_ends", [])),
        )
        self.vehicle = SimpleNamespace(
            getIDList=lambda: self.frame["active"],
            getPosition=lambda vehicle: self.frame.get("position", (10.0, -1.6)),
            getRoute=lambda vehicle: self.frame.get("route", ROUTE),
            getRoadID=lambda vehicle: self.frame["road"],
            getRouteIndex=lambda vehicle: self.frame["index"],
        )

    def advance(self):
        self.frame = self.frames[self.next_frame]
        self.next_frame += 1


@pytest.fixture
def environment(monkeypatch):
    connection = TripConnection()
    process = Mock(returncode=0)
    process.wait.return_value = 0
    process.poll.return_value = 0
    popen = Mock(return_value=process)
    connect = Mock(return_value=connection)
    monkeypatch.setattr(sumo_smoke.shutil, "which", lambda name: "sumo-test")
    monkeypatch.setattr(sumo_smoke.subprocess, "Popen", popen)
    monkeypatch.setattr(sumo_smoke.importlib, "import_module",
                        lambda name: SimpleNamespace(connect=connect, __version__="test"))
    return SimpleNamespace(connection=connection, process=process, popen=popen, connect=connect)


def test_normal_trip_and_cleanup(environment, monkeypatch):
    close = Mock(wraps=close_sumo)
    monkeypatch.setattr(sumo_smoke, "close_sumo", close)
    result = sumo_smoke.run_sumo_smoke(FIXTURE, horizon_s=1.5)
    close.assert_called_once_with(environment.connection, environment.process, result)
    assert result["status"] == "passed"
    assert result["steps"] == 3
    assert result["departures"] == [{"vehicle_id": VEHICLE, "time_s": 0.5}]
    assert result["arrivals"] == [{"vehicle_id": VEHICLE, "time_s": 1.5}]
    assert result["position_reads"] == 2
    assert result["first_observation"]["road_id"] == "approach"
    assert result["last_observation"]["route"] == ROUTE
    assert result["destination_confirmed"] == "destination"
    assert result["teleport_starts"] == result["teleport_ends"] == []
    assert result["connection_closed"] and result["process_stopped"]
    assert not result["forced_process_stop"]
    environment.connection.close.assert_called_once_with(False)
    environment.process.wait.assert_called_once_with(timeout=5)
    command = environment.popen.call_args.args[0]
    assert command[command.index("--time-to-teleport") + 1] == "-1"
    assert command[command.index("--seed") + 1] == "0"
    for getter in (environment.connection.simulation.getStartingTeleportIDList,
                   environment.connection.simulation.getEndingTeleportIDList):
        assert getter.call_count == 3


@pytest.mark.parametrize("event", ["teleport_starts", "teleport_ends"])
def test_teleport_even_with_arrival_fails(environment, event):
    environment.connection.frames[-1][event] = [VEHICLE]
    result = sumo_smoke.run_sumo_smoke(FIXTURE)
    assert result["status"] == "failed"
    assert "Téléportation" in result["reason"]
    assert result[event] == [{"vehicle_id": VEHICLE, "time_s": 1.5}]
    assert result["arrivals"]
    assert result["connection_closed"] and result["process_stopped"]


@pytest.mark.parametrize("problem, message", [
    ("missing_departure", "départ unique"),
    ("wrong_route", "route observée"),
    ("wrong_destination", "destination"),
    ("wrong_route_index", "destination"),
    ("nonfinite_position", "non finie"),
    ("disappearance", "Disparition"),
    ("no_position", "destination"),
    ("still_active_on_arrival", "destination"),
    ("simultaneous_events", "départ unique"),
])
def test_trip_inconsistencies_fail_and_close(environment, problem, message):
    frames = environment.connection.frames
    if problem == "missing_departure":
        frames[0]["departed"] = []
    elif problem == "wrong_route":
        frames[0]["route"] = ["other", "destination"]
    elif problem == "wrong_destination":
        frames[1]["road"] = "approach"
    elif problem == "wrong_route_index":
        frames[1]["index"] = 0
    elif problem == "nonfinite_position":
        frames[0]["position"] = (float("nan"), 0)
    elif problem == "disappearance":
        frames[1]["active"] = []
    elif problem == "no_position":
        for frame in frames:
            frame["active"] = []
    elif problem == "still_active_on_arrival":
        frames[-1].update(active=[VEHICLE], road="destination", index=1)
    elif problem == "simultaneous_events":
        frames[0].update(active=[], arrived=[VEHICLE])
    result = sumo_smoke.run_sumo_smoke(FIXTURE)
    assert result["status"] == "failed"
    assert message in result["reason"]
    assert result["destination_confirmed"] is None
    assert result["connection_closed"] and result["process_stopped"]


def test_invalid_loaded_route(environment):
    environment.connection.route.getEdges.return_value = ["missing"]
    result = sumo_smoke.run_sumo_smoke(FIXTURE)
    assert result["status"] == "failed"
    assert "route chargée" in result["reason"]
    assert result["steps"] == 0
    assert result["connection_closed"] and result["process_stopped"]


@pytest.mark.parametrize("departed", [True, False])
def test_horizon_without_arrival(environment, departed):
    if not departed:
        for frame in environment.connection.frames:
            frame.update(active=[], departed=[])
    result = sumo_smoke.run_sumo_smoke(FIXTURE, horizon_s=1.0)
    assert result["status"] == "failed"
    assert "Horizon atteint" in result["reason"]
    assert result["steps"] == result["max_steps"] == 2
    assert result["simulation_time_s"] == 1.0
    assert result["connection_closed"] and result["process_stopped"]


def test_traci_error_still_closes(environment):
    environment.connection.simulationStep.side_effect = RuntimeError("Erreur TraCI simulée")
    result = sumo_smoke.run_sumo_smoke(FIXTURE)
    assert result["status"] == "failed"
    assert "Erreur TraCI simulée" in result["reason"]
    assert result["connection_closed"] and result["process_stopped"]


def test_connection_refused_stops_launched_process(environment):
    environment.connect.side_effect = ConnectionRefusedError("Connexion refusée")
    environment.process.wait.side_effect = [subprocess.TimeoutExpired("sumo", 5), 0]
    result = sumo_smoke.run_sumo_smoke(FIXTURE)
    assert result["status"] == "failed"
    assert "Connexion refusée" in result["reason"]
    assert not result["connected"]
    assert result["forced_process_stop"] and result["process_stopped"]
    environment.process.terminate.assert_called_once()


def test_close_error_cannot_validate_trip(environment):
    environment.connection.close.side_effect = RuntimeError("Fermeture refusée")
    environment.process.wait.side_effect = [
        subprocess.TimeoutExpired("sumo", 5), subprocess.TimeoutExpired("sumo", 5), 0,
    ]
    result = sumo_smoke.run_sumo_smoke(FIXTURE)
    assert result["status"] == "failed"
    assert result["arrivals"]
    assert result["cleanup_errors"]
    assert not result["connection_closed"]
    assert result["process_stopped"]
    environment.process.terminate.assert_called_once()
    environment.process.kill.assert_called_once()


@pytest.mark.parametrize("stop_confirmed", [True, False])
def test_wait_system_error_still_attempts_process_stop(environment, stop_confirmed):
    environment.process.wait.side_effect = (
        [OSError("Attente initiale impossible"), 0] if stop_confirmed else [
            OSError("Attente initiale impossible"),
            OSError("Attente après terminate impossible"),
            OSError("Attente après kill impossible"),
        ]
    )
    environment.process.poll.return_value = 0 if stop_confirmed else None
    environment.process.returncode = 0 if stop_confirmed else None

    result = sumo_smoke.run_sumo_smoke(FIXTURE)

    assert result["status"] == "failed"
    assert any("Attente initiale impossible" in error for error in result["cleanup_errors"])
    assert result["forced_process_stop"]
    assert result["process_stopped"] is stop_confirmed
    assert result["process_returncode"] == (0 if stop_confirmed else None)
    environment.process.terminate.assert_called_once()
    if stop_confirmed:
        environment.process.kill.assert_not_called()
    else:
        environment.process.kill.assert_called_once()
        assert any("Attente après terminate impossible" in error for error in result["cleanup_errors"])
        assert any("Attente après kill impossible" in error for error in result["cleanup_errors"])


def test_terminate_system_error_does_not_prevent_kill(environment):
    environment.process.wait.side_effect = [
        OSError("Attente initiale impossible"), subprocess.TimeoutExpired("sumo", 5), 0,
    ]
    environment.process.terminate.side_effect = OSError("Terminate impossible")

    result = sumo_smoke.run_sumo_smoke(FIXTURE)

    environment.process.terminate.assert_called_once()
    environment.process.kill.assert_called_once()
    assert result["status"] == "failed"
    assert result["process_stopped"]
    assert any("Terminate impossible" in error for error in result["cleanup_errors"])


def test_failed_kill_keeps_process_stop_unconfirmed(environment):
    environment.process.wait.side_effect = OSError("Attente impossible")
    environment.process.kill.side_effect = OSError("Kill impossible")
    environment.process.poll.return_value = None
    environment.process.returncode = None

    result = sumo_smoke.run_sumo_smoke(FIXTURE)

    environment.process.terminate.assert_called_once()
    environment.process.kill.assert_called_once()
    assert result["status"] == "failed"
    assert not result["process_stopped"]
    assert result["process_returncode"] is None
    assert any("Kill impossible" in error for error in result["cleanup_errors"])


def test_poll_error_cannot_confirm_process_stop(environment):
    environment.process.poll.side_effect = OSError("État du processus inaccessible")

    result = sumo_smoke.run_sumo_smoke(FIXTURE)

    assert result["status"] == "failed"
    assert not result["process_stopped"]
    assert result["process_returncode"] is None
    assert any("État du processus inaccessible" in error for error in result["cleanup_errors"])


def test_nonzero_process_exit_fails(environment):
    environment.process.returncode = 1
    environment.process.poll.return_value = 1
    result = sumo_smoke.run_sumo_smoke(FIXTURE)
    assert result["status"] == "failed"
    assert result["process_returncode"] == 1


@pytest.mark.parametrize("dependency", ["sumo", "traci"])
def test_missing_dependency_is_readable_failure(environment, monkeypatch, dependency):
    if dependency == "sumo":
        monkeypatch.setattr(sumo_smoke.shutil, "which", lambda name: None)
    else:
        def unavailable(name):
            raise ImportError("Non installé")
        monkeypatch.setattr(sumo_smoke.importlib, "import_module", unavailable)
    result = sumo_smoke.run_sumo_smoke(FIXTURE)
    assert result["status"] == "failed"
    assert "introuvable" in result["reason"]
    environment.popen.assert_not_called()


@pytest.mark.parametrize("horizon", [0, -1, float("nan"), float("inf")])
def test_invalid_horizon_refused_before_start(environment, horizon):
    with pytest.raises(sumo_smoke.SmokeInputError, match="horizon"):
        sumo_smoke.run_sumo_smoke(FIXTURE, horizon_s=horizon)
    environment.popen.assert_not_called()


def test_missing_fixture_refused_before_start(environment, tmp_path):
    with pytest.raises(sumo_smoke.SmokeInputError, match="Fixture invalide"):
        sumo_smoke.run_sumo_smoke(tmp_path)
    environment.popen.assert_not_called()


@pytest.mark.parametrize("arguments, code, message", [
    (["--horizon", "0"], 2, "Contrôle refusé"),
    (["--sumo-binary", "sumo-definitely-not-installed"], 1, "introuvable"),
])
def test_cli_failure_codes(arguments, code, message):
    completed = subprocess.run([sys.executable, str(ROOT / "scripts/check_sumo.py"), *arguments],
                               capture_output=True, text=True, encoding="utf-8", timeout=15)
    assert completed.returncode == code
    assert message in completed.stdout + completed.stderr


def test_real_sumo_traci_integration():
    """Vérifie le trajet réel si SUMO et TraCI sont disponibles."""
    if shutil.which("sumo") is None or importlib.util.find_spec("traci") is None:
        pytest.skip("Intégration réelle : SUMO dans PATH et TraCI importable sont nécessaires.")
    completed = subprocess.run([sys.executable, str(ROOT / "scripts/check_sumo.py")],
                               capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout[completed.stdout.index("{"):])
    assert result["status"] == "passed"
    assert result["sumo_version"] and result["connected"]
    assert 0 < result["steps"] <= result["max_steps"]
    assert 0 < result["simulation_time_s"] <= result["horizon_s"]
    assert len(result["departures"]) == len(result["arrivals"]) == 1
    assert result["departures"][0]["vehicle_id"] == VEHICLE
    assert result["arrivals"][0]["vehicle_id"] == VEHICLE
    assert result["position_reads"] > 0
    assert result["first_observation"]["route"] == ROUTE
    assert result["last_observation"]["road_id"] == result["destination_confirmed"] == "destination"
    assert result["last_observation"]["route_index"] == 1
    assert result["last_observation"]["time_s"] < result["arrivals"][0]["time_s"]
    assert result["teleport_starts"] == result["teleport_ends"] == []
    assert result["time_to_teleport_s"] == -1
    assert result["assistance_commands"] == []
    assert result["connection_closed"] and result["process_stopped"]
    assert not result["forced_process_stop"]
    assert result["process_returncode"] == 0
    assert result["cleanup_errors"] == []
