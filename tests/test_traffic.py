"""Contrôles synthétiques des demandes, missions et bilans de simulation."""

from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import pytest

from cav_recovery.simulation import traffic_demand as demand
from cav_recovery.simulation import traffic_run as run
from cav_recovery.simulation import traffic_scenario as scenario
from cav_recovery.simulation import road_network as network
from cav_recovery.simulation.sumo_process import close_sumo

FIXTURE = Path(__file__).parent / "fixtures/traffic/demand.json"


@pytest.fixture
def contract():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def mission(item="car", scheduled=0):
    return demand.Mission(item, "LOW", demand.ENTRY_GATES[0], demand.EXIT_GATES[0],
                          "straight", ("start", "end"), "end", scheduled)


class Connection:
    """Double limité aux lectures nécessaires, sans commande de déplacement."""

    def __init__(self, frames):
        self.frames = iter(frames)
        self.frame = {}
        self.time_s = 0
        self.closed = False
        self.simulation = SimpleNamespace(
            getTime=lambda: self.time_s,
            getDepartedIDList=lambda: self.frame.get("departed", []),
            getArrivedIDList=lambda: self.frame.get("arrived", []),
            getStartingTeleportIDList=lambda: self.frame.get("teleport_starts", []),
            getEndingTeleportIDList=lambda: self.frame.get("teleport_ends", []),
            getCollidingVehiclesIDList=lambda: self.frame.get("collisions", []),
            getDeltaT=lambda: 0.5,
        )
        self.vehicle = SimpleNamespace(
            getIDList=lambda: list(self.frame.get("active", {})),
            getDeparture=lambda item: self.frame["active"][item].get("departure", 0),
            getRoute=lambda item: self.frame["active"][item].get("route", ("start", "end")),
            getRoadID=lambda item: self.frame["active"][item].get("road", "start"),
            getRouteIndex=lambda item: self.frame["active"][item].get("index", 0),
            getPosition=lambda item: self.frame["active"][item].get("position", (0, 0)),
            getShapeClass=lambda item: "passenger/sedan",
        )
        self.route = SimpleNamespace(getEdges=lambda item: ("start", "end"))
        logic = SimpleNamespace(programID="0", type=0,
                                phases=[SimpleNamespace(duration=d, state=s) for d, s in network.CENTER_PHASES])
        self.trafficlight = SimpleNamespace(getAllProgramLogics=lambda item: [logic],
                                           getProgram=lambda item: "0",
                                           getRedYellowGreenState=lambda item: "GGrrr")

    def simulationStep(self):
        self.time_s += 0.5
        self.frame = next(self.frames)

    def getVersion(self):
        return (22, "SUMO 1.27.1")

    def close(self, wait):
        self.closed = True


def normal_frames():
    return [{"departed": ["car"], "active": {"car": {}}},
            {"active": {"car": {"road": "end", "index": 1}}},
            {"arrived": ["car"]}]


def observe(ledger, connection):
    connection.simulationStep()
    return ledger.observe(connection)


@pytest.mark.parametrize("name,total", [("LOW", 5), ("MID", 7), ("HIGH", 8)])
def test_plans_conserve_entries_and_movements(contract, name, total):
    plan = demand.demand_plans(contract)[name]
    assert sum(plan["entries"].values()) == total
    assert sum(sum(counts.values()) for counts in plan["allocation"].values()) == total
    for gate in demand.ENTRY_GATES:
        assert sum(plan["allocation"][gate].values()) == plan["entries"][gate]
    assert plan["censored_exit"] == (2 if name == "HIGH" else 0)


def test_high_allocation_preserves_observed_distribution():
    observed = {"east": 131, "west": 8}
    assert demand.allocate_counts(141, observed) == {"east": 133, "west": 8}
    assert observed == {"east": 131, "west": 8}


@pytest.mark.parametrize("total,counts", [(1, {"a": 0}), (-1, {"a": 1}), (2, {"a": -1}), (2.0, {"a": 1})])
def test_invalid_allocation(total, counts):
    with pytest.raises(demand.TrafficInputError):
        demand.allocate_counts(total, counts)


def test_remainder_tie_and_zero():
    assert demand.allocate_counts(3, {"b": 1, "a": 1}) == {"a": 2, "b": 1}
    assert demand.allocate_counts(0, {"b": 0, "a": 0}) == {"a": 0, "b": 0}


def test_departure_grid_regular_distinct_and_half_open():
    times = demand.departure_times(7, 10)
    assert times == [0, 1, 2.5, 4, 5.5, 7, 8.5]
    assert len(set(times)) == 7
    assert all(t % 0.5 == 0 and 0 <= t < 10 for t in times)
    assert demand.departure_times(0, 10) == []


@pytest.mark.parametrize("count,duration", [(21, 10), (2, 0), (2, 1.2), (-1, 10), (2, float("nan")), (1.5, 2)])
def test_invalid_departure_grid(count, duration):
    with pytest.raises(demand.TrafficInputError):
        demand.departure_times(count, duration)


def test_interleaving_is_deterministic_and_not_destination_blocks():
    assert demand.interleave_movements({"a": 2, "b": 4}) == ["b", "a", "b", "b", "a", "b"]
    assert demand.interleave_movements({"b": 4, "a": 2}) == demand.interleave_movements({"a": 2, "b": 4})


def test_missions_deterministic_assign_routes_and_destination(contract):
    plan = demand.demand_plans(contract)["HIGH"]
    first = demand.build_missions("HIGH", plan, network.ROUTES)
    assert first == demand.build_missions("HIGH", deepcopy(plan), network.ROUTES)
    assert len({m.vehicle_id for m in first}) == 8
    assert all(m.destination == m.route[-1] and m.route == tuple(network.ROUTES[m.route_id]) for m in first)
    for gate in demand.ENTRY_GATES:
        selected = [m for m in first if m.entry_gate == gate]
        assert len({m.scheduled_s for m in selected}) == len(selected)
        assert dict(Counter(m.exit_gate for m in selected)) == plan["allocation"][gate]


@pytest.mark.parametrize("change", ["denominator", "censoring", "duplicate", "exit"])
def test_inconsistent_profiles_refused(contract, change):
    row = contract["passenger_cav_contract"]["regimes"][0]
    if change == "denominator":
        row["passenger_movement_denominators"][demand.ENTRY_GATES[0]] += 1
    elif change == "censoring":
        row["censored_exit"] = 1
    elif change == "duplicate":
        contract["passenger_cav_contract"]["regimes"].append(deepcopy(row))
    else:
        row["passenger_movement_counts"][demand.ENTRY_GATES[0]].pop(demand.EXIT_GATES[0])
    with pytest.raises(demand.TrafficInputError):
        demand.demand_plans(contract)


def test_input_hash_and_missing_file(tmp_path):
    path = tmp_path / "input.json"
    with pytest.raises(demand.TrafficInputError, match="absent"):
        demand.verify_identity(path, (3, "bad"))
    path.write_bytes(b"abc")
    identity = (3, demand.file_hash(path))
    assert demand.verify_identity(path, identity)["sha256"] == identity[1]
    path.write_bytes(b"abd")
    with pytest.raises(demand.TrafficInputError, match="SHA"):
        demand.verify_identity(path, identity)


def test_canonical_contract_cannot_be_replaced_by_synthetic_fixture():
    with pytest.raises(demand.TrafficInputError, match="SHA"):
        demand.read_contract(FIXTURE)


def test_vehicle_and_gui_settings_do_not_change_dynamics(tmp_path, contract):
    plan = demand.demand_plans(contract)["LOW"]
    missions = demand.build_missions("LOW", plan, network.ROUTES)
    scenario.write_traffic_files(tmp_path, missions, network.ROUTES, False)
    scenario.write_view(tmp_path / "view.xml", {"x": "0", "y": "0"})
    root = ET.parse(tmp_path / "traffic.rou.xml").getroot()
    assert root.find("vType").attrib == scenario.VEHICLE_TYPE
    assert root.find("vType").get("guiShape") == "passenger/sedan"
    config = ET.parse(tmp_path / "simulation.sumocfg").getroot()
    assert config.find("time/step-length").get("value") == "0.5"
    assert config.find("processing/time-to-teleport").get("value") == "-1"
    assert config.find("processing/max-depart-delay").get("value") == "-1"
    assert ET.parse(tmp_path / "view.xml").find("scheme/vehicles").get("vehicleQuality") == "2"
    assert len(root.findall("vehicle")) == 5


def test_normal_arrival_and_conservation():
    ledger = run.TrafficLedger([mission()])
    connection = Connection(normal_frames())
    for _ in range(3):
        counts = observe(ledger, connection)
        assert ledger.observed_active == ledger.validated_active
        assert counts["active"] == counts["observed_active"] == counts["validated_active"]
        assert counts["missing_without_arrival"] == counts["active_unvalidated"] == 0
        assert ledger.failure_observation is None
        assert counts["scheduled"] == counts["pending"] + counts["active"] + counts["arrived"]
        assert counts["departed"] == counts["active"] + counts["arrived"]
    assert counts["arrived"] == 1 and counts["pending"] == counts["active"] == 0


def test_future_and_delayed_insertion_are_distinct():
    ledger = run.TrafficLedger([mission(), mission("later", 2)])
    connection = Connection([{}, {"departed": ["car"], "active": {"car": {"departure": 0.5}}}])
    counts = observe(ledger, connection)
    assert counts["future"] == counts["delayed_not_inserted"] == 1
    counts = observe(ledger, connection)
    assert counts["departed"] == 1 and counts["future"] == 1 and counts["delayed_not_inserted"] == 0
    assert counts["max_insertion_delay_s"] == 0.5


@pytest.mark.parametrize("bad_frame,message", [
    ({"teleport_starts": ["car"]}, "Téléportation"),
    ({"teleport_ends": ["car"]}, "Téléportation"),
    ({"collisions": ["car"]}, "Collision"),
    ({"active": {"unknown": {}}}, "inconnu"),
    ({"departed": ["car"], "active": {"car": {}}}, "répété"),
    ({"active": {"car": {"route": ("start", "wrong")}}}, "Route"),
    ({"active": {"car": {"position": (float("nan"), 0)}}}, "Position"),
    ({"arrived": ["car"]}, "destination"),
    ({}, "Disparition"),
])
def test_abnormal_events_never_validate_arrival(bad_frame, message):
    ledger = run.TrafficLedger([mission()])
    connection = Connection([normal_frames()[0], bad_frame])
    observe(ledger, connection)
    with pytest.raises(RuntimeError, match=message):
        observe(ledger, connection)
    assert not ledger.arrivals


def write_trip(path, **attributes):
    root = ET.Element("tripinfos")
    ET.SubElement(root, "tripinfo", {"id": "car", "depart": "0", "arrival": "1", "arrivalLane": "end_0", **attributes})
    network.write_xml(path, root)


@pytest.mark.parametrize("attributes", [{}, {"arrivalLane": "wrong_0"}, {"vaporized": "true"}, {"arrival": "nan"}])
def test_tripinfo_corrobates_destination(tmp_path, attributes):
    ledger = run.TrafficLedger([mission()])
    connection = Connection(normal_frames())
    for _ in range(3):
        observe(ledger, connection)
    path = tmp_path / "tripinfo.xml"
    write_trip(path, **attributes)
    if attributes:
        with pytest.raises(RuntimeError):
            run.verify_trips(path, ledger)
    else:
        assert run.verify_trips(path, ledger)["car"]["arrival_s"] == 1


def fake_run(monkeypatch, tmp_path, frames, gui=False, cleanup_error=False):
    connection = Connection(frames)
    monkeypatch.setattr(run, "read_scenario", lambda path: {"regimes": {"LOW": {
        "missions": demand.mission_records([mission()]), "plan": {"injection_s": 0.5}}}})
    monkeypatch.setattr(run, "check_environment", lambda *args: {"binary": args[0]})
    monkeypatch.setattr(run, "file_hash", lambda path: "synthetic")
    monkeypatch.setattr(run, "code_provenance", lambda: {"sha256": "synthetic"})
    monkeypatch.setattr(run.shutil, "which", lambda name: name)
    process = Mock()
    process.poll.return_value = 0
    process.wait.side_effect = [OSError("attente") , None] if cleanup_error else None
    launch = Mock(return_value=process)
    monkeypatch.setattr(run.subprocess, "Popen", launch)
    monkeypatch.setattr(run.importlib, "import_module", lambda name: SimpleNamespace(connect=lambda **kwargs: connection))
    monkeypatch.setattr(run, "verify_trips", lambda path, ledger: {i: {"arrival_s": 1} for i in ledger.arrivals})
    result = run.run_traffic(tmp_path, "LOW", tmp_path / "result", gui=gui, gui_delay_ms=7, drain_horizon_s=1)
    return result, connection, process, launch.call_args.args[0]


def test_runner_normal_success_and_files(monkeypatch, tmp_path):
    close = Mock(wraps=close_sumo)
    monkeypatch.setattr(run, "close_sumo", close)
    result, connection, process, command = fake_run(monkeypatch, tmp_path, normal_frames())
    close.assert_called_once_with(connection, process, result)
    assert result["status"] == "passed"
    assert result["counts"]["scheduled"] == result["counts"]["arrived"] == 1
    assert connection.closed and result["process_stopped"] and result["connection_closed"]
    assert process.wait.called and command[0] == "sumo"
    assert {"vehicles.csv", "timeline.csv", "summary.json", "sumo.log"}.issubset(p.name for p in (tmp_path / "result").iterdir())


@pytest.mark.parametrize("filename,included", [("sumo_process.py", True), ("sumo_smoke.py", False)])
def test_code_digest_tracks_shared_cleanup_not_smoke(monkeypatch, filename, included):
    original = Path.read_bytes
    before = run.code_provenance()["sha256"]

    def read_bytes(path):
        contents = original(path)
        return contents + b"\n" if path.name == filename else contents

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    assert (run.code_provenance()["sha256"] != before) is included


def test_failure_summary_missing_without_arrival(monkeypatch, tmp_path):
    result, connection, _, _ = fake_run(monkeypatch, tmp_path, [normal_frames()[0], {}])
    assert result["status"] == "failed" and "Disparition" in result["reason"]
    counts = result["counts"]
    assert counts["departed"] == 1 and counts["arrived"] == 0
    assert counts["active"] == counts["pending"] == counts["delayed_not_inserted"] == 0
    assert counts["observed_active"] == 0 and counts["validated_active"] == 1
    assert counts["missing_without_arrival"] == 1
    row = result["vehicles"][0]
    assert row["status"] == "missing_without_arrival" and row["actual_departure_s"] == 0
    assert not row["observed_active"] and row["validated_active"]
    assert result["failure_observation"]["active_ids"] == []
    assert result["failure_observation"]["time_s"] == 1
    assert result["last_validated_state"]["active_ids"] == ["car"]
    assert result["last_validated_state"]["time_s"] == 0.5
    assert connection.closed and result["process_stopped"]
    for group in ("by_entry", "by_movement"):
        assert result[group][0]["departed"] == result[group][0]["missing_without_arrival"] == 1
        assert result[group][0]["active"] == result[group][0]["delayed_not_inserted"] == 0
    saved = json.loads((tmp_path / "result/summary.json").read_text(encoding="utf-8"))
    assert saved["failure_observation"] == result["failure_observation"]
    assert saved["vehicles"][0]["status"] == "missing_without_arrival"


def test_failure_summary_invalid_route_at_departure(monkeypatch, tmp_path):
    frame = {"departed": ["car"], "active": {"car": {"route": ("start", "wrong")}}}
    result, connection, _, _ = fake_run(monkeypatch, tmp_path, [frame])
    assert result["status"] == "failed" and "Route" in result["reason"]
    counts = result["counts"]
    assert counts["departed"] == 1 and counts["arrived"] == counts["pending"] == 0
    row = result["vehicles"][0]
    assert row["status"] != "delayed_not_inserted"
    assert row["status"] == "active_unvalidated" and row["actual_departure_s"] == 0
    assert row["observed_active"] and not row["validated_active"]
    assert counts["active"] == counts["observed_active"] == 1 and counts["validated_active"] == 0
    assert counts["delayed_not_inserted"] == counts["missing_without_arrival"] == 0
    observation = result["failure_observation"]
    assert observation["active_ids"] == observation["departed_ids"] == ["car"]
    assert observation["vehicles"]["car"]["route"] == ["start", "wrong"]
    assert result["last_validated_state"]["active_ids"] == []
    assert result["last_validated_state"]["time_s"] == 0
    assert connection.closed and result["process_stopped"]
    for group in ("by_entry", "by_movement"):
        assert result[group][0]["active"] == result[group][0]["active_unvalidated"] == 1
        assert result[group][0]["delayed_not_inserted"] == 0
    saved = json.loads((tmp_path / "result/summary.json").read_text(encoding="utf-8"))
    assert saved["vehicles"][0]["status"] == "active_unvalidated"
    assert saved["failure_observation"] == observation


def test_runner_horizon_preserves_remaining_and_closes(monkeypatch, tmp_path):
    result, connection, _, _ = fake_run(monkeypatch, tmp_path, [{}, {}, {}])
    assert result["status"] == "failed" and "Horizon" in result["reason"]
    assert result["remaining_ids"] == ["car"] and result["counts"]["delayed_not_inserted"] == 1
    assert connection.closed


def test_runner_closes_after_error(monkeypatch, tmp_path):
    result, connection, _, _ = fake_run(monkeypatch, tmp_path, [{"teleport_starts": ["car"]}])
    assert result["status"] == "failed" and connection.closed
    assert result["counts"]["teleport_starts"] == 1 and result["counts"]["arrived"] == 0


def test_cleanup_fallback_stays_failure(monkeypatch, tmp_path):
    result, connection, process, _ = fake_run(monkeypatch, tmp_path, normal_frames(), cleanup_error=True)
    assert result["status"] == "failed" and result["cleanup_errors"]
    assert process.terminate.called and connection.closed


def test_gui_changes_only_display_command(monkeypatch, tmp_path):
    result, _, _, command = fake_run(monkeypatch, tmp_path, normal_frames(), gui=True)
    assert command[0] == "sumo-gui"
    assert command[command.index("--delay") + 1] == "7"
    assert command[command.index("--start") + 1] == "true"
    assert result["step_s"] == 0.5 and result["seed"] == 0 and result["status"] == "passed"


@pytest.mark.parametrize("change", ["step", "route", "phases"])
def test_loaded_scenario_mismatch_refused(change):
    connection = Connection([])
    if change == "step":
        connection.simulation.getDeltaT = lambda: 1
    elif change == "route":
        connection.route.getEdges = lambda item: ("start", "other")
    else:
        connection.trafficlight.getProgram = lambda item: "changed"
    with pytest.raises(RuntimeError):
        run.verify_loaded_scenario(connection, [mission()])


def test_prepare_failure_is_not_partially_published(monkeypatch, tmp_path, contract):
    monkeypatch.setattr(scenario, "read_contract", lambda path: contract)
    monkeypatch.setattr(scenario, "check_environment", lambda *args: {})
    monkeypatch.setattr(scenario, "convert_network", Mock(side_effect=RuntimeError("conversion")))
    output = tmp_path / "scenario"
    with pytest.raises(RuntimeError, match="conversion"):
        scenario.prepare_traffic("unused", "unused", output)
    assert not output.exists()
    assert not list(tmp_path.glob(".traffic-*"))


def test_output_refuses_nonempty_directory(tmp_path):
    (tmp_path / "keep.txt").write_text("à conserver", encoding="utf-8")
    with pytest.raises(demand.TrafficInputError):
        scenario.new_output_directory(tmp_path)
    assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "à conserver"


def test_scenario_hash_rejects_changed_files(tmp_path):
    manifest = {"schema_version": "traffic-scenario-1", "status": "prepared", "files_sha256": {}}
    names = ["network.net.xml", "view.xml"] + [f"{r}/{f}" for r in ("LOW", "MID", "HIGH")
                                               for f in ("traffic.rou.xml", "simulation.sumocfg")]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("synthetic", encoding="utf-8")
        manifest["files_sha256"][name] = demand.file_hash(path)
    scenario.write_json(tmp_path / "scenario.json", manifest)
    (tmp_path / "view.xml").write_text("changed", encoding="utf-8")
    with pytest.raises(demand.TrafficInputError, match="modifié"):
        scenario.read_scenario(tmp_path)


def load_cli(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["prepare_traffic", "run_traffic"])
def test_cli_rejects_absent_input(monkeypatch, tmp_path, name):
    cli = load_cli(name)
    args = (["--osm", "absent", "--contract", "absent"] if name == "prepare_traffic"
            else ["--scenario-dir", "absent", "--regime", "LOW"])
    monkeypatch.setattr("sys.argv", [name, *args, "--output-dir", str(tmp_path / "result")])
    assert cli.main() == 2


@pytest.mark.parametrize("name", ["prepare_traffic", "run_traffic"])
def test_cli_technical_error(monkeypatch, tmp_path, name):
    cli = load_cli(name)
    args = (["--osm", "synthetic", "--contract", "synthetic"] if name == "prepare_traffic"
            else ["--scenario-dir", "synthetic", "--regime", "LOW"])
    monkeypatch.setattr("sys.argv", [name, *args, "--output-dir", str(tmp_path / "result")])
    monkeypatch.setattr(cli, name, Mock(side_effect=OSError("écriture")))
    assert cli.main() == 1


def synthetic_network(path):
    """Décrit un carrefour fictif avec les noms utilisés par le mapping."""
    root = ET.Element("net")
    ET.SubElement(root, "location", netOffset="0,0", projParameter="!", convBoundary="0,0,100,100")
    ET.SubElement(root, "junction", id=network.CENTER_NODE, type="traffic_light", x="50", y="50")
    tls = ET.SubElement(root, "tlLogic", id=network.CENTER_NODE, type="static", programID="0", offset="0")
    for duration, state in network.CENTER_PHASES:
        ET.SubElement(tls, "phase", duration=str(duration), state=state)
    endpoints = {
        "23183369#1": ("a", "2725672310"), "23183369#2": ("2725672310", "b"),
        "23183369#3": ("b", network.CENTER_NODE), "23183369#4": (network.CENTER_NODE, "c"),
        "23183369#5": ("c", "d"), "284241336#1": ("e", "f"),
        "284241336#2": ("f", network.CENTER_NODE), "284241336#3": (network.CENTER_NODE, "g"),
    }
    for name, (start, end) in endpoints.items():
        edge = ET.SubElement(root, "edge", id=name, **{"from": start, "to": end})
        for index in range(1 if name.startswith("231") else 2):
            lane = ET.SubElement(edge, "lane", id=f"{name}_{index}", length="25", speed="10")
            ET.SubElement(lane, "param", key="origId", value=name.split("#")[0])
    for a in ("23183369#3", "284241336#2"):
        for b in ("23183369#4", "284241336#3"):
            ET.SubElement(root, "connection", **{"from": a, "to": b, "tl": network.CENTER_NODE})
    network.write_xml(path, root)
    return root


class NetworkDouble:
    def __init__(self, root):
        self.edges = {e.get("id"): e for e in root.findall("edge")}
        self.forbidden = False
        self.detour = False

    def getEdge(self, name):
        row = self.edges[name]
        edge = ComparableEdge(getID=lambda: name, getFunction=lambda: "",
                              allows=lambda vehicle: not self.forbidden)
        # Les mêmes objets nœuds sont comparés par identité dans sumolib.
        edge.getFromNode = lambda: self.node(row.get("from"))
        edge.getToNode = lambda: self.node(row.get("to"))
        edge.getAllowedOutgoing = lambda vehicle: [self.getEdge(k) for k, e in self.edges.items()
                                                   if e.get("from") == row.get("to")]
        return edge

    def node(self, item):
        if not hasattr(self, "nodes"):
            self.nodes = {}
        return self.nodes.setdefault(item, SimpleNamespace(getID=lambda: item))

    def getShortestPath(self, start, end, **kwargs):
        expected = next(route for route in network.ROUTES.values() if route[0] == start.getID() and route[-1] == end.getID())
        if self.detour:
            return (None, 0)
        route = [self.getEdge(e) for e in expected]
        if expected[0].startswith("231"):
            route.insert(1, SimpleNamespace(getID=lambda: ":2725672310_0", getFunction=lambda: "internal",
                                            allows=lambda vehicle: True))
        return (route, 1)


class ComparableEdge(SimpleNamespace):
    def __eq__(self, other):
        return self.getID() == other.getID()


@pytest.mark.parametrize("change", [None, "cycle", "controller", "detour", "permission", "origin", "lanes"])
def test_network_mapping_routes_and_provenance(monkeypatch, tmp_path, change):
    import sumolib
    path = tmp_path / "network.net.xml"
    root = synthetic_network(path)
    double = NetworkDouble(root)
    if change == "cycle":
        root.find("tlLogic/phase").set("duration", "38")
    elif change == "controller":
        root.find("junction").set("type", "priority")
    elif change == "detour":
        double.detour = True
    elif change == "permission":
        double.forbidden = True
    elif change == "origin":
        root.find("edge/lane/param").set("value", "unknown")
    elif change == "lanes":
        root.find("edge").append(deepcopy(root.find("edge/lane")))
    network.write_xml(path, root)
    monkeypatch.setattr(sumolib.net, "readNet", lambda *args, **kwargs: double)
    if change:
        with pytest.raises(demand.TrafficInputError):
            network.inspect_network(path, FIXTURE.with_name("road.osm"))
    else:
        result = network.inspect_network(path, FIXTURE.with_name("road.osm"))
        assert result["gate_mapping"]["W23183369_IN"] == "23183369#1"
        assert result["gate_mapping"]["entry_connector"] == ":2725672310_0"
        assert result["routes"] == {name: list(edges) for name, edges in network.ROUTES.items()}
        assert result["traffic_lights"][0]["cycle_s"] == 90
        assert all(e["speed_origin"] == "typemap/règle SUMO" for e in result["edges"])


def test_conversion_parameters_and_error(monkeypatch, tmp_path):
    monkeypatch.setattr(network, "verify_identity", lambda *args: {"sha256": "synthetic"})
    monkeypatch.setattr(network.shutil, "which", lambda tool: tool)
    root = synthetic_network(tmp_path / "network.net.xml")
    execute = Mock(return_value=SimpleNamespace(returncode=0, stderr=""))
    monkeypatch.setattr(network.subprocess, "run", execute)
    network.convert_network(FIXTURE.with_name("road.osm"), tmp_path)
    command = execute.call_args.args[0]
    for option, value in (("--tls.default-type", "static"), ("--tls.cycle.time", "90"),
                           ("--tls.join", "false"), ("--junctions.join", "false"),
                           ("--output.original-names", "true"), ("--osm.annotate-defaults", "true")):
        assert command[command.index(option) + 1] == value
    execute.return_value = SimpleNamespace(returncode=1, stderr="conversion impossible")
    with pytest.raises(RuntimeError, match="conversion impossible"):
        network.convert_network(FIXTURE.with_name("road.osm"), tmp_path)


@pytest.mark.parametrize("change", [None, "mission", "allocation", "xml_route", "xml_step"])
def test_preparation_publishes_complete_directory_and_is_deterministic(monkeypatch, tmp_path, contract, change):
    monkeypatch.setattr(scenario, "read_contract", lambda path: contract)
    monkeypatch.setattr(scenario, "check_environment", lambda *args: {})
    def convert(source, output):
        synthetic_network(output / "network.net.xml")
        return {"source": {"sha256": "synthetic"}}
    monkeypatch.setattr(scenario, "convert_network", convert)
    monkeypatch.setattr(scenario, "inspect_network", lambda *args: {
        "center": {"x": "50", "y": "50"}, "routes": {name: list(edges) for name, edges in network.ROUTES.items()},
        "gate_mapping": {**network.GATE_EDGES, "entry_connector": ":2725672310_0"}})
    monkeypatch.setattr(scenario, "build_scenery", lambda *args: {"polygons": 0})
    results = []
    for name in ("first", "second"):
        output = tmp_path / name
        if name == "first":
            output.mkdir()
        results.append(scenario.prepare_traffic("synthetic", "synthetic", output))
        assert scenario.read_scenario(output)["status"] == "prepared"
        assert all((output / regime / "simulation.sumocfg").exists() for regime in ("LOW", "MID", "HIGH"))
    assert results[0] == results[1]
    assert not list(tmp_path.glob(".traffic-*"))
    if change:
        output = tmp_path / "first"
        manifest = scenario.read_scenario(output)
        if change == "mission":
            manifest["regimes"]["LOW"]["missions"][0]["destination"] = "wrong"
        elif change == "allocation":
            manifest["regimes"]["LOW"]["plan"]["allocation"][demand.ENTRY_GATES[0]][demand.EXIT_GATES[0]] += 1
        else:
            name = "LOW/traffic.rou.xml" if change == "xml_route" else "LOW/simulation.sumocfg"
            path = output / name
            root = ET.parse(path).getroot()
            if change == "xml_route":
                root.find("vehicle").set("route", "wrong")
            else:
                root.find("time/step-length").set("value", "1")
            network.write_xml(path, root)
            manifest["files_sha256"][name] = demand.file_hash(path)
        scenario.write_json(output / "scenario.json", manifest)
        with pytest.raises(demand.TrafficInputError):
            scenario.read_scenario(output)


@pytest.mark.parametrize("version", ["missing", "1.20.0"])
def test_versions_refused(monkeypatch, version):
    monkeypatch.setattr(network.shutil, "which", lambda tool: None if version == "missing" else tool)
    monkeypatch.setattr(network.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=f"SUMO {version}"))
    with pytest.raises(demand.TrafficInputError):
        network.check_environment("sumo")


@pytest.mark.parametrize("field", ["schema", "sector", "gates", "coverage", "counts"])
def test_contract_identity_fields_checked_after_hash(monkeypatch, tmp_path, contract, field):
    contract.update(schema_version="CGR-E03-1", status="complete", empirical_context={
        "sector": {"id": "C3", "osm_node_id": 250691665, "entry_gates": list(demand.ENTRY_GATES),
                   "exit_gates": list(demand.EXIT_GATES)},
        "coverage": {"status": "known", "intervals": [[0.0, 802.8]]}})
    contract["passenger_cav_contract"].update(source_categories=["Car", "Taxi"], population_id="passenger_CAV")
    if field == "schema":
        contract["schema_version"] = "other"
    elif field == "sector":
        contract["empirical_context"]["sector"]["osm_node_id"] = 1
    elif field == "gates":
        contract["empirical_context"]["sector"]["entry_gates"] = ["wrong"]
    elif field == "coverage":
        contract["empirical_context"]["coverage"]["status"] = "unknown"
    path = tmp_path / "contract.json"
    scenario.write_json(path, contract)
    monkeypatch.setattr(demand, "CONTRACT_IDENTITY", (path.stat().st_size, demand.file_hash(path)))
    with pytest.raises(demand.TrafficInputError, match="Effectifs" if field == "counts" else "Identité"):
        demand.read_contract(path)


@pytest.mark.parametrize("existing_empty", [False, True])
def test_generation_error_never_publishes_network_only(monkeypatch, tmp_path, contract, existing_empty):
    monkeypatch.setattr(scenario, "read_contract", lambda path: contract)
    monkeypatch.setattr(scenario, "check_environment", lambda *args: {})
    def convert(source, output):
        synthetic_network(output / "network.net.xml")
        return {}
    monkeypatch.setattr(scenario, "convert_network", convert)
    monkeypatch.setattr(scenario, "inspect_network", lambda *args: {
        "center": {"x": "50", "y": "50"}, "routes": {name: list(edges) for name, edges in network.ROUTES.items()}})
    monkeypatch.setattr(scenario, "build_scenery", lambda *args: {})
    monkeypatch.setattr(scenario, "write_traffic_files", Mock(side_effect=OSError("écriture interrompue")))
    output = tmp_path / "scenario"
    if existing_empty:
        output.mkdir()
    with pytest.raises(OSError, match="interrompue"):
        scenario.prepare_traffic("synthetic", "synthetic", output)
    assert (output.exists() and not list(output.iterdir())) if existing_empty else not output.exists()
    assert not list(tmp_path.glob(".traffic-*"))


def test_missions_never_use_random_generator(monkeypatch, contract):
    import random
    for name in ("random", "randrange", "choice", "shuffle"):
        monkeypatch.setattr(random, name, Mock(side_effect=AssertionError("tirage interdit")))
    assert len(demand.build_missions("MID", demand.demand_plans(contract)["MID"], network.ROUTES)) == 7


@pytest.mark.parametrize("name", ["prepare_traffic", "run_traffic"])
def test_cli_success(monkeypatch, tmp_path, name):
    cli = load_cli(name)
    args = (["--osm", "synthetic", "--contract", "synthetic"] if name == "prepare_traffic"
            else ["--scenario-dir", "synthetic", "--regime", "LOW"])
    monkeypatch.setattr("sys.argv", [name, *args, "--output-dir", str(tmp_path / "result")])
    result = ({"regimes": {"LOW": {"missions": ["car"], "plan": {"injection_s": 1}}}}
              if name == "prepare_traffic" else {"status": "passed", "reason": None,
                "counts": run.TrafficLedger([mission()]).snapshot(), "connection_closed": True, "process_stopped": True})
    monkeypatch.setattr(cli, name, Mock(return_value=result))
    assert cli.main() == 0
