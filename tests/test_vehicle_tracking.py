"""Régressions du suivi générique : présence, missions et arrivées."""

from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from cav_recovery.simulation import traffic_demand as demand
from cav_recovery.simulation import vehicle_tracking as run
from cav_recovery.simulation import road_network as network


def mission(item="car", scheduled=0):
    return demand.Mission(item, "LOW", "entry", "exit",
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
def test_tripinfo_corroborates_destination(tmp_path, attributes):
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


def test_disappearance_preserves_observed_and_validated_diagnostics():
    ledger = run.TrafficLedger([mission()])
    connection = Connection([normal_frames()[0], {}])
    observe(ledger, connection)
    with pytest.raises(RuntimeError, match="Disparition"):
        observe(ledger, connection)
    counts = ledger.snapshot()
    assert counts["departed"] == 1 and counts["arrived"] == 0
    assert counts["observed_active"] == counts["active"] == counts["pending"] == 0
    assert counts["validated_active"] == counts["missing_without_arrival"] == 1
    assert ledger.failure_observation["active_ids"] == []
    assert ledger.last_validated_state["active_ids"] == ["car"]
    assert ledger.vehicle_records({})[0]["status"] == "missing_without_arrival"


def test_invalid_route_at_departure_keeps_actual_departure():
    ledger = run.TrafficLedger([mission()])
    connection = Connection([{"departed": ["car"], "active": {"car": {"route": ("start", "wrong")}}}])
    with pytest.raises(RuntimeError, match="Route"):
        observe(ledger, connection)
    counts = ledger.snapshot()
    assert counts["departed"] == counts["observed_active"] == 1
    assert counts["validated_active"] == counts["delayed_not_inserted"] == 0
    assert ledger.vehicle_records({})[0]["status"] == "active_unvalidated"
    assert ledger.failure_observation["vehicles"]["car"]["route"] == ["start", "wrong"]


def test_subscribed_values_match_direct_reads():
    direct = run.TrafficLedger([mission()])
    subscribed = run.TrafficLedger([mission()])
    first = Connection(normal_frames())
    second = Connection(normal_frames())
    for _ in range(3):
        first.simulationStep()
        second.simulationStep()
        active = second.frame.get("active", {})
        readings = {item: {"route": data.get("route", ("start", "end")),
                           "position": data.get("position", (0, 0)),
                           "road_id": data.get("road", "start"),
                           "route_index": data.get("index", 0), "shape": "passenger/sedan"}
                    for item, data in active.items()}
        assert direct.observe(first) == subscribed.observe(second, readings)
        assert direct.last == subscribed.last


def test_subscription_error_does_not_erase_departure():
    ledger = run.TrafficLedger([mission()])
    connection = Connection([normal_frames()[0]])
    connection.simulationStep()
    def failed_readings(connection):
        raise OSError("lecture groupée")
    with pytest.raises(OSError, match="groupée"):
        ledger.observe(connection, failed_readings)
    assert ledger.departures == {"car": 0}
    assert ledger.observed_active == {"car"} and ledger.validated_active == set()
    assert ledger.vehicle_records({})[0]["status"] == "active_unvalidated"
    assert ledger.failure_observation["departed_ids"] == ["car"]


def test_empty_poisson_population_has_no_insertion_delay():
    ledger = run.TrafficLedger([])
    counts = ledger.snapshot()
    assert counts["scheduled"] == counts["pending"] == counts["departed"] == 0
    assert counts["max_insertion_delay_s"] is None
    assert counts["mean_insertion_delay_s"] is None
