"""Contraintes collectives contemporaines, notamment sans cycle de dépendances."""

import copy

import pytest

from cav_recovery.communication.reports import make_report, ReportError
from cav_recovery.crdg import LocalCRDG
from cav_recovery.d01.observer import EgoObserver
from tests.support.d01_fixtures import packet


@pytest.fixture(scope="module")
def proven_diagnostics():
    results = {}
    for ego, blocker, position in (
        ("V0", "V1", 0.0),
        ("V1", "V2", 5.0),
        ("V3", "V4", 20.0),
    ):
        observer = EgoObserver()
        for tick in range(151):
            row = packet(
                round(0.4 + 0.2 * tick, 7),
                ego_x=position,
                other_x=position + 5.0,
                sender=blocker,
            )
            row["ego_id"] = ego
            row["mission"]["movement_id"] = ego + ":forward"
            result = observer.update(row)
        assert result["status"] == "CONFIRMED_OBSTRUCTION"
        results[ego] = result
    return results


def knowledge(*diagnostics):
    received = {}
    for diagnostic in diagnostics:
        sender = diagnostic["ego_id"]
        at_s = diagnostic["emitted_at_s"]
        received[sender] = dict(
            received_at_s=at_s,
            message=dict(
                protocol_version=1,
                sender_id=sender,
                message_id=sender + ":report:1",
                sequence_number=1,
                observed_at_s=at_s,
                sent_at_s=at_s,
                valid_until_s=at_s + 1.2,
                position_xy_m=diagnostic["own_state"]["position_xy_m"],
                report=make_report(diagnostic),
            ),
        )
    return dict(received=received)


def test_acyclic_chain_is_a_collective_with_two_affected_requesters(proven_diagnostics):
    graph = LocalCRDG("V0")
    result = graph.update(
        proven_diagnostics["V0"], knowledge(proven_diagnostics["V1"]), 30.4
    )
    assert result["collective_status"] == "CONFIRMED_COLLECTIVE"
    assert result["groups"][0]["affected_requesters"] == ["V0", "V1"]
    assert not result["groups"][0]["cycle_present"]
    assert result["indirect_dependencies"][0]["path"] == ["V0", "V1", "V2"]


def test_two_unrelated_confirmations_are_not_a_collective(proven_diagnostics):
    result = LocalCRDG("V0").update(
        proven_diagnostics["V0"], knowledge(proven_diagnostics["V3"]), 30.4
    )
    assert result["simple_count_baseline"]
    assert not result["groups"]
    assert result["collective_status"] == "NOT_ESTABLISHED"


def test_historical_report_never_becomes_current_occupancy_by_ttl(proven_diagnostics):
    own = copy.deepcopy(proven_diagnostics["V0"])
    own.update(
        evidence_through_s=30.6,
        emitted_at_s=30.6,
        status="SUSPECTED",
        current_occupants=[],
        anonymous_obstacles=[],
    )
    own["coverage"]["persistence_window"] = None
    graph = LocalCRDG("V0")
    result = graph.update(own, knowledge(proven_diagnostics["V1"]), 30.6)
    assert not result["groups"]
    assert all(edge["state"] == "HISTORICAL" for edge in result["edges"])
    assert not result["indirect_dependencies"]


def test_anonymous_returns_do_not_create_identified_vehicle_dependencies():
    row = packet()
    row["v2v"] = None
    diagnostic = EgoObserver().update(row)
    graph = LocalCRDG("V0").update(diagnostic, None, 0.4)
    assert graph["anonymous_obstacles"]
    assert graph["edges"] == [] and graph["groups"] == []


def test_graph_uses_only_received_reports_and_never_reads_reference_files(
    proven_diagnostics, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("reference access")

    monkeypatch.setattr("builtins.open", forbidden)
    first = LocalCRDG("V0").update(proven_diagnostics["V0"], None, 30.4)
    assert not first["groups"]
    second = LocalCRDG("V0").update(
        proven_diagnostics["V0"], knowledge(proven_diagnostics["V1"]), 30.4
    )
    assert second["groups"]
    with pytest.raises(ReportError):
        LocalCRDG("another").update(proven_diagnostics["V0"], None, 30.4)
