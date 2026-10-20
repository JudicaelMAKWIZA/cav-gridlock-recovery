"""Le canal idéal ne livre que des rapports locaux aux destinataires à portée."""

import pytest

from cav_recovery.communication.reports import (
    DiagnosticExchange,
    make_report,
    validate_report,
    ReportError,
)
from cav_recovery.d01.observer import EgoObserver
from tests.support.d01_fixtures import packet


def diagnostic(at_s, ego="V0"):
    row = packet(at_s, sender="peer-" + ego)
    row["ego_id"] = ego
    return EgoObserver().update(row)


def test_ideal_report_delivery_respects_range_cadence_and_timestamp():
    exchange = DiagnosticExchange()
    for at_s in (0.2, 0.4):
        rows = {
            key: dict(
                position_xy_m=[position, 0.0], report=make_report(diagnostic(at_s, key))
            )
            for key, position in (("V0", 0.0), ("V1", 10.0), ("remote", 100.0))
        }
        exchange.observe(at_s, rows)
    inbox = exchange.knowledge("V0")
    assert set(inbox["received"]) == {"V1"}
    received = inbox["received"]["V1"]
    assert received["received_at_s"] == 0.4
    assert received["message"]["report"]["ego_id"] == "V1"
    assert exchange.knowledge("remote")["received"] == {}


@pytest.mark.parametrize(
    "field", ["hidden_world", "guard_blockers", "reference_labels"]
)
def test_reports_reject_privileged_or_opaque_fields(field):
    report = make_report(diagnostic(0.4))
    report[field] = {}
    with pytest.raises(ReportError):
        validate_report(report, "V0", 0.4)
    report = make_report(diagnostic(0.4))
    report["own_effect"] = {field: 1}
    with pytest.raises(ReportError):
        validate_report(report, "V0", 0.4)


def test_wrong_sender_and_future_evidence_are_rejected():
    report = make_report(diagnostic(0.4))
    with pytest.raises(ReportError):
        validate_report(report, "other", 0.4)
    with pytest.raises(ReportError):
        validate_report(report, "V0", 0.2)
