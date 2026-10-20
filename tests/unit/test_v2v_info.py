"""Le détail V2V représente le paquet transmis, jamais une reconstruction globale."""

from copy import deepcopy

from cav_recovery.experiments.info_graph import message_snapshot
from cav_recovery.experiments.v2v_info import correspondents, message_lines, own_lines


def item(sender="agent0", recipients=None, channel="telemetry", sequence=1):
    return dict(
        channel=channel,
        message_id=f"{sender}:{sequence}",
        sender=sender,
        recipients=recipients or ["agent1", "agent2"],
        sent_at_s=10.0 + sequence / 10,
        received_at_s=10.0 + sequence / 10,
        observed_at_s=10.0,
        payload=dict(
            sender_id=sender, position_xy_m=[1.25, 2.5], signed_speed_mps=3.125
        ),
    )


def test_correspondents_group_receipts_and_preserve_outgoing_broadcast_identity():
    items = [
        item(),
        item(channel="diagnostic"),
        item(sender="agent1", recipients=["agent0"]),
        item(sender="agent1", recipients=["agent0"], sequence=2),
        item(sender="agent8", recipients=["agent9"]),
    ]
    before = deepcopy(items)
    pairs = {("agent0", "agent1"), ("agent0", "agent2"), ("agent1", "agent0")}
    snapshot = message_snapshot(items, pairs, {}, useful=False)
    rows = correspondents(snapshot, "agent0")
    assert items == before and len(rows) == 2
    assert (rows[0]["sent"], rows[0]["received"]) == (2, 2)
    assert (rows[1]["sent"], rows[1]["received"]) == (2, 0)
    assert rows[0]["latest_out"] == rows[1]["latest_out"]
    assert set(rows[0]["latest_out"]) == {"telemetry", "diagnostic"}


def test_summary_selects_latest_sent_packet_for_each_channel():
    items = [item(sequence=1), item(sequence=2)]
    snapshot = message_snapshot(items, {("agent0", "agent1")}, {}, useful=False)
    row = correspondents(snapshot, "agent0")[0]
    key = row["latest_out"]["telemetry"]
    assert snapshot["nodes"][key]["message_id"] == "agent0:2"


def test_readable_packet_preserves_actual_transmitted_values_and_missing_fields():
    node = dict(item(), label="M1")
    before = deepcopy(node)
    text = "\n".join(line for line, _ in message_lines(node, "agent0"))
    assert "1.25; 2.5" in text and "3.125" in text and "ÉMIS" in text
    assert "Destination annoncée" not in text  # Le paquet ne porte pas ce champ.
    assert node == before


def test_raw_packet_retains_unabridged_geometry_and_nested_report():
    node = dict(item(channel="diagnostic"), label="M2")
    node["payload"]["report"] = dict(
        status="UNKNOWN",
        reason="NON_NOMINAL_REVERSE",
        relations=[dict(resource_wkt="POLYGON ((0 0, 1 0, 0 0))")],
    )
    text = "\n".join(line for line, _ in message_lines(node, "agent1"))
    assert "REÇU" in text and "Information insuffisante" in text
    assert "le détecteur s'abstient" in text
    raw = "\n".join(line for line, _ in message_lines(node, "agent1", raw=True))
    assert '"status": "UNKNOWN"' in raw
    assert "POLYGON ((0 0, 1 0, 0 0))" in raw


def test_legacy_packet_with_no_payload_is_explicitly_unavailable():
    node = dict(item(), label="M1")
    del node["payload"]
    text = "\n".join(line for line, _ in message_lines(node, "agent0"))
    assert "aucune donnée n'est reconstruite" in text


def test_own_state_is_only_that_of_the_selected_vehicle():
    frame = dict(
        measurements={
            "agent0": dict(ego=dict(position_xy_m=[1, 2], signed_speed_mps=3)),
            "agent1": dict(ego=dict(position_xy_m=[987, 654], signed_speed_mps=100)),
        },
        diagnostics={"agent0": dict(status="NO_ALERT", destination="a2_sink")},
    )
    text = "\n".join(line for line, _ in own_lines(frame, "agent0"))
    assert "1.0 ; 2.0" in text and "sortie 2" in text
    assert "987" not in text and "100" not in text
