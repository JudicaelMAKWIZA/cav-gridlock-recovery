"""Le panneau INFO projette les preuves sans les recalculer ni les enrichir."""

from copy import deepcopy

from cav_recovery.experiments.info_graph import (
    detector_relations,
    graph_pages,
    graph_positions,
    message_snapshot,
    physical_snapshot,
)


def relation(requester="agent0", blocker="agent1", **extra):
    return dict(
        requester=requester,
        blocker=blocker,
        movement_id="turn-left",
        resource_wkt="POLYGON ((0 0, 2 0, 2 2, 0 0))",
        observed_at_s=10.4,
        level="SUSPECTED",
        **extra,
    )


def test_physical_projection_keeps_resource_and_original_proof_unchanged():
    rows = [relation(), relation("agent1", "agent2")]
    before = deepcopy(rows)
    snapshot = physical_snapshot(rows, {})
    assert rows == before
    assert {row["kind"] for row in snapshot["nodes"].values()} == {
        "vehicle",
        "resource",
    }
    for a, middle, b in snapshot["units"]:
        assert snapshot["nodes"][a]["kind"] == snapshot["nodes"][b]["kind"] == "vehicle"
        assert snapshot["nodes"][middle]["kind"] == "resource"
        assert any(
            edge["source"] == a and edge["target"] == middle
            for edge in snapshot["edges"]
        )
        assert any(
            edge["source"] == middle and edge["target"] == b
            for edge in snapshot["edges"]
        )
    assert all(edge["level"] == "SUSPECTED" for edge in snapshot["edges"])


def test_missing_physical_space_does_not_create_a_fictitious_resource():
    row = relation()
    del row["resource_wkt"]
    assert physical_snapshot([row], {}) == dict(nodes={}, edges=[], units=[])


def test_detector_occupancy_does_not_inherit_temporal_confirmation():
    diagnostic = dict(
        ego_id="agent0",
        status="CONFIRMED_OBSTRUCTION",
        movement_id="turn-left",
        current_occupants=[
            dict(
                sender_id="agent1",
                observed_at_s=40.4,
                resource_polygon_wkt=relation()["resource_wkt"],
            )
        ],
    )
    before = deepcopy(diagnostic)
    rows = detector_relations(diagnostic)
    assert diagnostic == before and rows[0]["level"] == "OBSERVED"
    assert physical_snapshot(rows, {})["edges"][0]["level"] == "OBSERVED"


def messages():
    return [
        dict(
            channel=channel,
            message_id=f"packet-{index}",
            sender=f"agent{index + 1}",
            recipients=["agent0"],
            observed_at_s=59.8,
            sent_at_s=59.8,
            received_at_s=60.0,
        )
        for index, channel in enumerate(["diagnostic"] * 8 + ["telemetry"])
    ]


def test_v2v_uses_actual_packets_and_receipts_without_road_resources():
    rows = messages()
    before = deepcopy(rows)
    pairs = {(row["sender"], "agent0") for row in rows}
    snapshot = message_snapshot(rows, pairs, {})
    assert rows == before
    assert len(snapshot["units"]) == 8
    assert {row["kind"] for row in snapshot["nodes"].values()} == {"vehicle", "message"}
    assert all(
        snapshot["nodes"][middle]["received_at_s"] == 60
        for _, middle, _ in snapshot["units"]
    )
    assert len(message_snapshot(rows, pairs, {}, useful=False)["units"]) == 9
    assert not message_snapshot(rows, set(), {})["nodes"]


def test_dense_message_pages_keep_triplets_complete_and_readable():
    rows = messages()
    pairs = {(row["sender"], "agent0") for row in rows}
    snapshot = message_snapshot(rows, pairs, {}, useful=False)
    pages = graph_pages(snapshot, "agent0", budget=12, middle_budget=3)
    assert len(pages) == 3
    for page in pages:
        assert sum(snapshot["nodes"][key]["kind"] == "message" for key in page) <= 3
        for key in page:
            assert any(key in unit and set(unit) <= page for unit in snapshot["units"])
    assert all(any(set(unit) <= page for page in pages) for unit in snapshot["units"])


def test_display_components_prioritize_selected_ego_and_layout_is_deterministic():
    snapshot = physical_snapshot(
        [relation("agent5", "agent6"), dict(relation(), movement_id="straight")], {}
    )
    pages = graph_pages(snapshot, "agent0")
    assert "vehicle:agent0" in pages[0]
    nodes = tuple(snapshot["nodes"])
    edges = tuple((e["source"], e["target"]) for e in snapshot["edges"])
    assert graph_positions(nodes, edges) == graph_positions(nodes, edges)
    assert set(graph_positions(nodes, edges)) == set(nodes)
