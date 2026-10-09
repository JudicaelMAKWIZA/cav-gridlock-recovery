"""Vérifie que la page reprend les graphes sources et fonctionne sans dépendance distante."""

import json
import re

import pytest

from cav_recovery.crdg_view import read_snapshots, write_view


def frame(time=5):
    return {"time_s": time, "nodes": [{"id": "vehicle:A", "node_type": "vehicle", "vehicle_id": "A"},
                                      {"id": "vehicle:B", "node_type": "vehicle", "vehicle_id": "B"}],
            "edges": [{"source": "vehicle:A", "target": "vehicle:B", "edge_type": "leader", "evidence": "native_follow_speed"}],
            "cycle_candidates": [], "closed_cycle_candidates": []}


def write_source(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def embedded(path):
    return json.loads(re.search(r'<script id="crdg-data" type="application/json">(.*?)</script>',
                               path.read_text(encoding="utf-8"), flags=re.S)[1])


def test_export_keeps_real_edges_and_only_source_cycles(tmp_path):
    source, output = tmp_path / "crdg.jsonl", tmp_path / "view.html"
    a, b = frame(), frame(10)
    b["edges"].append({"source": "vehicle:B", "target": "vehicle:A", "edge_type": "connection_obstacle"})
    b["cycle_candidates"] = [{"nodes": ["vehicle:A", "vehicle:B"], "vehicle_count": 2, "resource_count": 0}]
    write_source(source, [a, b])
    result = write_view(source, output)
    data = embedded(output)
    assert result["snapshots"] == 2
    assert data["snapshots"] == [a, b]
    assert data["snapshots"][0]["cycle_candidates"] == []
    assert data["snapshots"][1]["closed_cycle_candidates"] == []
    assert data["metadata"]["export_interval_s"] is None
    text = output.read_text(encoding="utf-8")
    assert not re.search(r'<(?:script|link)\b[^>]*(?:src|href)\s*=', text)
    assert "fetch(" not in text and "XMLHttpRequest" not in text and "WebSocket" not in text


def test_empty_dependencies_and_large_graph_are_not_replaced_by_demo_data(tmp_path):
    source = tmp_path / "crdg.jsonl"
    empty = {**frame(), "nodes": [], "edges": []}
    large = frame(10)
    large["nodes"] = [{"id": f"vehicle:{i}", "node_type": "vehicle"} for i in range(1000)]
    large["edges"] = [{"source": f"vehicle:{i}", "target": f"vehicle:{i+1}", "edge_type": "leader"} for i in range(999)]
    write_source(source, [empty, large])
    output = tmp_path / "view.html"
    write_view(source, output)
    data = embedded(output)["snapshots"]
    assert data[0]["nodes"] == data[0]["edges"] == []
    assert len(data[1]["nodes"]) == 1000 and len(data[1]["edges"]) == 999
    assert data[1]["cycle_candidates"] == []


def test_selection_requires_an_exact_exported_time_and_preserves_existing_file(tmp_path):
    source = tmp_path / "crdg.jsonl"
    write_source(source, [frame(), frame(10)])
    assert read_snapshots(source, 10)[0]["time_s"] == 10
    with pytest.raises(ValueError, match="Aucun instant"):
        read_snapshots(source, 7)
    output = tmp_path / "view.html"
    output.write_text("keep")
    with pytest.raises(ValueError, match="existe déjà"):
        write_view(source, output)
    assert output.read_text() == "keep"


@pytest.mark.parametrize("bad", ["absent_node", "duplicate_node", "false_cycle", "unsorted_time", "missing_relation", "missing_cycle_analysis"])
def test_invalid_sources_are_refused_not_repaired(tmp_path, bad):
    row, rows = frame(), None
    if bad == "absent_node":
        row["edges"][0]["target"] = "vehicle:missing"
    elif bad == "duplicate_node":
        row["nodes"].append(row["nodes"][0])
    elif bad == "false_cycle":
        row["cycle_candidates"] = [{"nodes": ["vehicle:missing"]}]
    elif bad == "missing_relation":
        del row["edges"][0]["edge_type"]
    elif bad == "missing_cycle_analysis":
        del row["closed_cycle_candidates"]
    else:
        rows = [row, frame(4)]
    source = tmp_path / "crdg.jsonl"
    write_source(source, rows or [row])
    with pytest.raises(ValueError, match="Graphe illisible"):
        read_snapshots(source)


def test_source_text_cannot_close_the_embedded_data_script(tmp_path):
    row = frame()
    row["nodes"][0]["street_name"] = '</script><script>alert("not executable")</script>'
    source, output = tmp_path / "crdg.jsonl", tmp_path / "view.html"
    write_source(source, [row])
    write_view(source, output)
    assert embedded(output)["snapshots"][0]["nodes"][0]["street_name"] == row["nodes"][0]["street_name"]
    assert '\\u003c/script>' in output.read_text()
