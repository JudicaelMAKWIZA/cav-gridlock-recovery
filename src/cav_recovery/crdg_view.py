"""Produit une page locale à partir des graphes C-RDG réellement exportés."""

import hashlib
import json
import math
from pathlib import Path


NODE_FIELDS = ("id", "node_type", "vehicle_id", "resource_type", "street_name", "lane", "edge",
               "current_lane", "next_edge", "speed_m_per_s", "lane_position_m", "waiting_reason",
               "halted_age_s", "release_mode", "candidate_lanes", "lane_states", "required_space_m",
               "max_free_space_m", "via_lane", "foe_lanes", "junction", "internal", "lane_spaces", "service")


def read_snapshots(path, time_s=None):
    """Lit en séquence ; garde les arcs et cycles sources, sans les recalculer."""
    frames, previous = [], None
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                time = row["time_s"]
                if type(time) not in (float, int) or not math.isfinite(time) or (previous is not None and time <= previous):
                    raise ValueError("Les dates exportées doivent être finies et croissantes.")
                previous = time
                if time_s is not None and time != time_s:
                    continue
                nodes = [{key: node[key] for key in NODE_FIELDS if key in node} for node in row["nodes"]]
                ids = {node["id"] for node in nodes}
                if (len(ids) != len(nodes) or any(not isinstance(item, str) or not item for item in ids)
                        or any(node["node_type"] not in ("vehicle", "resource") for node in nodes)):
                    raise ValueError("Nœuds du graphe invalides.")
                edges = row["edges"]
                if any(e["source"] not in ids or e["target"] not in ids for e in edges):
                    raise ValueError("Un arc référence un nœud absent.")
                if any(not isinstance(e["edge_type"], str) or not e["edge_type"] for e in edges):
                    raise ValueError("Un arc doit avoir un type de relation.")
                cycles = {kind: row[kind] for kind in ("cycle_candidates", "closed_cycle_candidates")}
                if any(not set(c["nodes"]).issubset(ids) for candidates in cycles.values() for c in candidates):
                    raise ValueError("Un candidat référence un nœud absent.")
                frames.append({"time_s": time, "nodes": nodes, "edges": edges, **cycles})
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Graphe illisible à la ligne {line_number} : {error}") from error
    if not frames:
        raise ValueError("Aucun instant exporté ne correspond à la sélection.")
    return frames


def write_view(source, output, *, time_s=None):
    """Crée un HTML autonome ; les observations physiques brutes restent dans les sorties SUMO."""
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("Le fichier de visualisation existe déjà.")
    frames = read_snapshots(source, time_s)
    summary_path = source.with_name("crdg_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    data = {"snapshots": frames, "metadata": {"source": source.name, "source_sha256": digest.hexdigest(),
            "calculation_interval_s": summary.get("calculation_interval_s"),
            "export_interval_s": summary.get("export_interval_s"), "selection_time_s": time_s,
            "producer_code": summary.get("code"), "network_sha256": summary.get("network_sha256")}}
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("crdg_view.html").read_text(encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__CRDG_DATA__", encoded), encoding="utf-8")
    return {"snapshots": len(frames), "source_sha256": digest.hexdigest(), "output": str(output.resolve())}
