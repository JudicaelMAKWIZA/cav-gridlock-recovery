"""Qualifie un chemin occupé depuis l'état courant, jamais depuis une référence."""

import math

import networkx as nx
from shapely.geometry import Polygon

from .observation import rectangle


def movement_occupancy(row, others, surface, max_distance_m=0.5):
    motion = row["desired_motion"]
    poses = motion.get("path_poses", [])
    tested = 0.0
    previous = row["position_xy_m"]
    first = None
    blockers = set()
    outside = False
    for xy, heading in poses:
        tested += math.dist(previous, xy)
        previous = xy
        if tested > max_distance_m + 1e-7:
            break
        shape = rectangle(xy, heading, row["length_m"], row["width_m"])
        hits = [
            key
            for key, other in sorted(others.items())
            if shape.intersection(Polygon(other["body_xy_m"])).area > 1e-8
        ]
        if hits and first is None:
            first = tested
        blockers.update(hits)
        outside = outside or not surface.covers(shape)
    covered = bool(poses) and motion.get("path_complete", False) and not outside
    state = "blocked" if blockers else "free_on_tested_path" if covered else "unknown"
    return dict(
        state=state,
        first_body_intersection_m=first,
        blockers=sorted(blockers),
        tested_path_length_m=min(tested, max_distance_m),
        outside_surface=outside,
        alternatives="unknown_not_evaluated",
        absolute_blockage="unknown",
        evidence="current_body_intersection_on_assigned_native_path",
        source="global_development_instrumentation_not_distributed_knowledge",
    )


def build_graph(time_s, readings, surface, first_halted):
    graph = nx.DiGraph(time_s=time_s, waiting_states={})
    evidence = {}
    for key, row in sorted(readings.items()):
        graph.add_node(
            "vehicle:" + key,
            node_type="vehicle",
            vehicle_id=key,
            position_xy_m=row["position_xy_m"],
            first_halted_s=first_halted.get(key),
        )
    for key, row in sorted(readings.items()):
        observed = movement_occupancy(
            row, {k: v for k, v in readings.items() if k != key}, surface
        )
        evidence[key] = observed
        qualifies = (
            math.hypot(*row["velocity_xy_mps"]) < 0.1
            and observed["state"] == "blocked"
            and not observed["outside_surface"]
            and row["desired_motion"].get("path_complete", False)
            and row["desired_motion"].get("requested_forward", False)
        )
        graph.graph["waiting_states"][key] = (
            "limited_assigned_path" if qualifies else "no_qualified_dependency"
        )
        if not qualifies:
            continue
        mission = row["mission"]
        movement = (
            "/".join(map(str, mission["origin_lane"])) + "->" + mission["destination"]
        )
        # Deux CAV du même trajet ne demandent pas forcément la même portion à cet instant.
        resource = "resource:movement:" + movement + ":request:" + key
        graph.add_node(
            resource,
            node_type="resource",
            resource_type="movement_occupancy",
            movement=movement,
            requested_by=key,
            release_mode="all_blockers_clear",
            proof_scope="assigned_forward_path_within_0.5m_not_all_manoeuvres",
            **observed,
        )
        graph.add_edge(
            "vehicle:" + key,
            resource,
            edge_type="waits_for",
            qualification="limited_assigned_forward_path_not_absolute_blockage",
        )
        for other in observed["blockers"]:
            graph.add_edge(
                resource,
                "vehicle:" + other,
                edge_type="blocked_by",
                evidence="current_measured_body_intersection",
            )
    return graph, evidence
