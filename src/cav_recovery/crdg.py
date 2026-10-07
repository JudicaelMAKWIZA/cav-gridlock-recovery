"""Explique les attentes entre véhicules et espace disponible, sans agir sur SUMO."""

import networkx as nx


def read_network(path) -> tuple[dict, dict]:
    """Prépare les longueurs, noms et raccords passenger une seule fois."""
    import sumolib

    net = sumolib.net.readNet(str(path), withInternal=True)
    lanes, movements = {}, {}
    for edge in net.getEdges(withInternal=True):
        internal = edge.getFunction() == "internal"
        for lane in edge.getLanes():
            lanes[lane.getID()] = {"edge": edge.getID(), "length_m": lane.getLength(),
                                   "street_name": edge.getName() or None, "internal": internal}
            if internal or not lane.allows("passenger"):
                continue
            for connection in lane.getOutgoing():
                target = connection.getToLane()
                if not target.allows("passenger") or connection.getDirection() == "t":
                    continue
                key = (lane.getID(), target.getEdge().getID())
                row = {"lane": target.getID(), "tls": connection.getTLSID() or None,
                       "link_index": connection.getTLLinkIndex()}
                if row not in movements.setdefault(key, []):
                    movements[key].append(row)
    for rows in movements.values():
        rows.sort(key=lambda r: (r["lane"], r["tls"] or "", r["link_index"]))
    return lanes, movements


def update_waiting(first_halted: dict, readings: dict, time_s: float, halting_speed: float) -> None:
    """Suit chaque pas, pour ne pas manquer une reprise entre deux graphes."""
    for item in list(first_halted):
        if item not in readings or readings[item]["speed"] >= halting_speed:
            del first_halted[item]
    for item, row in readings.items():
        if row["speed"] < halting_speed:
            first_halted.setdefault(item, time_s)


def receiving_spaces(readings: dict, lanes: dict) -> dict:
    """Mesure la place devant l'arrière du véhicule le plus proche de l'entrée."""
    occupants = {}
    for item, row in readings.items():
        rear = row["lane_position"] - row["length"]
        occupants.setdefault(row["lane"], []).append((rear, item))
    spaces = {}
    for lane, info in lanes.items():
        present = occupants.get(lane, [])
        rear, item = min(present) if present else (info["length_m"], None)
        spaces[lane] = {"free_space_m": max(0, rear), "occupied": bool(present), "occupant_id": item}
    return spaces


def next_edge(row: dict) -> str | None:
    route, index = row["route"], row["route_index"]
    if 0 <= index < len(route) - 1 and route[index] == row["road_id"]:
        return route[index + 1]
    return None


def build_graph(readings: dict, leaders: dict, spaces: dict, lanes: dict, movements: dict,
                tls_states: dict, first_halted: dict, time_s: float, *, min_wait_s: float,
                halting_speed: float, min_gap_m: float, step_s: float) -> nx.DiGraph:
    """Reconstruit les seules dépendances observées : A → B signifie A attend B.

    Une voie libre parmi celles actuellement servies suffit : les alternatives
    ne deviennent pas des obligations simultanées. Les conflits internes ne sont pas modélisés.
    """
    graph = nx.DiGraph(time_s=time_s)
    reasons = {}

    def add_vehicle(item):
        row = readings[item]
        info = lanes[row["lane"]]
        graph.add_node("vehicle:" + item, node_type="vehicle", vehicle_id=item,
                       edge=row["road_id"], lane=row["lane"], street_name=info["street_name"],
                       lane_position_m=row["lane_position"], speed_m_per_s=row["speed"],
                       route_index=row["route_index"], next_edge=next_edge(row),
                       first_halted_s=first_halted.get(item),
                       waiting_s=time_s - first_halted[item] if item in first_halted else 0,
                       waiting_reason=reasons.get(item, "unknown"), internal=info["internal"])

    for item, row in sorted(readings.items()):
        reasons[item] = "unknown"
        if row["speed"] >= halting_speed:
            reasons[item] = "moving"
            continue
        following = leaders.get(item)
        # getLeader donne un écart hors minGap. On exige un espace de progrès
        # inférieur à celui d'un pas encore considéré comme arrêté.
        close_leader = (following and following[0] in readings and following[0] != item
                        and readings[following[0]]["speed"] < halting_speed
                        and following[1] <= halting_speed * step_s)
        info = lanes[row["lane"]]
        target = next_edge(row)
        required = row["length"] + min_gap_m
        near_end = info["length_m"] - row["lane_position"] <= required
        connections = movements.get((row["lane"], target), []) if target else []
        service = [{**c, "state": tls_states[c["tls"]][c["link_index"]] if c["tls"] else None}
                   for c in connections]
        # Un g peut encore devoir céder le passage ; ces conflits restent hors modèle.
        available_service = [c for c in service if c["state"] in (None, "G", "g")]
        served = bool(available_service)
        if near_end and service and not served:
            reasons[item] = "signal"
        if close_leader:
            reasons[item] = "leader"
        if item not in first_halted or time_s - first_halted[item] < min_wait_s:
            continue
        # Les leaders SUMO restent observables sur une voie interne ; on n'y
        # invente pas de ressource de conflit ou de réception.
        if close_leader:
            add_vehicle(item)
            add_vehicle(following[0])
            graph.add_edge("vehicle:" + item, "vehicle:" + following[0], edge_type="leader",
                           leader_gap_m=following[1], progress_margin_m=halting_speed * step_s)
        if info["internal"] or not near_end or not target or not served:
            continue
        # service garde les connexions légales ; candidate_lanes ne garde que
        # les voies utilisables maintenant, sans alternative rouge ou jaune.
        candidates = sorted({c["lane"] for c in available_service})
        if not candidates:
            continue
        # Une voie vide est libre, même sur les petits raccords SUMO où une
        # carrosserie peut chevaucher plusieurs portions de route.
        if any(not spaces[lane]["occupied"] or spaces[lane]["free_space_m"] >= required
               for lane in candidates):
            continue
        if close_leader and readings[following[0]]["lane"] == row["lane"]:
            continue
        reasons[item] = "receiving_space"
        add_vehicle(item)
        resource = f"resource:receiving:{row['lane']}:{target}"
        graph.add_node(resource, node_type="resource", resource_type="receiving_space",
                       current_lane=row["lane"], next_edge=target, candidate_lanes=candidates,
                       blocked_lanes=candidates, required_space_m=required,
                       max_free_space_m=max(spaces[lane]["free_space_m"] for lane in candidates),
                       street_name=lanes[candidates[0]]["street_name"], service=service,
                       lane_spaces={lane: spaces[lane].copy() for lane in candidates})
        graph.add_edge("vehicle:" + item, resource, edge_type="waits_for")
        for lane in candidates:
            occupant = spaces[lane]["occupant_id"]
            if occupant is not None and occupant in readings and readings[occupant]["lane"] == lane:
                add_vehicle(occupant)
                graph.add_edge(resource, "vehicle:" + occupant, edge_type="occupied_by",
                               lanes=sorted(set(graph.get_edge_data(resource, "vehicle:" + occupant, {})
                                                .get("lanes", [])) | {lane}))
    # Un véhicule cible peut avoir été ajouté avant que sa propre cause soit lue.
    for _, data in graph.nodes(data=True):
        if data["node_type"] == "vehicle":
            data["waiting_reason"] = reasons[data["vehicle_id"]]
    return graph


def cycle_candidates(graph: nx.DiGraph) -> list[dict]:
    groups = []
    for component in nx.strongly_connected_components(graph):
        vehicles = sum(graph.nodes[item]["node_type"] == "vehicle" for item in component)
        resources = sum(graph.nodes[item]["node_type"] == "resource" for item in component)
        if vehicles >= 2:
            groups.append({"nodes": sorted(component), "vehicle_count": vehicles, "resource_count": resources})
    return sorted(groups, key=lambda group: group["nodes"])


def snapshot(graph: nx.DiGraph) -> dict:
    return {"time_s": graph.graph["time_s"],
            "vehicle_count": sum(d["node_type"] == "vehicle" for _, d in graph.nodes(data=True)),
            "resource_count": sum(d["node_type"] == "resource" for _, d in graph.nodes(data=True)),
            "edge_count": graph.number_of_edges(),
            "nodes": [{"id": item, **graph.nodes[item]} for item in sorted(graph.nodes)],
            "edges": [{"source": a, "target": b, **graph.edges[a, b]} for a, b in sorted(graph.edges)],
            "cycle_candidates": cycle_candidates(graph)}


def empty_summary() -> dict:
    return {"sample_count": 0, "snapshots_with_dependencies": 0, "first_dependency_s": None,
            "max_vehicle_nodes": 0, "max_resource_nodes": 0, "max_edges": 0,
            "snapshots_with_cycle_candidates": 0, "first_cycle_candidate_s": None,
            "max_cycle_vehicle_count": 0, "max_cycle_resource_count": 0}


def record_snapshot(summary: dict, current: dict, peak: dict | None) -> dict:
    summary["sample_count"] += 1
    if current["edge_count"]:
        summary["snapshots_with_dependencies"] += 1
        if summary["first_dependency_s"] is None:
            summary["first_dependency_s"] = current["time_s"]
    for field, source in (("max_vehicle_nodes", "vehicle_count"), ("max_resource_nodes", "resource_count"),
                          ("max_edges", "edge_count")):
        summary[field] = max(summary[field], current[source])
    if current["cycle_candidates"]:
        summary["snapshots_with_cycle_candidates"] += 1
        if summary["first_cycle_candidate_s"] is None:
            summary["first_cycle_candidate_s"] = current["time_s"]
        summary["max_cycle_vehicle_count"] = max(summary["max_cycle_vehicle_count"],
                                                 max(g["vehicle_count"] for g in current["cycle_candidates"]))
        summary["max_cycle_resource_count"] = max(summary["max_cycle_resource_count"],
                                                  max(g["resource_count"] for g in current["cycle_candidates"]))

    def rank(row):
        groups = row["cycle_candidates"]
        return (bool(groups), max(g["vehicle_count"] for g in groups) if groups else row["edge_count"])

    # En cas d'égalité, on garde le premier pas, pas le dernier.
    return current if peak is None or rank(current) > rank(peak) else peak
