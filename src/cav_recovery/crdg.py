"""Explique les attentes entre véhicules et espace disponible, sans agir sur SUMO."""

import math

import networkx as nx


def read_network(path) -> tuple[dict, dict]:
    """Prépare les voies et les mouvements passenger, y compris leurs traversées."""
    import sumolib

    net = sumolib.net.readNet(str(path), withInternal=True)
    lanes, movements = {}, {}
    for edge in net.getEdges(withInternal=True):
        internal = edge.getFunction() == "internal"
        for lane in edge.getLanes():
            lanes[lane.getID()] = {"edge": edge.getID(), "length_m": lane.getLength(),
                                   "street_name": edge.getName() or None, "internal": internal,
                                   "junction": edge.getToNode().getID()}
            if internal or not lane.allows("passenger"):
                continue
            for connection in lane.getOutgoing():
                target = connection.getToLane()
                if not target.allows("passenger") or connection.getDirection() == "t":
                    continue
                key = (lane.getID(), target.getEdge().getID())
                row = {"lane": target.getID(), "tls": connection.getTLSID() or None,
                       "link_index": connection.getTLLinkIndex(), "from_lane": lane.getID(),
                       "via_lane": connection.getViaLaneID(), "internal_lanes": []}
                via = row["via_lane"]
                while via and via not in row["internal_lanes"]:
                    row["internal_lanes"].append(via)
                    following = [c for c in net.getLane(via).getOutgoing() if c.getToLane() == target]
                    if len(following) != 1:
                        # Sans continuation unique, on ne relie pas la traversée à la mission.
                        row["internal_lanes"] = []
                        break
                    via = following[0].getViaLaneID()
                if via:
                    row["internal_lanes"] = []
                if row not in movements.setdefault(key, []):
                    movements[key].append(row)
    for (_, target), rows in list(movements.items()):
        for row in rows:
            for via in row["internal_lanes"]:
                movements.setdefault((via, target), []).append(row)
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


def junction_movements(row: dict, lanes: dict, movements: dict) -> list[dict]:
    """Relie la voie observée à une continuation vérifiée de la mission."""
    target = next_edge(row)
    if not lanes[row["lane"]]["internal"]:
        return movements.get((row["lane"], target), []) if target else []
    route, index = row["route"], row["route_index"]
    if not 0 <= index < len(route) - 1:
        return []
    return [c for c in movements.get((row["lane"], route[index + 1]), [])
            if lanes[c["from_lane"]]["edge"] == route[index]
            and row["lane"] in c["internal_lanes"]]


def close_leader(item: str, readings: dict, leaders: dict, halting_speed: float, step_s: float) -> bool:
    following = leaders.get(item)
    return bool(following and following[0] in readings and following[0] != item
                and readings[following[0]]["speed"] < halting_speed
                and following[1] <= halting_speed * step_s)


def junction_blockers(row: dict, readings: dict, lanes: dict, movements: dict, observation: dict) -> list[dict]:
    """Garde les conflits natifs encore occupés, pas les adversaires lointains."""
    result = []
    required = row["length"] + observation["min_gap_m"]
    for foe in observation["foes"]:
        item, ego_dist, foe_dist, ego_exit, foe_exit, ego_lane, foe_lane, ego_response, foe_response = foe
        if (item not in readings or readings[item]["lane"] != foe_lane
                or not all(math.isfinite(d) for d in (ego_dist, foe_dist, ego_exit, foe_exit))
                or not 0 <= ego_dist <= required or ego_exit <= ego_dist
                or foe_exit <= foe_dist or not foe_dist <= 0 < foe_exit + readings[item]["length"]):
            continue
        if foe_lane not in observation["internal_foes"].get(ego_lane, []):
            continue
        matches = [c for c in junction_movements(row, lanes, movements) if ego_lane in c["internal_lanes"]]
        if len(matches) != 1:
            continue
        movement = matches[0]
        links = [link for link in observation["links"]
                 if link[0] == movement["lane"] and (link[4] or row["lane"]) in movement["internal_lanes"]]
        if len(links) != 1:
            continue
        link = links[0]
        if link[5] in ("r", "y", "u") or not link[3]:
            continue
        if link[1]:
            # La priorité ne supprime pas une occupation physique du conflit.
            evidence = "native_conflict_occupied"
        else:
            priority = observation["priority_foes"].get((movement["from_lane"], movement["lane"]), [])
            foe_origins = {c["from_lane"] for c in junction_movements(readings[item], lanes, movements)
                           if foe_lane in c["internal_lanes"]}
            if not ego_response or link[2] or not foe_origins.intersection(priority):
                continue
            evidence = "native_response_closed_link_and_occupied_conflict"
        result.append({"foe_id": item, "ego_distance_m": ego_dist, "foe_distance_m": foe_dist,
                       "ego_exit_distance_m": ego_exit, "foe_exit_distance_m": foe_exit,
                       "ego_response": ego_response, "foe_response": foe_response,
                       "via_lane": ego_lane, "foe_lane": foe_lane, "from_lane": movement["from_lane"],
                       "to_lane": movement["lane"], "junction": lanes[ego_lane]["junction"],
                       "signal_state": link[5] if link[5] in ("G", "g") else None,
                       "link_state": link[5],
                       "link_has_priority": link[1], "link_is_open": link[2], "link_has_foe": link[3],
                       "priority_evidence": evidence})
    return sorted(result, key=lambda r: (r["via_lane"], r["foe_lane"], r["foe_id"]))


def build_graph(readings: dict, leaders: dict, spaces: dict, lanes: dict, movements: dict,
                tls_states: dict, first_halted: dict, time_s: float, *, min_wait_s: float,
                halting_speed: float, min_gap_m: float, step_s: float,
                junctions: dict | None = None) -> nx.DiGraph:
    """Reconstruit les seules dépendances observées : A → B signifie A attend B.

    Une voie libre parmi celles actuellement servies suffit : les alternatives
    ne deviennent pas des obligations simultanées. Les conflits de carrefour
    exigent une occupation et des preuves natives du même pas.
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
        blocked_by_leader = close_leader(item, readings, leaders, halting_speed, step_s)
        info = lanes[row["lane"]]
        target = next_edge(row)
        required = row["length"] + min_gap_m
        near_end = info["length_m"] - row["lane_position"] <= required
        connections = movements.get((row["lane"], target), []) if target else []
        service = [{**c, "state": tls_states[c["tls"]][c["link_index"]] if c["tls"] else None}
                   for c in connections]
        available_service = [c for c in service if c["state"] in (None, "G", "g")]
        served = bool(available_service)
        if near_end and service and not served:
            reasons[item] = "signal"
        if blocked_by_leader:
            reasons[item] = "leader"
        if item not in first_halted or time_s - first_halted[item] < min_wait_s:
            continue
        if blocked_by_leader:
            add_vehicle(item)
            add_vehicle(following[0])
            graph.add_edge("vehicle:" + item, "vehicle:" + following[0], edge_type="leader",
                           leader_gap_m=following[1], progress_margin_m=halting_speed * step_s)
            continue
        candidates = sorted({c["lane"] for c in available_service})
        if (not info["internal"] and near_end and target and served and candidates
                and all(spaces[lane]["occupied"] and spaces[lane]["free_space_m"] < required
                        for lane in candidates)):
            reasons[item] = "receiving_space"
            add_vehicle(item)
            resource = f"resource:receiving:{row['lane']}:{target}"
            graph.add_node(resource, node_type="resource", resource_type="receiving_space",
                           release_mode="one_candidate_free",
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
            continue
        if not info["internal"] and service and not served:
            continue
        observation = (junctions or {}).get(item)
        blockers = junction_blockers(row, readings, lanes, movements, observation) if observation else []
        for blocker in blockers:
            if blocker["foe_id"] == item:
                continue
            reasons[item] = "junction_conflict"
            add_vehicle(item)
            add_vehicle(blocker["foe_id"])
            resource = f"resource:junction:{blocker['via_lane']}:{blocker['foe_lane']}"
            graph.add_node(resource, node_type="resource", resource_type="junction_conflict",
                           release_mode="all_blockers_clear", via_lane=blocker["via_lane"],
                           junction=blocker["junction"], street_name=lanes[blocker["via_lane"]]["street_name"],
                           foe_lanes=[blocker["foe_lane"]])
            # Chaque demande garde ses propres distances et sa priorité courante.
            graph.add_edge("vehicle:" + item, resource, edge_type="waits_for",
                           street_name=lanes[blocker["from_lane"]]["street_name"],
                           **{key: blocker[key] for key in (
                               "from_lane", "to_lane",
                               "ego_distance_m", "ego_exit_distance_m", "ego_response", "foe_response",
                               "signal_state", "link_state", "link_has_priority", "link_is_open",
                               "link_has_foe", "priority_evidence")})
            graph.add_edge(resource, "vehicle:" + blocker["foe_id"], edge_type="blocked_by",
                           **{key: blocker[key] for key in (
                               "foe_id", "foe_lane", "foe_distance_m", "foe_exit_distance_m")})
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


def closed_cycle_candidates(graph: nx.DiGraph, candidates: list | None = None) -> list[dict]:
    """Écarte une SCC si une alternative de réception peut être libérée hors du groupe."""
    closed = []
    for group in cycle_candidates(graph) if candidates is None else candidates:
        component = set(group["nodes"])
        valid = True
        for item in group["nodes"]:
            data = graph.nodes[item]
            if data["node_type"] != "resource":
                continue
            if data.get("release_mode") == "one_candidate_free":
                candidates_lanes = data.get("candidate_lanes", [])
                if not candidates_lanes or set(data.get("blocked_lanes", [])) != set(candidates_lanes):
                    valid = False
                    break
                for lane in candidates_lanes:
                    space = data.get("lane_spaces", {}).get(lane, {})
                    occupant = "vehicle:" + space["occupant_id"] if space.get("occupant_id") else None
                    edge = graph.get_edge_data(item, occupant, {}) if occupant else {}
                    if (not space.get("occupied") or space.get("free_space_m", math.inf) >= data["required_space_m"]
                            or occupant not in component or edge.get("edge_type") != "occupied_by"
                            or lane not in edge.get("lanes", [])):
                        valid = False
                        break
            elif data.get("release_mode") == "all_blockers_clear":
                # On reste prudent : même les bloqueurs connus extérieurs font refuser le groupe.
                successors = list(graph.successors(item))
                valid = bool(successors) and all(target in component
                                                and graph.edges[item, target]["edge_type"] == "blocked_by"
                                                for target in successors)
            else:
                valid = False
            if not valid:
                break
        if valid:
            closed.append(group)
    return closed


def snapshot(graph: nx.DiGraph) -> dict:
    candidates = cycle_candidates(graph)
    return {"time_s": graph.graph["time_s"],
            "vehicle_count": sum(d["node_type"] == "vehicle" for _, d in graph.nodes(data=True)),
            "resource_count": sum(d["node_type"] == "resource" for _, d in graph.nodes(data=True)),
            "edge_count": graph.number_of_edges(),
            "nodes": [{"id": item, **graph.nodes[item]} for item in sorted(graph.nodes)],
            "edges": [{"source": a, "target": b, **graph.edges[a, b]} for a, b in sorted(graph.edges)],
            "cycle_candidates": candidates,
            "closed_cycle_candidates": closed_cycle_candidates(graph, candidates)}


def empty_summary() -> dict:
    return {"sample_count": 0, "snapshots_with_dependencies": 0, "first_dependency_s": None,
            "max_vehicle_nodes": 0, "max_resource_nodes": 0, "max_edges": 0,
            "snapshots_with_cycle_candidates": 0, "first_cycle_candidate_s": None,
            "max_cycle_vehicle_count": 0, "max_cycle_resource_count": 0,
            "snapshots_with_junction_dependencies": 0, "first_junction_dependency_s": None,
            "max_junction_resources": 0, "snapshots_with_closed_cycle_candidates": 0,
            "first_closed_cycle_candidate_s": None,
            "max_closed_cycle_vehicle_count": 0, "max_closed_cycle_resource_count": 0}


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
    junctions = sum(row.get("resource_type") == "junction_conflict" for row in current["nodes"])
    if junctions:
        summary["snapshots_with_junction_dependencies"] += 1
        if summary["first_junction_dependency_s"] is None:
            summary["first_junction_dependency_s"] = current["time_s"]
    summary["max_junction_resources"] = max(summary["max_junction_resources"], junctions)
    closed = current.get("closed_cycle_candidates", [])
    if closed:
        summary["snapshots_with_closed_cycle_candidates"] += 1
        if summary["first_closed_cycle_candidate_s"] is None:
            summary["first_closed_cycle_candidate_s"] = current["time_s"]
        summary["max_closed_cycle_vehicle_count"] = max(summary["max_closed_cycle_vehicle_count"],
                                                        max(g["vehicle_count"] for g in closed))
        summary["max_closed_cycle_resource_count"] = max(summary["max_closed_cycle_resource_count"],
                                                         max(g["resource_count"] for g in closed))

    def rank(row):
        closed = row.get("closed_cycle_candidates", [])
        if closed:
            return (2, max(g["vehicle_count"] for g in closed))
        groups = row["cycle_candidates"]
        return (1, max(g["vehicle_count"] for g in groups)) if groups else (0, row["edge_count"])

    # En cas d'égalité, on garde le premier pas, pas le dernier.
    return current if peak is None or rank(current) > rank(peak) else peak
