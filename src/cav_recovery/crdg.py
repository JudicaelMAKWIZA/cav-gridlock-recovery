"""Explique les attentes entre véhicules et espace disponible, sans agir sur SUMO."""

import math
from dataclasses import dataclass
from typing import NamedTuple

import networkx as nx


class JunctionFoe(NamedTuple):
    vehicle_id: str
    ego_distance: float
    foe_distance: float
    ego_exit: float
    foe_exit: float
    ego_lane: str
    foe_lane: str
    ego_response: bool
    foe_response: bool


@dataclass(frozen=True)
class Following:
    """Lecture native du suivi ; le gap exclut minGap et les vitesses sont en m/s."""

    vehicle_id: str
    gap_m: float
    follow_speed: float
    relation: str


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
            lanes[lane.getID()]["successors"] = sorted({c.getViaLaneID() or c.getToLane().getID()
                                                       for c in lane.getOutgoing() if c.getToLane().allows("passenger")})
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
    # Les changements de voie autorisés comptent dans la continuation de mission.
    import xml.etree.ElementTree as ET
    attributes = {lane.get("id"): lane.attrib for lane in ET.parse(path).findall("edge/lane")}
    for edge in net.getEdges():
        edge_lanes = edge.getLanes()
        for index, lane in enumerate(edge_lanes):
            lateral = []
            for offset, key in ((1, "changeLeft"), (-1, "changeRight")):
                neighbour = index + offset
                permission = attributes[lane.getID()].get(key)
                if (0 <= neighbour < len(edge_lanes) and edge_lanes[neighbour].allows("passenger")
                        and (permission is None or "passenger" in permission.split() or permission == "all")):
                    lateral.append(edge_lanes[neighbour].getID())
            lanes[lane.getID()]["lateral"] = lateral
    return lanes, movements


def update_waiting(first_halted: dict, readings: dict, time_s: float, halting_speed: float) -> None:
    """Suit chaque pas, pour ne pas manquer une reprise entre deux graphes."""
    for item in list(first_halted):
        if item not in readings or readings[item]["speed"] >= halting_speed:
            del first_halted[item]
    for item, row in readings.items():
        if row["speed"] < halting_speed:
            first_halted.setdefault(item, time_s)


def receiving_spaces(readings: dict, lanes: dict, footprints: dict | None = None) -> dict:
    """Mesure l'espace d'entrée, y compris l'arrière resté sur une autre portion."""
    occupants = {}
    unknown = set()
    for item, row in readings.items():
        body, uncertain = (footprints or {}).get(item, ([(row["lane"], row["lane_position"] - row["length"])], []))
        unknown.update(uncertain)
        for lane, rear in body:
            occupants.setdefault(lane, []).append((rear, item))
    spaces = {}
    for lane, info in lanes.items():
        present = occupants.get(lane, [])
        rear, item = min(present) if present else (info["length_m"], None)
        spaces[lane] = {"free_space_m": None if lane in unknown else max(0, rear),
                        "occupied": bool(present), "occupant_id": item,
                        "knowledge": "unknown" if lane in unknown else "known"}
        if item is not None:
            body = (footprints or {}).get(item, ([(readings[item]["lane"],
                                                 readings[item]["lane_position"] - readings[item]["length"])], []))[0]
            spaces[lane]["occupant_body"] = [{"lane": segment, "rear_m": rear_position}
                                               for segment, rear_position in body]
    return spaces


def route_paths(row: dict, lanes: dict, movements: dict) -> set[str]:
    route = row["route"]
    allowed = {lane for lane, info in lanes.items() if info["edge"] in route}
    for (lane, target), rows in movements.items():
        if any(lanes[lane]["edge"] == a and target == b for a, b in zip(route, route[1:])):
            allowed.update(via for c in rows for via in c.get("internal_lanes", []))
    return allowed


def update_footprints(trails: dict, readings: dict, lanes: dict, movements: dict) -> dict:
    """Suit les portions réellement empruntées ; un raccord ambigu reste inconnu."""
    for item in set(trails) - set(readings):
        del trails[item]
    predecessors = {lane: [] for lane in lanes}
    for lane, info in lanes.items():
        for target in info.get("successors", []):
            predecessors[target].append((lane, False))
        for target in info.get("lateral", []):
            predecessors[target].append((lane, True))
    footprints = {}
    for item, row in readings.items():
        current = row["lane"]
        if row["lane_position"] >= row["length"]:
            trails[item] = ([current], set())
            footprints[item] = ([(current, row["lane_position"] - row["length"])], [])
            continue
        path, uncertain = trails.get(item, ([current], set()))
        if path[-1] != current:
            route = row["route"]
            previous_edge = lanes[path[-1]]["edge"]
            origins = ({previous_edge} if previous_edge in route else
                       {lanes[c["from_lane"]]["edge"] for (lane, _), rows in movements.items()
                        if lane == path[-1] for c in rows})
            starts = [index for index, edge in enumerate(route) if edge in origins]
            if lanes[current].get("internal"):
                end = row["route_index"] + 1
            else:
                end = row.get("route_index", route.index(lanes[current]["edge"]))
            # On ne cherche que le trajet parcouru entre deux lectures, pas toute la mission.
            segment = route[min(starts):end + 1] if starts else []
            allowed = route_paths({**row, "route": segment}, lanes, movements)
            choices = []
            def visit(lane, seen):
                if len(choices) >= 2:
                    return
                if lane == current:
                    choices.append(seen)
                    return
                for target in lanes[lane].get("successors", []) + lanes[lane].get("lateral", []):
                    if target in allowed and target not in seen:
                        visit(target, seen + [target])
            visit(path[-1], [path[-1]])
            lateral = any(b in lanes[a].get("lateral", []) for choice in choices for a, b in zip(choice, choice[1:]))
            if len(choices) == 1 and not lateral:
                path = path + choices[0][1:]
            else:
                # Un changement latéral n'est pas une portion parcourue en longueur.
                uncertain.update(path)
                path = [current]
        remaining = row["length"] - row["lane_position"]
        body = [(current, row["lane_position"] - row["length"])]
        kept = [current]
        for previous in reversed(path[:-1]):
            if remaining <= 0:
                break
            body.append((previous, lanes[previous]["length_m"] - remaining))
            kept.insert(0, previous)
            remaining -= lanes[previous]["length_m"]
        if remaining > 0:
            # On couvre les arrière-trajets possibles sans en choisir un à la place du véhicule.
            route = row["route"]
            end = row["route_index"] + 1 if lanes[current].get("internal") else row.get(
                "route_index", route.index(lanes[current]["edge"]))
            allowed = route_paths({**row, "route": route[:end + 1]}, lanes, movements)
            def uncertain_back(lane, distance, seen):
                for previous, lateral in predecessors[lane]:
                    if previous not in allowed or previous in seen:
                        continue
                    uncertain.add(previous)
                    left = distance if lateral else distance - lanes[previous]["length_m"]
                    if left > 0:
                        uncertain_back(previous, left, seen | {previous})
            uncertain_back(kept[0], remaining, set(kept))
        uncertain.discard(current)
        trails[item] = (kept, uncertain)
        footprints[item] = (body, sorted(uncertain))
    return footprints


def admissible_lanes(row: dict, connections: list, lanes: dict, movements: dict) -> list[str]:
    """Vérifie la suite du trajet, sans exclure un changement de voie légal plus loin."""
    route, index = row["route"], row["route_index"] + 1
    def continues(lane):
        pending, seen = [(lane, index)], set()
        while pending:
            current, at = pending.pop()
            if (current, at) in seen:
                continue
            seen.add((current, at))
            if at == len(route) - 1:
                return True
            pending.extend((c["lane"], at + 1) for c in movements.get((current, route[at + 1]), []))
            pending.extend((other, at) for other in lanes[current].get("lateral", []))
        return False
    return sorted({c["lane"] for c in connections if continues(c["lane"])})


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


def limiting_following(item: str, readings: dict, following: dict, halting_speed: float) -> Following | None:
    evidence = following.get(item)
    if (evidence and evidence.vehicle_id in readings and evidence.vehicle_id != item
            and math.isfinite(evidence.follow_speed) and evidence.follow_speed < halting_speed
            and evidence.relation in ("longitudinal_following", "connection_obstacle")):
        return evidence
    return None


def junction_blockers(row: dict, readings: dict, lanes: dict, movements: dict, observation: dict) -> list[dict]:
    """Garde les conflits natifs encore occupés, pas les adversaires lointains."""
    result = []
    for raw in observation["foes"]:
        foe = JunctionFoe(*raw)
        item, ego_dist, foe_dist, ego_exit, foe_exit, ego_lane, foe_lane, ego_response, foe_response = foe
        if (item not in readings or readings[item]["lane"] != foe_lane
                or not all(math.isfinite(d) for d in (ego_dist, foe_dist, ego_exit, foe_exit))
                or ego_dist < 0 or ego_exit <= ego_dist
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
            # Une occupation ne prouve pas à elle seule que l'ego prioritaire attend ce véhicule.
            if observation.get("connection_blocker") != item:
                continue
            evidence = "native_connection_obstacle"
        else:
            priority = observation["priority_foes"].get((movement["from_lane"], movement["lane"]), [])
            foe_origins = {c["from_lane"] for c in junction_movements(readings[item], lanes, movements)
                           if foe_lane in c["internal_lanes"]}
            if (not ego_response or link[2] or not foe_origins.intersection(priority)
                    or observation.get("stop_line_speed", math.inf) >= observation["halting_speed"]):
                continue
            evidence = "native_priority_at_stopline"
        result.append({"foe_id": item, "ego_distance_m": ego_dist, "foe_distance_m": foe_dist,
                       "ego_exit_distance_m": ego_exit, "foe_exit_distance_m": foe_exit,
                       "ego_response": ego_response, "foe_response": foe_response,
                       "via_lane": ego_lane, "foe_lane": foe_lane, "from_lane": movement["from_lane"],
                       "to_lane": movement["lane"], "junction": lanes[ego_lane]["junction"],
                       "signal_state": link[5] if link[5] in ("G", "g") else None,
                       "link_state": link[5],
                       "link_has_priority": link[1], "link_is_open": link[2], "link_has_foe": link[3],
                       "stop_line_speed_m_per_s": observation["stop_line_speed"],
                       "priority_evidence": evidence})
    return sorted(result, key=lambda r: (r["via_lane"], r["foe_lane"], r["foe_id"]))


def build_graph(readings: dict, spaces: dict, lanes: dict, movements: dict,
                tls_states: dict, first_halted: dict, time_s: float, *,
                halting_speed: float, min_gap_m: float,
                junctions: dict | None = None, following: dict | None = None) -> nx.DiGraph:
    """Reconstruit les seules dépendances observées : A → B signifie A attend B.

    Une voie libre parmi celles actuellement servies suffit : les alternatives
    ne deviennent pas des obligations simultanées. Les conflits de carrefour
    exigent une occupation et des preuves natives du même pas.
    """
    graph = nx.DiGraph(time_s=time_s, receiving_observations=[], junction_observations=[], waiting_states={},
                       tls_states=dict(tls_states),
                       following_observations=[{"vehicle_id": item, "leader_id": e.vehicle_id,
                                                "gap_m": e.gap_m, "follow_speed_m_per_s": e.follow_speed,
                                                "relation": e.relation}
                                               for item, e in sorted((following or {}).items())])
    reasons = {}

    def add_vehicle(item):
        row = readings[item]
        info = lanes[row["lane"]]
        graph.add_node("vehicle:" + item, node_type="vehicle", vehicle_id=item,
                       edge=row["road_id"], lane=row["lane"], street_name=info["street_name"],
                       lane_position_m=row["lane_position"], speed_m_per_s=row["speed"],
                       length_m=row["length"],
                       route_index=row["route_index"], next_edge=next_edge(row),
                       first_halted_s=first_halted.get(item),
                       halted_age_s=time_s - first_halted[item] if item in first_halted else None,
                       waiting_reason=reasons.get(item, "unknown"), internal=info["internal"])

    for item, row in sorted(readings.items()):
        reasons[item] = "unknown"
        if row["speed"] >= halting_speed:
            reasons[item] = "moving"
            continue
        native_following = limiting_following(item, readings, following or {}, halting_speed)
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
        if (native_following and native_following.relation == "connection_obstacle"
                and not info["internal"] and service and not served):
            native_following = None
        if native_following:
            kind = "leader" if native_following.relation == "longitudinal_following" else "connection_obstacle"
            reasons[item] = kind
            add_vehicle(item)
            add_vehicle(native_following.vehicle_id)
            graph.add_edge("vehicle:" + item, "vehicle:" + native_following.vehicle_id, edge_type=kind,
                           leader_gap_m=native_following.gap_m, follow_speed_m_per_s=native_following.follow_speed,
                           evidence="native_follow_speed" if kind == "leader" else "native_connection_obstacle")
        candidates = admissible_lanes(row, available_service, lanes, movements) if target else []
        lane_states = {lane: ("unknown" if spaces[lane].get("knowledge") == "unknown"
                             else "blocked" if spaces[lane]["occupied"] and spaces[lane]["free_space_m"] < required
                             else "free" if spaces[lane]["free_space_m"] >= required else "unknown") for lane in candidates}
        if not info["internal"] and near_end and target and served:
            graph.graph["receiving_observations"].append({"vehicle_id": item, "lane_states": lane_states})
        if (not info["internal"] and near_end and target and served and candidates
                and all(state == "blocked" for state in lane_states.values())):
            reasons[item] = "receiving_space"
            add_vehicle(item)
            resource = f"resource:receiving:{row['lane']}:{target}:{','.join(candidates)}"
            graph.add_node(resource, node_type="resource", resource_type="receiving_space",
                           release_mode="one_candidate_free",
                           current_lane=row["lane"], next_edge=target, candidate_lanes=candidates,
                           blocked_lanes=candidates, required_space_m=required,
                           max_free_space_m=max(spaces[lane]["free_space_m"] for lane in candidates),
                           lane_states=lane_states,
                           street_name=lanes[candidates[0]]["street_name"], service=service,
                           lane_spaces={lane: spaces[lane].copy() for lane in candidates})
            graph.add_edge("vehicle:" + item, resource, edge_type="waits_for", evidence="receiving_occupancy")
            for lane in candidates:
                occupant = spaces[lane]["occupant_id"]
                if occupant is not None and occupant in readings:
                    add_vehicle(occupant)
                    graph.add_edge(resource, "vehicle:" + occupant, edge_type="occupied_by",
                                   lanes=sorted(set(graph.get_edge_data(resource, "vehicle:" + occupant, {})
                                                    .get("lanes", [])) | {lane}))
        if not info["internal"] and service and not served:
            continue
        observation = (junctions or {}).get(item)
        blockers = junction_blockers(row, readings, lanes, movements, observation) if observation else []
        if observation:
            active_foes = {b["foe_id"] for b in blockers}
            foes = []
            for raw in sorted(observation["foes"], key=lambda f: (f[5], f[6], f[0])):
                foe = JunctionFoe(*raw)
                present = readings.get(foe.vehicle_id)
                known = present is not None and present["lane"] == foe.foe_lane
                occupied = known and foe.foe_distance <= 0 < foe.foe_exit + present["length"]
                foes.append({"vehicle_id": foe.vehicle_id,
                             "state": "occupied" if occupied else "not_occupied" if known else "unknown",
                             "active_constraint": foe.vehicle_id in active_foes,
                             "native": [None if isinstance(value, float) and not math.isfinite(value)
                                        else value for value in foe]})
            graph.graph["junction_observations"].append({"vehicle_id": item, "foes": foes})
        for blocker in blockers:
            if blocker["foe_id"] == item:
                continue
            # Le même obstacle natif n'est pas compté deux fois sous deux noms.
            if native_following and native_following.vehicle_id == blocker["foe_id"]:
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
                           evidence=blocker["priority_evidence"],
                           street_name=lanes[blocker["from_lane"]]["street_name"],
                           **{key: blocker[key] for key in (
                               "from_lane", "to_lane",
                               "ego_distance_m", "ego_exit_distance_m", "ego_response", "foe_response",
                               "signal_state", "link_state", "link_has_priority", "link_is_open",
                               "link_has_foe", "priority_evidence", "stop_line_speed_m_per_s")})
            graph.add_edge(resource, "vehicle:" + blocker["foe_id"], edge_type="blocked_by",
                           **{key: blocker[key] for key in (
                               "foe_id", "foe_lane", "foe_distance_m", "foe_exit_distance_m")})
    # Un véhicule cible peut avoir été ajouté avant que sa propre cause soit lue.
    for _, data in graph.nodes(data=True):
        if data["node_type"] == "vehicle":
            causes = sorted({e["edge_type"] if e["edge_type"] != "waits_for"
                             else graph.nodes[target]["resource_type"]
                             for _, target, e in graph.out_edges("vehicle:" + data["vehicle_id"], data=True)})
            data["waiting_reason"] = causes[0] if len(causes) == 1 else "multiple" if causes else reasons[data["vehicle_id"]]
    graph.graph["waiting_states"] = {item: reasons[item] for item, row in sorted(readings.items())
                                     if row["speed"] < halting_speed}
    for observation in graph.graph["following_observations"]:
        observation["active_constraint"] = graph.has_edge("vehicle:" + observation["vehicle_id"],
                                                           "vehicle:" + observation["leader_id"])
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
    """Retire les échappatoires jusqu'à ce que les contraintes restantes se ferment."""
    closed = []
    pending = [set(group["nodes"]) for group in (cycle_candidates(graph) if candidates is None else candidates)]
    while pending:
        component = pending.pop()
        while True:
            removable = set()
            for item in sorted(component):
                data = graph.nodes[item]
                if data["node_type"] == "vehicle":
                    if not any(target in component for target in graph.successors(item)):
                        removable.add(item)
                elif not resource_held(graph, item, component):
                    removable.add(item)
            if not removable:
                break
            component -= removable
        # Une grande SCC peut laisser plusieurs groupes circulaires plus petits.
        for remaining in cycle_candidates(graph.subgraph(component)):
            subset = set(remaining["nodes"])
            if all(resource_held(graph, item, subset)
                   for item in remaining["nodes"] if graph.nodes[item]["node_type"] == "resource"):
                closed.append(remaining)
            elif subset != component:
                pending.append(subset)
    return sorted(closed, key=lambda group: group["nodes"])


def resource_held(graph: nx.DiGraph, item: str, component: set) -> bool:
    data = graph.nodes[item]
    if data.get("release_mode") == "one_candidate_free":
        candidate_lanes = data.get("candidate_lanes", [])
        if not candidate_lanes or set(data.get("blocked_lanes", [])) != set(candidate_lanes):
            return False
        for lane in candidate_lanes:
            space = data.get("lane_spaces", {}).get(lane, {})
            occupant = "vehicle:" + space["occupant_id"] if space.get("occupant_id") else None
            edge = graph.get_edge_data(item, occupant, {}) if occupant else {}
            if (space.get("knowledge") == "unknown" or not space.get("occupied")
                    or space.get("free_space_m") is None
                    or space.get("free_space_m", math.inf) >= data["required_space_m"]
                    or occupant not in component or edge.get("edge_type") != "occupied_by"
                    or lane not in edge.get("lanes", [])):
                return False
        return True
    if data.get("release_mode") == "all_blockers_clear":
        # Un seul bloqueur enfermé suffit ; les autres peuvent partir sans libérer le conflit.
        return any(target in component and graph.edges[item, target]["edge_type"] == "blocked_by"
                   for target in graph.successors(item))
    return False


def track_dependencies(graph: nx.DiGraph, history: dict) -> list[dict]:
    """Date les lectures d'une même contrainte, pas l'arrêt du véhicule."""
    active = history.setdefault("active", {})
    time_s = graph.graph["time_s"]
    events, observed = [], set()
    for source, target, data in sorted(graph.edges(data=True)):
        identity = f"{data['edge_type']}|{source}|{target}"
        observed.add(identity)
        halted_since = graph.nodes[source].get("first_halted_s")
        if (identity in active and graph.nodes[source]["node_type"] == "vehicle"
                and active[identity].get("source_first_halted_s") != halted_since):
            # Une reprise connue entre deux calculs coupe aussi l'âge de la demande.
            events.append({"event": "disappeared", "absent_at_observation_s": time_s,
                           "reason": "halt_period_changed", **active.pop(identity)})
        appeared = identity not in active
        if identity not in active:
            active[identity] = {"dependency_id": identity, "source": source, "target": target,
                                "edge_type": data["edge_type"], "first_seen_s": time_s,
                                "source_first_halted_s": halted_since,
                                "last_seen_s": time_s, "age_s": 0, "observations_count": 0}
        row = active[identity]
        row.update(last_seen_s=time_s, age_s=time_s - row["first_seen_s"],
                   observations_count=row["observations_count"] + 1)
        if appeared:
            events.append({"event": "appeared", **row})
        data.update(dependency_id=identity, dependency_first_seen_s=row["first_seen_s"],
                    dependency_last_seen_s=time_s, dependency_age_s=row["age_s"],
                    observations_count=row["observations_count"])
    for identity in sorted(set(active) - observed):
        events.append({"event": "disappeared", "absent_at_observation_s": time_s, **active.pop(identity)})
    causes = {item: (sorted(data["dependency_id"] for _, _, data in graph.out_edges("vehicle:" + item, data=True))
                    if "vehicle:" + item in graph else []) or [reason]
              for item, reason in graph.graph["waiting_states"].items()}
    previous = history.get("causes", {})
    for item in sorted(set(previous) & set(causes)):
        if previous[item] != causes[item]:
            events.append({"event": "cause_changed", "vehicle_id": item, "time_s": time_s,
                           "previous": previous[item], "current": causes[item]})
    history["causes"] = causes
    statistics = history.setdefault("statistics", {"observation_count": 0, "dependency_appearances": 0,
                                                    "dependency_disappearances": 0, "cause_changes": 0,
                                                    "max_observed_dependency_age_s": 0})
    statistics["observation_count"] += 1
    statistics["dependency_appearances"] += sum(e["event"] == "appeared" for e in events)
    statistics["dependency_disappearances"] += sum(e["event"] == "disappeared" for e in events)
    statistics["cause_changes"] += sum(e["event"] == "cause_changed" for e in events)
    statistics["max_observed_dependency_age_s"] = max(statistics["max_observed_dependency_age_s"],
                                                       max((r["age_s"] for r in active.values()), default=0))
    return events


def snapshot(graph: nx.DiGraph) -> dict:
    candidates = cycle_candidates(graph)
    return {"time_s": graph.graph["time_s"],
            "vehicle_count": sum(d["node_type"] == "vehicle" for _, d in graph.nodes(data=True)),
            "resource_count": sum(d["node_type"] == "resource" for _, d in graph.nodes(data=True)),
            "edge_count": graph.number_of_edges(),
            "nodes": [{"id": item, **graph.nodes[item]} for item in sorted(graph.nodes)],
            "edges": [{"source": a, "target": b, **graph.edges[a, b]} for a, b in sorted(graph.edges)],
            "waiting_states": graph.graph.get("waiting_states", {}),
            "receiving_observations": graph.graph.get("receiving_observations", []),
            "junction_observations": graph.graph.get("junction_observations", []),
            "following_observations": graph.graph.get("following_observations", []),
            "tls_states": graph.graph.get("tls_states", {}),
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
