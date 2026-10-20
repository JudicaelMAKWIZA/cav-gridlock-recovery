"""Sérialisation du graphe de développement historique, sans accès au simulateur."""

import networkx as nx


def cycle_candidates(graph: nx.DiGraph) -> list[dict]:
    groups = []
    for component in nx.strongly_connected_components(graph):
        vehicles = sum(
            graph.nodes[item]["node_type"] == "vehicle" for item in component
        )
        resources = sum(
            graph.nodes[item]["node_type"] == "resource" for item in component
        )
        if vehicles >= 2:
            groups.append(
                {
                    "nodes": sorted(component),
                    "vehicle_count": vehicles,
                    "resource_count": resources,
                }
            )
    return sorted(groups, key=lambda group: group["nodes"])


def closed_cycle_candidates(
    graph: nx.DiGraph, candidates: list | None = None
) -> list[dict]:
    """Retire les échappatoires jusqu'à ce que les contraintes restantes se ferment."""
    closed = []
    pending = [
        set(group["nodes"])
        for group in (cycle_candidates(graph) if candidates is None else candidates)
    ]
    while pending:
        component = pending.pop()
        while True:
            removable = set()
            for item in sorted(component):
                data = graph.nodes[item]
                if data["node_type"] == "vehicle":
                    if not any(
                        target in component for target in graph.successors(item)
                    ):
                        removable.add(item)
                elif not resource_held(graph, item, component):
                    removable.add(item)
            if not removable:
                break
            component -= removable
        # Une grande SCC peut laisser plusieurs groupes circulaires plus petits.
        for remaining in cycle_candidates(graph.subgraph(component)):
            subset = set(remaining["nodes"])
            if all(
                resource_held(graph, item, subset)
                for item in remaining["nodes"]
                if graph.nodes[item]["node_type"] == "resource"
            ):
                closed.append(remaining)
            elif subset != component:
                pending.append(subset)
    return sorted(closed, key=lambda group: group["nodes"])


def resource_held(graph: nx.DiGraph, item: str, component: set) -> bool:
    data = graph.nodes[item]
    if data.get("release_mode") == "one_candidate_free":
        candidate_lanes = data.get("candidate_lanes", [])
        if not candidate_lanes or set(data.get("blocked_lanes", [])) != set(
            candidate_lanes
        ):
            return False
        for lane in candidate_lanes:
            space = data.get("lane_spaces", {}).get(lane, {})
            occupant = (
                "vehicle:" + space["occupant_id"] if space.get("occupant_id") else None
            )
            edge = graph.get_edge_data(item, occupant, {}) if occupant else {}
            available = (
                space.get("free_space_m")
                if space.get("knowledge") != "unknown"
                else None
            )
            bound = (
                available if available is not None else space.get("max_free_space_m")
            )
            if (
                not space.get("occupied")
                or bound is None
                or bound >= data["required_space_m"]
                or occupant not in component
                or edge.get("edge_type") != "occupied_by"
                or lane not in edge.get("lanes", [])
            ):
                return False
        return True
    if data.get("release_mode") == "all_blockers_clear":
        # Un seul bloqueur enfermé suffit ; les autres peuvent partir sans libérer le conflit.
        return any(
            target in component
            and graph.edges[item, target]["edge_type"] == "blocked_by"
            for target in graph.successors(item)
        )
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
        if (
            identity in active
            and graph.nodes[source]["node_type"] == "vehicle"
            and active[identity].get("source_first_halted_s") != halted_since
        ):
            # Une reprise connue entre deux calculs coupe aussi l'âge de la demande.
            events.append(
                {
                    "event": "disappeared",
                    "absent_at_observation_s": time_s,
                    "reason": "halt_period_changed",
                    **active.pop(identity),
                }
            )
        appeared = identity not in active
        if identity not in active:
            active[identity] = {
                "dependency_id": identity,
                "source": source,
                "target": target,
                "edge_type": data["edge_type"],
                "first_seen_s": time_s,
                "source_first_halted_s": halted_since,
                "last_seen_s": time_s,
                "age_s": 0,
                "observations_count": 0,
            }
        row = active[identity]
        row.update(
            last_seen_s=time_s,
            age_s=time_s - row["first_seen_s"],
            observations_count=row["observations_count"] + 1,
        )
        if appeared:
            events.append({"event": "appeared", **row})
        data.update(
            dependency_id=identity,
            dependency_first_seen_s=row["first_seen_s"],
            dependency_last_seen_s=time_s,
            dependency_age_s=row["age_s"],
            observations_count=row["observations_count"],
        )
    for identity in sorted(set(active) - observed):
        events.append(
            {
                "event": "disappeared",
                "absent_at_observation_s": time_s,
                **active.pop(identity),
            }
        )
    causes = {
        item: (
            sorted(
                data["dependency_id"]
                for _, _, data in graph.out_edges("vehicle:" + item, data=True)
            )
            if "vehicle:" + item in graph
            else []
        )
        or [reason]
        for item, reason in graph.graph["waiting_states"].items()
    }
    previous = history.get("causes", {})
    for item in sorted(set(previous) & set(causes)):
        if previous[item] != causes[item]:
            events.append(
                {
                    "event": "cause_changed",
                    "vehicle_id": item,
                    "time_s": time_s,
                    "previous": previous[item],
                    "current": causes[item],
                }
            )
    history["causes"] = causes
    statistics = history.setdefault(
        "statistics",
        {
            "observation_count": 0,
            "dependency_appearances": 0,
            "dependency_disappearances": 0,
            "cause_changes": 0,
            "max_observed_dependency_age_s": 0,
        },
    )
    statistics["observation_count"] += 1
    statistics["dependency_appearances"] += sum(
        e["event"] == "appeared" for e in events
    )
    statistics["dependency_disappearances"] += sum(
        e["event"] == "disappeared" for e in events
    )
    statistics["cause_changes"] += sum(e["event"] == "cause_changed" for e in events)
    statistics["max_observed_dependency_age_s"] = max(
        statistics["max_observed_dependency_age_s"],
        max((r["age_s"] for r in active.values()), default=0),
    )
    return events


def snapshot(graph: nx.DiGraph) -> dict:
    candidates = cycle_candidates(graph)
    return {
        "time_s": graph.graph["time_s"],
        "vehicle_count": sum(
            d["node_type"] == "vehicle" for _, d in graph.nodes(data=True)
        ),
        "resource_count": sum(
            d["node_type"] == "resource" for _, d in graph.nodes(data=True)
        ),
        "edge_count": graph.number_of_edges(),
        "nodes": [{"id": item, **graph.nodes[item]} for item in sorted(graph.nodes)],
        "edges": [
            {"source": a, "target": b, **graph.edges[a, b]}
            for a, b in sorted(graph.edges)
        ],
        "waiting_states": graph.graph.get("waiting_states", {}),
        "receiving_observations": graph.graph.get("receiving_observations", []),
        "junction_observations": graph.graph.get("junction_observations", []),
        "following_observations": graph.graph.get("following_observations", []),
        "tls_states": graph.graph.get("tls_states", {}),
        "cycle_candidates": candidates,
        "closed_cycle_candidates": closed_cycle_candidates(graph, candidates),
    }
