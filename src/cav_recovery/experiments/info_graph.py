"""Projection d'affichage des preuves, dans l'esprit du panneau historique INFO C-RDG.

Les nœuds intermédiaires rendent explicites les espaces requis et les messages.
Cette projection ne construit aucune nouvelle dépendance pour les modèles.
"""

from functools import lru_cache
import hashlib
import math

import networkx as nx

from .presentation import label


def physical_snapshot(relations, aliases):
    nodes, edges, units = {}, [], []
    for row in relations:
        requester, blocker = row["requester"], row["blocker"]
        geometry = row.get("resource_wkt")
        # Sans espace étayé, ne pas inventer un rectangle pour compléter le dessin.
        if not geometry:
            continue
        identity = row.get("movement_id", "") + "\0" + geometry
        resource = "resource:" + hashlib.sha256(identity.encode()).hexdigest()[:24]
        if resource not in aliases:
            aliases[resource] = f"R{len(aliases) + 1}"
        a, b = "vehicle:" + requester, "vehicle:" + blocker
        for key, vehicle in ((a, requester), (b, blocker)):
            nodes[key] = dict(kind="vehicle", label=label(vehicle), vehicle_id=vehicle)
        nodes[resource] = dict(
            kind="resource",
            label=aliases[resource],
            resource_wkt=geometry,
            movement_id=row.get("movement_id"),
            observed_at_s=row.get("observed_at_s"),
            level=row.get("level", "OBSERVED"),
        )
        for source, target, role in (
            (a, resource, "demande"),
            (resource, b, "occupée par"),
        ):
            edge = dict(
                source=source,
                target=target,
                role=role,
                level=row.get("level", "OBSERVED"),
                proof=row,
            )
            if not any(
                e["source"] == source and e["target"] == target and e["role"] == role
                for e in edges
            ):
                edges.append(edge)
        units.append((a, resource, b))
    return dict(nodes=nodes, edges=edges, units=units)


def detector_relations(diagnostic):
    """L'occupation courante reste distincte de la qualification temporelle D01."""
    return [
        dict(
            requester=diagnostic["ego_id"],
            blocker=row["sender_id"],
            movement_id=diagnostic.get("movement_id", ""),
            resource_wkt=row["resource_polygon_wkt"],
            observed_at_s=row.get("observed_at_s"),
            level="OBSERVED",
        )
        for row in diagnostic.get("current_occupants", [])
    ]


def message_snapshot(messages, allowed_pairs, aliases, *, useful=True):
    nodes, edges, units = {}, [], []
    for item in messages:
        if useful and item["channel"] != "diagnostic":
            continue
        sender = item["sender"]
        recipients = [
            key for key in item["recipients"] if (sender, key) in allowed_pairs
        ]
        if not recipients:
            continue
        middle = "message:" + item["channel"] + ":" + item["message_id"]
        if middle not in aliases:
            aliases[middle] = f"M{len(aliases) + 1}"
        nodes[middle] = dict(kind="message", label=aliases[middle], **item)
        a = "vehicle:" + sender
        nodes[a] = dict(kind="vehicle", label=label(sender), vehicle_id=sender)
        edges.append(dict(source=a, target=middle, role="émet", level="MESSAGE"))
        for recipient in recipients:
            b = "vehicle:" + recipient
            nodes[b] = dict(
                kind="vehicle", label=label(recipient), vehicle_id=recipient
            )
            edges.append(dict(source=middle, target=b, role="livré à", level="MESSAGE"))
            units.append((a, middle, b))
    return dict(nodes=nodes, edges=edges, units=units)


def graph_pages(snapshot, selected, budget=12, middle_budget=3):
    """Garde les triplets complets ; les pages ne sont pas des groupes d'obstruction."""
    graph = nx.Graph()
    graph.add_nodes_from(snapshot["nodes"])
    graph.add_edges_from((edge["source"], edge["target"]) for edge in snapshot["edges"])
    focus = "vehicle:" + str(selected)
    components = sorted(
        nx.connected_components(graph), key=lambda c: (focus not in c, sorted(c))
    )
    pages = []
    for component in components:
        middle_count = sum(
            snapshot["nodes"][key]["kind"] != "vehicle" for key in component
        )
        if len(component) <= budget and middle_count <= middle_budget:
            pages.append(component)
            continue
        current = set()
        for unit in snapshot["units"]:
            if unit[0] not in component:
                continue
            enlarged = current | set(unit)
            middle_count = sum(
                snapshot["nodes"][key]["kind"] != "vehicle" for key in enlarged
            )
            if current and (len(enlarged) > budget or middle_count > middle_budget):
                pages.append(current)
                current = set()
            current.update(unit)
        if current:
            pages.append(current)
    return pages or [set()]


@lru_cache(maxsize=64)
def graph_positions(nodes, edges):
    """Disposition stable, par niveaux pour les chaînes véhicule-espace-véhicule."""
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    if not graph:
        return {}
    if nx.is_directed_acyclic_graph(graph):
        layers = [sorted(layer) for layer in nx.topological_generations(graph)]
        return {
            key: (index / max(1, len(layers) - 1), (row + 1) / (len(layer) + 1))
            for index, layer in enumerate(layers)
            for row, key in enumerate(layer)
        }
    # Un cycle garde simplement sa forme dans le dessin ; aucun critère scientifique
    # n'est tiré de cette disposition circulaire.
    count = len(nodes)
    return {
        key: (
            0.5 + 0.45 * math.cos(2 * math.pi * i / count),
            0.5 + 0.45 * math.sin(2 * math.pi * i / count),
        )
        for i, key in enumerate(sorted(nodes))
    }
