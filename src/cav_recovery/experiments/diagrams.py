"""Graphes d'inspection affichés à part ; aucun résultat n'est renvoyé aux modèles."""

import math

from .presentation import french, label, wrapped

INK = (225, 234, 246)
MUTED = (163, 183, 205)
BACKGROUND = (25, 35, 50)
COLORS = {
    "NO_ALERT": (67, 190, 130),
    "SUSPECTED": (246, 183, 58),
    "CONFIRMED_OBSTRUCTION": (232, 82, 94),
    "UNKNOWN": (154, 164, 180),
}


def text_lines(view, message, position, width, *, color=INK, small=False):
    font = view.small_font if small else view.panel_font
    x, y = position
    for line in wrapped(message, font, max(1, width)):
        view.screen_canvas.blit(font.render(line, True, color), (x, y))
        y += font.get_linesize() + 2
    return y


def heading(view, title, explanation):
    """Réserve la place du titre et de la portée avant de dessiner les données."""
    import pygame

    rect = view.graph_rect
    pygame.draw.rect(view.screen_canvas, BACKGROUND, rect)
    y = text_lines(
        view, title, (rect.x + 16, rect.y + 12), rect.width - 32, color=(131, 214, 226)
    )
    y = text_lines(
        view,
        explanation,
        (rect.x + 16, y + 4),
        rect.width - 32,
        small=True,
        color=MUTED,
    )
    return pygame.Rect(
        rect.x + 24, y + 14, rect.width - 48, max(30, rect.bottom - y - 46)
    )


def draw_network(view, *, communication=False):
    """Dessine le graphe typé dans le panneau compagnon, sans couvrir le trafic."""
    import pygame
    from .info_graph import (
        physical_snapshot,
        detector_relations,
        message_snapshot,
        graph_pages,
        graph_positions,
    )

    if communication:
        pairs = set(view._useful_links(view.frame["graphs"].get(view.selected, {})))
        view.displayed_links = sorted(pairs)
        snapshot = message_snapshot(
            view.recent_messages.values(),
            pairs,
            view.message_aliases,
            useful=view.v2v_mode == 1,
        )
        title = "INFO V2V · " + label(view.selected or "—")
        note = "Ronds : CAV · Rectangles M : messages livrés. U : diagnostics utiles / tous les canaux."
    else:
        if view.active_view == "d01":
            diagnostic = view.frame["diagnostics"].get(view.selected, {})
            rows = detector_relations(diagnostic) if diagnostic else []
            title = "INFO D01 · " + label(view.selected or "—")
            note = "Ronds : CAV · Rectangles R : espaces occupés observés. Le statut temporel reste distinct."
        else:
            rows = [
                edge
                for edge in view.frame["graphs"].get(view.selected, {}).get("edges", [])
                if edge["state"] == "CURRENT"
            ]
            title = "INFO C-RDG · " + label(view.selected or "—")
            note = "Ronds : CAV · Rectangles R : espaces requis. Demandeur vers ressource vers occupant."
        snapshot = physical_snapshot(rows, view.resource_aliases)
    view.info_snapshot = snapshot
    plot = heading(view, title, note)
    budget = 18 if plot.height > 250 else 12
    pages = graph_pages(
        snapshot,
        view.selected,
        budget,
        middle_budget=max(1, min(3, (plot.height - 30) // 48)),
    )
    view.diagram_page %= len(pages)
    visible = pages[view.diagram_page]
    edges = [
        edge
        for edge in snapshot["edges"]
        if edge["source"] in visible and edge["target"] in visible
    ]
    view.visible_info_edges = edges
    if len(pages) > 1:
        view._button("diagram_prev", "<", (plot.right - 72, plot.y, 30, 24))
        view._button("diagram_next", ">", (plot.right - 34, plot.y, 30, 24))
        text_lines(
            view,
            f"Voisinage {view.diagram_page + 1}/{len(pages)}",
            plot.topleft,
            plot.width - 80,
            small=True,
        )
        plot.y += 30
        plot.height = max(30, plot.height - 30)
    layout = graph_positions(
        tuple(sorted(visible)),
        tuple(sorted((edge["source"], edge["target"]) for edge in edges)),
    )
    positions = {
        key: (
            plot.left + 28 + point[0] * max(1, plot.width - 56),
            plot.top + 14 + point[1] * max(1, plot.height - 28),
        )
        for key, point in layout.items()
    }
    for edge in edges:
        start, end = positions[edge["source"]], positions[edge["target"]]
        distance = max(1, math.dist(start, end))
        ux, uy = (end[0] - start[0]) / distance, (end[1] - start[1]) / distance
        source_radius = (
            30 if snapshot["nodes"][edge["source"]]["kind"] != "vehicle" else 19
        )
        target_radius = (
            30 if snapshot["nodes"][edge["target"]]["kind"] != "vehicle" else 19
        )
        color = {
            "CONFIRMED": COLORS["CONFIRMED_OBSTRUCTION"],
            "SUSPECTED": COLORS["SUSPECTED"],
            "MESSAGE": (91, 165, 242),
        }.get(edge["level"], (101, 202, 189))
        view._arrow(
            (start[0] + ux * source_radius, start[1] + uy * source_radius),
            (end[0] - ux * target_radius, end[1] - uy * target_radius),
            color,
        )
    view.graph_nodes = {}
    for key, point in positions.items():
        node = snapshot["nodes"][key]
        selected = key == view.inspected_node or key == "vehicle:" + str(view.selected)
        if node["kind"] == "vehicle":
            color = (255, 221, 139) if selected else (201, 215, 232)
            pygame.draw.circle(view.screen_canvas, color, point, 17)
            rect = pygame.Rect(point[0] - 19, point[1] - 19, 38, 38)
        else:
            color = (147, 204, 222) if node["kind"] == "message" else (167, 216, 197)
            rect = pygame.Rect(point[0] - 26, point[1] - 15, 52, 30)
            pygame.draw.rect(view.screen_canvas, color, rect, border_radius=3)
        if selected:
            pygame.draw.rect(
                view.screen_canvas, INK, rect.inflate(5, 5), 1, border_radius=4
            )
        image = view.small_font.render(node["label"], True, (16, 27, 39))
        view.screen_canvas.blit(image, image.get_rect(center=point))
        view.graph_nodes[key] = rect
    count = f"{len(visible)}/{len(snapshot['nodes'])} nœuds · {len(edges)}/{len(snapshot['edges'])} liens"
    if not snapshot["nodes"]:
        text_lines(
            view,
            "Aucune preuve actuelle à représenter.",
            plot.topleft,
            plot.width,
            small=True,
        )
    text_lines(
        view,
        count,
        (view.graph_rect.x + 12, view.graph_rect.bottom - 23),
        view.graph_rect.width - 24,
        small=True,
        color=MUTED,
    )


def draw_d01(view):
    """Trace uniquement les sorties et mesures déjà disponibles pour l'ego sélectionné."""
    import pygame

    plot = heading(
        view,
        "D01 — historique de " + label(view.selected or "—"),
        "Passé observé uniquement. Le statut et les mesures sont affichés "
        "sans recalcul du détecteur ni modification de ses seuils.",
    )
    rows = list(view.d01_history.get(view.selected, []))
    if not rows:
        text_lines(view, "Aucune observation disponible.", plot.topleft, plot.width)
        return
    left, right = plot.left + 44, plot.right - 14
    begin, end = rows[0]["at_s"], max(rows[0]["at_s"] + 0.2, rows[-1]["at_s"])

    def x(at):
        return left + (at - begin) / (end - begin) * (right - left)

    y = text_lines(view, "Statut D01 au cours du temps", plot.topleft, plot.width)
    runs = []
    for row in rows:
        if not runs or runs[-1]["status"] != row["status"]:
            runs.append(row)
    for index, row in enumerate(runs):
        next_at = runs[index + 1]["at_s"] if index + 1 < len(runs) else end
        pygame.draw.rect(
            view.screen_canvas,
            COLORS[row["status"]],
            (
                x(row["at_s"]),
                y + 4,
                max(1, math.ceil(x(next_at)) - math.floor(x(row["at_s"]))),
                20,
            ),
        )
    y += 34
    for status in COLORS:
        pygame.draw.circle(view.screen_canvas, COLORS[status], (plot.x + 5, y + 7), 4)
        y = text_lines(
            view, french(status), (plot.x + 16, y), plot.width - 16, small=True
        )
    chart = pygame.Rect(
        left, y + 24, max(20, right - left), max(40, plot.bottom - y - 52)
    )
    maximum = max(1, max(max(abs(row["speed"]), row["requested"]) for row in rows))
    minimum = min(0, min(row["speed"] for row in rows))
    span = maximum - minimum
    for value in sorted({round(minimum, 1), 0, round(maximum, 1)}):
        at_y = chart.bottom - (value - minimum) / span * chart.height
        pygame.draw.line(
            view.screen_canvas, (68, 84, 105), (chart.left, at_y), (chart.right, at_y)
        )
        text_lines(view, f"{value:.1f}", (plot.x, at_y - 7), 40, small=True)
    for field, color in (("requested", (241, 194, 90)), ("speed", (103, 214, 232))):
        points = [
            (
                x(row["at_s"]),
                chart.bottom - (row[field] - minimum) / span * chart.height,
            )
            for row in rows
        ]
        if len(points) >= 2:
            pygame.draw.lines(view.screen_canvas, color, False, points, 2)
    text_lines(view, f"{begin:.1f} s", (chart.left, chart.bottom + 4), 100, small=True)
    text_lines(
        view, f"{end:.1f} s", (chart.right - 60, chart.bottom + 4), 60, small=True
    )
    text_lines(
        view,
        "m/s · cyan : vitesse mesurée · jaune : vitesse demandée",
        (plot.x, chart.top - 23),
        plot.width,
        small=True,
        color=MUTED,
    )
