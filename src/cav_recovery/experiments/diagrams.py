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

    rect = view.scene_rect
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
    """Une flèche représente un lien existant, sans en déduire un nouveau lien causal."""
    import pygame

    graph = view.frame["graphs"].get(view.selected, {})
    if communication:
        pairs = view._useful_links(graph)
        view.displayed_links = pairs
        edges = [(a, b, (91, 165, 242)) for a, b in pairs]
        title = "V2V — échanges liés à " + label(view.selected or "—")
        note = (
            "Flèche : émetteur vers destinataire. Livraisons réelles des 0,6 dernières "
            "secondes ; canaux regroupés. U : reçus utiles / tous les liens de l'ego."
        )
    else:
        current = [
            edge for edge in graph.get("edges", []) if edge["state"] == "CURRENT"
        ]
        edges = [
            (
                edge["requester"],
                edge["blocker"],
                COLORS[
                    "CONFIRMED_OBSTRUCTION"
                    if edge["level"] == "CONFIRMED"
                    else "SUSPECTED"
                ],
            )
            for edge in current
        ]
        title = "C-RDG — vue locale reçue par " + label(view.selected or "—")
        note = (
            "Flèche : demandeur vers bloqueur. Rouge : relation confirmée ; jaune : "
            "relation suspectée. Le graphe partiel ne représente pas l'oracle global."
        )
    plot = heading(view, title, note)
    keys = {key for a, b, _ in edges for key in (a, b)}
    keys.discard(view.selected)
    ordered = sorted(keys, key=lambda key: (len(label(key)), label(key)))
    page_size = max(
        7,
        min(
            29,
            int(2 * math.pi * max(30, min(plot.width, plot.height) / 2 - 32) / 42) - 1,
        ),
    )
    pages = max(1, math.ceil(len(ordered) / page_size))
    view.diagram_page %= pages
    visible = ordered[
        view.diagram_page * page_size : (view.diagram_page + 1) * page_size
    ]
    if view.selected:
        visible.insert(0, view.selected)
    if pages > 1:
        text_lines(
            view,
            f"Page {view.diagram_page + 1}/{pages} · gauche / droite : changer de page",
            (plot.x, plot.y),
            plot.width,
            small=True,
        )
        plot.y += 26
        plot.height = max(30, plot.height - 26)
        view._button("diagram_prev", "< Préc.", (plot.right - 164, plot.y, 76, 26))
        view._button("diagram_next", "Suiv. >", (plot.right - 80, plot.y, 76, 26))
        plot.y += 30
        plot.height = max(30, plot.height - 30)
    positions = {}
    if communication and visible:
        positions[visible[0]] = plot.center
        ring = visible[1:]
    else:
        ring = visible
    for index, key in enumerate(ring):
        angle = -math.pi / 2 + index * 2 * math.pi / max(1, len(ring))
        positions[key] = (
            plot.centerx + max(12, plot.width / 2 - 32) * math.cos(angle),
            plot.centery + max(12, plot.height / 2 - 32) * math.sin(angle),
        )
    if len(visible) == 1:
        positions[visible[0]] = plot.center
    drawn = 0
    for a, b, color in edges:
        if a not in positions or b not in positions or a == b:
            continue
        start, end = positions[a], positions[b]
        distance = max(1, math.dist(start, end))
        ux, uy = (end[0] - start[0]) / distance, (end[1] - start[1]) / distance
        # Décaler les deux directions pour garder les flèches réciproques distinctes.
        view._arrow(
            (start[0] + ux * 21 - uy * 3, start[1] + uy * 21 + ux * 3),
            (end[0] - ux * 21 - uy * 3, end[1] - uy * 21 + ux * 3),
            color,
        )
        drawn += 1
    view.graph_nodes = {}
    for key, point in positions.items():
        # La couleur d'un tiers ne doit pas suggérer que son diagnostic global
        # ferait partie des connaissances reçues par l'ego.
        pygame.draw.circle(view.screen_canvas, (183, 204, 226), point, 19)
        if key == view.selected:
            pygame.draw.circle(view.screen_canvas, INK, point, 23, 2)
        image = view.small_font.render(label(key), True, (16, 27, 39))
        view.screen_canvas.blit(image, image.get_rect(center=point))
        view.graph_nodes[key] = pygame.Rect(point[0] - 24, point[1] - 24, 48, 48)
    message = (
        f"{drawn}/{len(edges)} liens représentés · clic sur un nœud : sélectionner"
    )
    if not edges:
        message = "Aucun lien actuel étayé dans cette vue."
    text_lines(
        view,
        message,
        (view.scene_rect.x + 16, view.scene_rect.bottom - 26),
        view.scene_rect.width - 32,
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


def draw_resources(view, polygons):
    """Projette les empreintes de preuve dans une vue géométrique dédiée."""
    import pygame

    plot = heading(
        view,
        "Ressources — preuves reçues par " + label(view.selected or "—"),
        "Contours des espaces occupés étayés ; jaune : suspicion, rouge : "
        "confirmation. Les identifiants désignent demandeur vers bloqueur.",
    )
    edges = [
        edge
        for edge in view.frame["graphs"].get(view.selected, {}).get("edges", [])
        if edge["state"] == "CURRENT"
    ]
    shapes = [
        (edge, polygon) for edge in edges for polygon in polygons(edge["resource_wkt"])
    ]
    if not shapes:
        text_lines(
            view,
            "Aucune ressource occupée étayée dans cette vue.",
            plot.topleft,
            plot.width,
        )
        return
    points = [point for _, shape in shapes for point in shape]
    xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
    ymin, ymax = min(p[1] for p in points), max(p[1] for p in points)
    scale = min(
        (plot.width - 40) / max(1, xmax - xmin),
        (plot.height - 40) / max(1, ymax - ymin),
    )

    def pixel(point):
        return (
            plot.centerx + (point[0] - (xmin + xmax) / 2) * scale,
            plot.centery - (point[1] - (ymin + ymax) / 2) * scale,
        )

    labels = []
    for edge, shape in shapes:
        color = COLORS[
            "CONFIRMED_OBSTRUCTION" if edge["level"] == "CONFIRMED" else "SUSPECTED"
        ]
        pygame.draw.polygon(
            view.screen_canvas, color, [pixel(point) for point in shape], 2
        )
        name = label(edge["requester"]) + " vers " + label(edge["blocker"])
        point = pixel(shape[0])
        image = view.small_font.render(name, True, INK, BACKGROUND)
        rect = image.get_rect(topleft=point).clamp(plot)
        while (
            any(rect.colliderect(other) for other in labels)
            and rect.bottom + 20 <= plot.bottom
        ):
            rect.y += 20
        labels.append(rect)
        view.screen_canvas.blit(image, rect)
