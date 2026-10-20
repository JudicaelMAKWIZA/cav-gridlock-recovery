"""Navigation et panneau d'information adaptés à la fenêtre courante."""

import time

from .diagrams import text_lines
from .presentation import wrapped

BACKGROUND = (23, 32, 48)


def draw_chrome(view):
    import pygame

    screen = view.screen_canvas
    width, height = screen.get_size()
    pygame.draw.rect(screen, BACKGROUND, (0, 0, width, view.scene_rect.top))
    text_lines(view, view.header_text(), (12, 7), width - 24)
    y = view.navigation_y
    for index, (key, title) in enumerate(
        (
            ("traffic", "Trafic"),
            ("graph", "G · C-RDG"),
            ("v2v", "V · V2V"),
            ("d01", "D · D01"),
            ("resources", "R · Détails"),
        )
    ):
        view._button(
            key,
            title,
            (12 + index * 116, y, 108, 28),
            view.show_resources if key == "resources" else view.active_view == key,
        )
    y += 36
    view._button(
        "pause",
        "Reprendre · Espace" if view.paused else "Pause · Espace",
        (12, y, 148, 28),
        view.paused,
    )
    view._button("slower", "−", (168, y, 32, 28))
    screen.blit(
        view.small_font.render(f"×{view.playback_rate:g}", True, (240, 246, 253)),
        (208, y + 7),
    )
    view._button("faster", "+", (250, y, 32, 28))
    view._button("capture", "S · Capture", (292, y, 96, 28))
    view._button("details", "P · Infos", (396, y, 88, 28), view.show_details)
    view._button("help", "? / F1", (492, y, 72, 28), view.show_help)
    if view.active_view == "v2v":
        view._button("v2v_filter", "U · Filtre", (572, y, 88, 28), view.v2v_mode == 1)
    elif view.active_view in ("d01", "history"):
        view._button(
            "history",
            "H · Historique",
            (572, y, 110, 28),
            view.active_view == "history",
        )
    draw_panel(view)
    pygame.draw.rect(screen, BACKGROUND, view.footer_rect)
    notice = (
        view.notice
        if time.monotonic() < view.notice_until
        else (
            "Le trafic reste visible · G / V / D : informations · clic sur un nœud : détails"
        )
    )
    text_lines(
        view,
        notice,
        (12, view.footer_rect.top + 4),
        width - 24,
        color=(246, 220, 152),
        small=True,
    )
    suffix = " (modifications locales)" if view.stamp["dirty"] else ""
    text_lines(
        view,
        "Commit : " + view.stamp["commit"] + suffix,
        (12, height - 20),
        width - 24,
        color=(178, 197, 215),
        small=True,
    )
    if view.show_help:
        draw_legend(view)


def draw_panel(view):
    import pygame

    panel = view.details_rect
    if panel.width == 0:
        view.max_scroll = 0
        return
    pygame.draw.rect(view.screen_canvas, BACKGROUND, panel)
    viewport = panel.inflate(-20, -40)
    viewport.y = panel.y + 8
    rows = []
    y = 0
    step = view.panel_font.get_linesize() + 2
    for message, title in view._information_lines():
        if title:
            y += 10
        for line in wrapped(message, view.panel_font, viewport.width - 4):
            rows.append((y, line, title))
            y += step
    view.max_scroll = max(0, y - viewport.height)
    view.scroll = min(view.scroll, view.max_scroll)
    for y, line, title in rows:
        top = viewport.y + y - view.scroll
        if top < viewport.top or top + step > viewport.bottom:
            continue
        color = (131, 214, 226) if title else (229, 235, 244)
        view.screen_canvas.blit(
            view.panel_font.render(line, True, color), (viewport.x, top)
        )
    if view.max_scroll:
        text_lines(
            view,
            "Molette / Page haut/bas : défiler",
            (panel.x + 10, panel.bottom - 23),
            panel.width - 20,
            color=(168, 185, 204),
            small=True,
        )
        rail = pygame.Rect(panel.right - 6, viewport.top, 3, viewport.height)
        pygame.draw.rect(view.screen_canvas, (62, 79, 100), rail)
        handle = max(18, viewport.height**2 // max(1, y + step))
        at = rail.y + view.scroll / view.max_scroll * (rail.height - handle)
        pygame.draw.rect(view.screen_canvas, (131, 214, 226), (rail.x, at, 3, handle))


def draw_legend(view):
    import pygame

    box = view.panel_rect.inflate(-16, -8)
    pygame.draw.rect(view.screen_canvas, BACKGROUND, box, border_radius=8)
    lines = [
        "LÉGENDE ET COMMANDES — F1 pour fermer",
        "Le trafic reste toujours visible. Clic sur une voiture ou Tab : sélectionner.",
        "G : INFO C-RDG. Cercle CAV, rectangle R espace requis, puis CAV occupant.",
        "Liens C-RDG : rouge confirmé, jaune suspecté, turquoise occupation observée.",
        "V : INFO V2V. Cercle CAV, rectangle M message, puis CAV destinataire.",
        "D : INFO D01 et occupations observées. H : historique du statut et des vitesses.",
        "R ou clic sur un nœud : détails. G / V / D une seconde fois : fermer le panneau.",
        "Vert : pas d'alerte ; jaune : suspicion ; rouge : confirmé ; gris : abstention.",
        "Espace : pause / reprise. + / − : cadence cible. S : capture PNG et provenance.",
        "P : afficher / masquer les détails. Molette sur la carte : zoom.",
        "Molette sur le panneau ou Page haut/bas : défiler. Gauche / droite : voisinages.",
        "I : référence globale d'inspection facultative ; jamais une entrée des modèles.",
        "Échap : fermeture de la vue ; le calcul termine sans fenêtre.",
    ]
    rows = []
    for line in lines:
        rows.extend(wrapped(line, view.small_font, box.width - 20))
        rows.append("")
    step = view.small_font.get_linesize() + 2
    view.max_scroll = max(0, len(rows) * step - box.height + 20)
    view.scroll = min(view.scroll, view.max_scroll)
    for index, line in enumerate(rows):
        y = box.y + 10 + index * step - view.scroll
        if box.top <= y and y + step < box.bottom:
            view.screen_canvas.blit(
                view.small_font.render(line, True, (234, 241, 249)), (box.x + 10, y)
            )
