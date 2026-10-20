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
            ("resources", "R · Ressources"),
        )
    ):
        view._button(
            key, title, (12 + index * 116, y, 108, 28), view.active_view == key
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
    draw_panel(view)
    pygame.draw.rect(screen, BACKGROUND, view.footer_rect)
    notice = (
        view.notice
        if time.monotonic() < view.notice_until
        else (
            "Clic / Tab : sélectionner · T : trafic · G / V / D / R : vues · Échap : fermer"
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

    panel = view.panel_rect
    if panel.width == 0:
        view.max_scroll = 0
        return
    pygame.draw.rect(view.screen_canvas, BACKGROUND, panel)
    viewport = panel.inflate(-20, -40)
    viewport.y = panel.y + 8
    rows = []
    y = 0
    step = view.panel_font.get_linesize() + 2
    for message, title in view._panel_lines():
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

    box = view.scene_rect.inflate(-24, -16)
    pygame.draw.rect(view.screen_canvas, BACKGROUND, box, border_radius=8)
    lines = [
        "LÉGENDE ET COMMANDES — F1 pour fermer",
        "T : trafic sans flèches de diagnostic. Clic sur une voiture ou Tab : sélectionner.",
        "G : graphe C-RDG local, demandeur vers bloqueur, dans sa propre vue.",
        "V : graphe des livraisons V2V. U : reçus utiles / tous les liens de l'ego.",
        "D : historique D01 et vitesses propres. R : géométrie des ressources étayées.",
        "Une seconde pression sur G, V, D ou R revient au trafic.",
        "Vert : pas d'alerte ; jaune : suspicion ; rouge : confirmé ; gris : abstention.",
        "Espace : pause / reprise. + / − : cadence cible. S : capture PNG et provenance.",
        "P : afficher / masquer les informations. Molette sur la carte : zoom.",
        "Molette sur le panneau ou Page haut/bas : défiler. gauche / droite : pages des graphes.",
        "I : référence globale d'inspection facultative ; jamais une entrée des modèles.",
        "Échap : fermeture de la vue ; le calcul termine sans fenêtre.",
    ]
    y = box.y + 10
    for line in lines:
        y = text_lines(
            view, line, (box.x + 10, y), box.width - 20, small=box.height < 430
        )
        y += 5
