"""Dimensions de la fenêtre dans la zone réellement utilisable du bureau Windows."""

import sys


def available_size():
    """Mesure la zone de travail, barre des tâches et mise à l'échelle comprises.

    Cette lecture appartient à l'application ; elle ne change ni les réglages DPI
    de Windows ni ceux des autres programmes.
    """
    import pygame

    pygame.display.init()
    width, height = pygame.display.get_desktop_sizes()[0]
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        area = wintypes.RECT()
        if ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(area), 0):
            width = min(width, area.right - area.left)
            height = min(height, area.bottom - area.top)
    # Garder de la place pour le cadre et la barre de titre de la fenêtre.
    return max(320, min(1440, width - 32)), max(300, min(900, height - 64))


def rectangles(size, *, title_lines=1, details=True, diagnostic=False):
    """Répartit la place disponible sans coordonnées liées à un écran particulier."""
    import pygame

    width, height = size
    header = 86 + title_lines * 22
    footer = 64
    sidebar = (
        width // 2
        if diagnostic
        else (min(300, max(240, width // 4)) if details and width >= 720 else 0)
    )
    content = pygame.Rect(0, header, width - sidebar, max(1, height - header - footer))
    panel = pygame.Rect(content.right, header, sidebar, content.height)
    return content, panel, pygame.Rect(0, height - footer, width, footer)
