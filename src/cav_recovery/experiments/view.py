"""Inspection 2D interactive ; les commandes de la vue n'agissent pas sur les CAV."""

from functools import lru_cache
import math
from pathlib import Path
import time

from shapely import wkt

from ..metadrive.view import TopDownView
from .presentation import french, label, save_capture, source_stamp, wrapped

COLORS = {
    "NO_ALERT": (67, 190, 130),
    "SUSPECTED": (246, 183, 58),
    "CONFIRMED_OBSTRUCTION": (232, 82, 94),
    "UNKNOWN": (154, 164, 180),
}
RATES = (0.25, 0.5, 1, 2, 4, 8)
V2V_MODES = ("masqué", "utile reçu", "tous liés à l'ego")


@lru_cache(maxsize=256)
def resource_polygons(text):
    """Une preuve affichée en pause ne nécessite pas de relire son WKT."""
    geometry = wkt.loads(text)
    parts = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
    return [list(part.exterior.coords) for part in parts if part.geom_type == "Polygon"]


class ExperimentView(TopDownView):
    """Vue locale reçue, indépendante de la référence globale d'inspection."""

    def __init__(self, *, window=True, output=None, context=None):
        import pygame

        self.frame = None
        self.selected = None
        self.manual_selection = False
        self.v2v_mode = 0
        self.show_graph = True
        self.show_resources = True
        self.show_reference = False
        self.show_help = False
        self.paused = False
        self.rate_index = 2
        self.output = Path(output) if output is not None else None
        self.context = dict(context or {})
        self.stamp = source_stamp()
        self.panel_font = None
        self.small_font = None
        self.buttons = {}
        self.scroll = self.max_scroll = 0
        self.notice = "Cliquez sur un véhicule ou utilisez Tab pour l'inspecter."
        self.notice_until = time.monotonic() + 8
        self.capture_requested = False
        self.capture_count = 0
        self.last_capture = None
        self.dirty = True
        self.recent_links = {}
        self.displayed_links = []
        self.scene_rect = pygame.Rect(0, 56, 1080, 790)
        super().__init__(
            window=window,
            camera_position=(20, 0),
            scaling=4.5,
            screen_size=(1440, 900),
            film_size=(1600, 1600),
        )
        self.clock = pygame.time.Clock()
        self.deadline = time.monotonic()
        if not self.no_window:
            pygame.display.set_caption("CAV Recovery — inspection MetaDrive")
            pygame.key.set_repeat()

    @property
    def show_v2v(self):
        return self.v2v_mode != 0

    @property
    def frames_per_second(self):
        return 5 * RATES[self.rate_index]

    def _pixel(self, point):
        cx, cy = self._frame_canvas.pos2pix(*self.position)
        px, py = self._frame_canvas.pos2pix(*point)
        return px - cx + self.scene_rect.centerx, py - cy + self.scene_rect.centery

    def _arrow(self, start, end, color, width=2):
        import pygame

        pygame.draw.line(self.screen_canvas, color, start, end, width)
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        wings = [
            (end[0] - 8 * math.cos(angle + side), end[1] - 8 * math.sin(angle + side))
            for side in (-0.45, 0.45)
        ]
        pygame.draw.polygon(self.screen_canvas, color, [end, *wings])

    def _remember_deliveries(self):
        """Conserve les livraisons réelles au plus 0,6 s pour leur seule représentation."""
        now = self.frame["at_s"]
        self.recent_links = {
            pair: at for pair, at in self.recent_links.items() if now - at <= 0.6 + 1e-8
        }
        for event in self.frame["deliveries"]:
            if event["event"] == "emitted":
                sender = event["message"]["sender_id"]
                for recipient in event["delivered_to"]:
                    self.recent_links[sender, recipient] = now

    def _useful_links(self, graph):
        if not self.show_v2v:
            return []
        relevant = {
            edge["blocker"]
            for edge in graph.get("edges", [])
            if edge["state"] == "CURRENT"
        }
        relevant.update(
            key
            for group in graph.get("groups", [])
            for key in group["affected_requesters"]
        )
        links = []
        for sender, recipient in sorted(self.recent_links):
            if self.selected not in (sender, recipient):
                continue
            if self.v2v_mode == 1 and not (
                recipient == self.selected and sender in relevant
            ):
                continue
            if sender in self.engine.agents and recipient in self.engine.agents:
                links.append((sender, recipient))
        return links[:4] if self.v2v_mode == 1 else links

    def _draw(self, *args, **kwargs):
        if self.frame:
            for key, diagnostic in self.frame["diagnostics"].items():
                self.labels[key] = label(key)
                self.colors[key] = COLORS[diagnostic["status"]]
        super()._draw(*args, **kwargs)
        if not self.frame:
            return
        import pygame

        self.screen_canvas.set_clip(self.scene_rect)
        graph = self.frame["graphs"].get(self.selected, {})
        packet = self.frame["measurements"].get(self.selected, {})
        plan = packet.get("own_controls", {}).get("planned_path_xyh_m", [])
        if len(plan) >= 2:
            pygame.draw.lines(
                self.screen_canvas,
                (116, 222, 238),
                False,
                [self._pixel(pose[:2]) for pose in plan],
                2,
            )
        self.displayed_links = self._useful_links(graph)
        for sender, recipient in self.displayed_links:
            self._arrow(
                self._pixel(self.engine.agents[sender].position),
                self._pixel(self.engine.agents[recipient].position),
                (67, 149, 245),
                1,
            )
        for edge in graph.get("edges", []):
            if edge["state"] != "CURRENT":
                continue
            color = COLORS[
                "CONFIRMED_OBSTRUCTION" if edge["level"] == "CONFIRMED" else "SUSPECTED"
            ]
            if self.show_resources:
                for polygon in resource_polygons(edge["resource_wkt"]):
                    pygame.draw.polygon(
                        self.screen_canvas,
                        color,
                        [self._pixel(point) for point in polygon],
                        2,
                    )
            if (
                self.show_graph
                and edge["requester"] in self.engine.agents
                and edge["blocker"] in self.engine.agents
            ):
                self._arrow(
                    self._pixel(self.engine.agents[edge["requester"]].position),
                    self._pixel(self.engine.agents[edge["blocker"]].position),
                    color,
                )
        if self.selected in self.engine.agents:
            pygame.draw.circle(
                self.screen_canvas,
                (242, 247, 255),
                self._pixel(self.engine.agents[self.selected].position),
                15,
                2,
            )
        self.screen_canvas.set_clip(None)

    def _tell(self, message):
        self.notice = message
        self.notice_until = time.monotonic() + 5
        self.dirty = True

    def _panel_lines(self):
        diagnostic = self.frame["diagnostics"].get(self.selected)
        graph = self.frame["graphs"].get(self.selected, {})
        lines = [("VÉHICULE " + label(self.selected or "—"), True)]
        if diagnostic:
            coverage = diagnostic["coverage"]
            destination = diagnostic["destination"]
            if destination.startswith("a") and destination.endswith("_sink"):
                destination = "Sortie " + destination[1:-5]
            lines += [
                (french(diagnostic["status"]), True),
                (french(diagnostic["reason"]), False),
                (f"Destination : {destination}", False),
                (
                    f"Vitesse : {diagnostic['own_state']['signed_speed_mps']:.2f} m/s",
                    False,
                ),
                (
                    "Demandée / acceptée : "
                    f"{diagnostic['own_controls']['requested_speed_mps']:.1f} / "
                    f"{diagnostic['own_controls']['accepted_speed_mps']:.1f} m/s",
                    False,
                ),
            ]
            for field, title in (
                ("series_start_s", "Preuves depuis"),
                ("effect_start_s", "Effet depuis"),
            ):
                at = coverage[field]
                lines.append(
                    (
                        f"{title} : {at:.1f} s"
                        if at is not None
                        else f"{title} : non établi",
                        False,
                    )
                )
            lines.append(
                (f"Rendez-vous étayés : {len(coverage['proof_times_s'])}", False)
            )
            if diagnostic["events"]:
                event = diagnostic["events"][-1]
                lines.append(
                    (
                        f"{french(event['kind'])} à {event['available_at_s']:.1f} s",
                        False,
                    )
                )
        affected = {
            key
            for group in graph.get("groups", [])
            for key in group["affected_requesters"]
        }
        current = sum(edge["state"] == "CURRENT" for edge in graph.get("edges", []))
        lines += [
            ("C-RDG LOCAL REÇU", True),
            (
                f"Groupes : {len(graph.get('groups', []))} | affectés : {len(affected)}",
                False,
            ),
            (f"Relations courantes : {current}", False),
        ]
        for group in graph.get("groups", []):
            lines.append(
                (
                    "Affectés : "
                    + ", ".join(label(key) for key in group["affected_requesters"]),
                    False,
                )
            )
        costs = self.frame["costs_ms"]
        elapsed = sum(
            costs.get(key, 0)
            for key in ("adaptation", "detector", "diagnostic_communication", "crdg")
        )
        lines += [
            (f"D01 + rapports + graphe : {elapsed:.1f} ms", False),
            ("COMMUNICATION V2V", True),
            (
                f"Traits : {V2V_MODES[self.v2v_mode]} ({len(self.displayed_links)})",
                False,
            ),
            ("Livraisons des 0,6 dernières secondes ; canaux regroupés.", False),
        ]
        inbox = self.frame["measurements"].get(self.selected, {}).get("v2v")
        received = (inbox or {}).get("received", [])
        lines.append((f"Messages embarqués reçus : {len(received)}", False))
        for entry in received[:3]:
            message = entry["message"]
            lines.append(
                (
                    f"{label(message['sender_id'])} : âge "
                    f"{self.frame['at_s'] - message['observed_at_s']:.1f} s",
                    False,
                )
            )
        independent = self.frame.get("independent_reference")
        lines.append(("RÉFÉRENCE GLOBALE D'INSPECTION", True))
        if self.show_reference and independent:
            lines.append(
                (
                    f"Affectés (préfixe) : {len(independent['affected_requesters'])}",
                    False,
                )
            )
        else:
            lines.append(
                ("Désactivée pour la réactivité. I : activer (coût CPU).", False)
            )
        lines.append(("Jamais transmise aux détecteurs ni au graphe local.", False))
        return lines

    def _button(self, key, title, rectangle, active=False):
        import pygame

        rect = pygame.Rect(rectangle)
        self.buttons[key] = rect
        pygame.draw.rect(
            self.screen_canvas,
            (43, 91, 111) if active else (47, 59, 76),
            rect,
            border_radius=5,
        )
        image = self.small_font.render(title, True, (240, 246, 253))
        self.screen_canvas.blit(image, image.get_rect(center=rect.center))

    def _add_text(self, text):
        if not self.frame:
            return
        import pygame

        if self.panel_font is None:
            self.panel_font = pygame.font.Font(None, 23)
            self.small_font = pygame.font.Font(None, 20)
        screen = self.screen_canvas
        pygame.draw.rect(screen, (23, 32, 48), (1080, 0, 360, 900))
        pygame.draw.rect(screen, (23, 32, 48), (0, 0, 1080, 56))
        pygame.draw.rect(screen, (23, 32, 48), (0, 846, 1080, 54))
        scenario = self.context.get("scenario", "scénario non renseigné")
        seed = self.context.get("seed", "—")
        heading = (
            f"{scenario} | graine {seed} | t = {self.frame['at_s']:.1f} s | "
            f"{len(self.frame['diagnostics'])} CAV actifs"
        )
        screen.blit(self.panel_font.render(heading, True, (240, 246, 253)), (16, 18))
        state = "PAUSE" if self.paused else "EN COURS"
        self._button("pause", state + " · Espace", (1092, 12, 224, 32), self.paused)
        self._button("help", "? / F1", (1324, 12, 104, 32), self.show_help)
        self._button(
            "v2v",
            "V · V2V : " + V2V_MODES[self.v2v_mode],
            (1092, 52, 336, 30),
            self.show_v2v,
        )
        self._button(
            "graph",
            "G · C-RDG : " + ("visible" if self.show_graph else "masqué"),
            (1092, 88, 164, 30),
            self.show_graph,
        )
        self._button(
            "resources",
            "R · Ressources : " + ("oui" if self.show_resources else "non"),
            (1264, 88, 164, 30),
            self.show_resources,
        )
        self._button("slower", "−", (1092, 126, 42, 30))
        self._button("faster", "+", (1222, 126, 42, 30))
        screen.blit(
            self.small_font.render(
                f"×{RATES[self.rate_index]:g}", True, (240, 246, 253)
            ),
            (1154, 133),
        )
        self._button("capture", "S · Capture", (1272, 126, 156, 30))
        panel = pygame.Rect(1092, 170, 336, 684)
        rows = []
        y = 0
        for message, title in self._panel_lines():
            if title:
                y += 12
            for line in wrapped(message, self.panel_font, panel.width - 10):
                rows.append((y, line, title))
                y += 23
        self.max_scroll = max(0, y - panel.height)
        self.scroll = min(self.scroll, self.max_scroll)
        screen.set_clip(panel)
        for y, line, title in rows:
            color = (131, 214, 226) if title else (229, 235, 244)
            screen.blit(
                self.panel_font.render(line, True, color),
                (panel.x, panel.y + y - self.scroll),
            )
        screen.set_clip(None)
        hint = (
            "Molette / Pg↑ Pg↓ : défiler"
            if self.max_scroll
            else "F1 : légende et commandes"
        )
        screen.blit(self.small_font.render(hint, True, (168, 185, 204)), (1092, 873))
        notice = (
            self.notice
            if time.monotonic() < self.notice_until
            else ("Clic / Tab : sélection | Espace : pause | Échap : fermer la vue")
        )
        screen.blit(self.small_font.render(notice, True, (246, 220, 152)), (16, 854))
        suffix = " (modifications locales)" if self.stamp["dirty"] else ""
        screen.blit(
            self.small_font.render(
                "Commit : " + self.stamp["commit"] + suffix, True, (178, 197, 215)
            ),
            (16, 877),
        )
        if self.show_help:
            self._legend()

    def _legend(self):
        import pygame

        box = pygame.Rect(36, 82, 650, 302)
        pygame.draw.rect(self.screen_canvas, (23, 32, 48), box, border_radius=8)
        lines = [
            "LÉGENDE ET COMMANDES — F1 pour fermer",
            "Vert : pas d'alerte | jaune : suspicion | rouge : confirmé",
            "Gris : information insuffisante (abstention du détecteur)",
            "Cyan : mouvement demandé | contour : ressource occupée",
            "Flèche jaune/rouge : demandeur → bloqueur (vue locale)",
            "Bleu : livraison V2V récente, liée à l'ego sélectionné",
            "V : masqué → utile reçu (4 max.) → tous liés à l'ego",
            "G : graphe | R : ressources | I : référence globale facultative",
            "Clic / Tab : choisir | Espace : pause / reprise",
            "+ / − (AZERTY ou pavé numérique) : cadence cible",
            "S : capture PNG + provenance | molette / Pg↑ Pg↓ : panneau",
            "Échap : ferme la vue ; le calcul finit sans fenêtre.",
        ]
        for index, line in enumerate(lines):
            self.screen_canvas.blit(
                self.panel_font.render(line, True, (234, 241, 249)),
                (52, 98 + index * 23),
            )

    def _action(self, action):
        if action == "close":
            self.user_closed = True
            self.paused = False
            self.close()
            return
        if action == "pause":
            self.paused = not self.paused
            self.deadline = time.monotonic()
            self._tell("Simulation en pause." if self.paused else "Simulation reprise.")
        elif action == "v2v":
            self.v2v_mode = (self.v2v_mode + 1) % len(V2V_MODES)
            self._tell("V2V : " + V2V_MODES[self.v2v_mode])
        elif action in ("graph", "resources", "reference", "help"):
            field = {
                "graph": "show_graph",
                "resources": "show_resources",
                "reference": "show_reference",
                "help": "show_help",
            }[action]
            setattr(self, field, not getattr(self, field))
            title = {
                "graph": "C-RDG",
                "resources": "Ressources",
                "reference": "Référence globale (prochain pas)",
                "help": "Légende",
            }[action]
            self._tell(
                title + (" : activé" if getattr(self, field) else " : désactivé")
            )
        elif action in ("faster", "slower"):
            self.rate_index = max(
                0,
                min(
                    len(RATES) - 1, self.rate_index + (1 if action == "faster" else -1)
                ),
            )
            self.deadline = time.monotonic()
            self._tell(
                f"Cadence cible : ×{RATES[self.rate_index]:g} (selon le coût CPU)."
            )
        elif action == "capture":
            self.capture_requested = True
            self._tell("Capture demandée…")
        elif action == "next" and self.frame:
            ids = sorted(
                self.frame["diagnostics"], key=lambda key: (len(label(key)), label(key))
            )
            if ids:
                self.selected = (
                    ids[(ids.index(self.selected) + 1) % len(ids)]
                    if self.selected in ids
                    else ids[0]
                )
                self.manual_selection = True
                self.scroll = 0
                self._tell("Véhicule sélectionné : " + label(self.selected))
        self.dirty = True

    def _handle_event(self):
        if self.no_window or self.closed:
            return
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._action("close")
                return
            if event.type in (pygame.WINDOWEXPOSED, pygame.WINDOWRESTORED):
                self.dirty = True
            elif event.type == pygame.MOUSEWHEEL:
                self.scroll = max(0, min(self.max_scroll, self.scroll - event.y * 60))
                self.dirty = True
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                button = next(
                    (
                        key
                        for key, rect in self.buttons.items()
                        if rect.collidepoint(event.pos)
                    ),
                    None,
                )
                if button:
                    self._action(button)
                elif self.frame and self.scene_rect.collidepoint(event.pos):
                    candidates = [
                        (math.dist(event.pos, self._pixel(vehicle.position)), key)
                        for key, vehicle in self.engine.agents.items()
                    ]
                    if candidates and min(candidates)[0] < 35:
                        self.selected = min(candidates)[1]
                        self.manual_selection = True
                        self.scroll = 0
                        self._tell("Véhicule sélectionné : " + label(self.selected))
            elif event.type == pygame.KEYDOWN and not getattr(event, "repeat", False):
                character = getattr(event, "unicode", "").lower()
                if event.key == pygame.K_ESCAPE:
                    self._action("close")
                    return
                if character == "+" or event.key in (
                    pygame.K_PLUS,
                    pygame.K_EQUALS,
                    pygame.K_KP_PLUS,
                ):
                    action = "faster"
                elif character in ("-", "−") or event.key in (
                    pygame.K_MINUS,
                    pygame.K_KP_MINUS,
                ):
                    action = "slower"
                else:
                    action = {
                        pygame.K_SPACE: "pause",
                        pygame.K_v: "v2v",
                        pygame.K_g: "graph",
                        pygame.K_r: "resources",
                        pygame.K_s: "capture",
                        pygame.K_TAB: "next",
                        pygame.K_F1: "help",
                        pygame.K_i: "reference",
                    }.get(event.key)
                if action:
                    self._action(action)
                elif event.key in (pygame.K_PAGEUP, pygame.K_PAGEDOWN):
                    self.scroll = max(
                        0,
                        min(
                            self.max_scroll,
                            self.scroll
                            + (240 if event.key == pygame.K_PAGEDOWN else -240),
                        ),
                    )
                    self.dirty = True

    def _capture(self):
        self.capture_requested = False
        if self.output is None:
            self._tell("Capture impossible : aucun dossier de sortie configuré.")
            return
        self.capture_count += 1
        metadata = dict(
            **self.context,
            **self.stamp,
            at_s=self.frame["at_s"],
            selected=self.selected,
            paused=self.paused,
            v2v=V2V_MODES[self.v2v_mode],
            graph=self.show_graph,
            resources=self.show_resources,
            reference=self.show_reference,
            authentic_pygame_frame=True,
        )
        try:
            self.last_capture = save_capture(
                self.screen_canvas,
                self.output / "captures",
                metadata,
                self.capture_count,
            )
            self._tell("Capture enregistrée (PNG + JSON) : " + self.last_capture.name)
        except (OSError, RuntimeError) as exc:
            self._tell("Échec de la capture : " + str(exc))

    def _redraw(self):
        self.render({}, to_image=False)
        self.dirty = False
        if self.capture_requested and not self.closed:
            self._capture()
            self._add_text({})
            self.blit()

    def inspect(self, frame):
        self.frame = frame
        if not self.manual_selection:
            holders = [key for key, view in frame["graphs"].items() if view["groups"]]
            if holders:
                self.selected = holders[0]
        if self.selected not in frame["diagnostics"] and frame["diagnostics"]:
            self.selected = next(iter(frame["diagnostics"]))
        if self.closed:
            return False
        self._remember_deliveries()
        self._redraw()
        if self.no_window:
            return True
        # Les événements sont servis à 60 Hz en pause et pendant la temporisation.
        # Une image immobile n'est redessinée qu'après une commande ou une exposition.
        while not self.closed:
            self._handle_event()
            if self.closed:
                return False
            if self.notice and time.monotonic() >= self.notice_until:
                self.notice = ""
                self.dirty = True
            if self.dirty:
                self._redraw()
            if not self.paused and time.monotonic() >= self.deadline:
                self.deadline = time.monotonic() + 1 / self.frames_per_second
                return True
            self.clock.tick(60)
        return False
