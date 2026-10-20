"""Inspection 2D interactive ; les commandes de la vue n'agissent pas sur les CAV."""

from collections import defaultdict, deque
from collections import OrderedDict
import copy
import math
from pathlib import Path
import time

from ..metadrive.view import TopDownView
from .presentation import french, label, save_capture, source_stamp, wrapped
from .layout import available_size, rectangles
from . import chrome, diagrams, v2v_info

COLORS = {
    "NO_ALERT": (67, 190, 130),
    "SUSPECTED": (246, 183, 58),
    "CONFIRMED_OBSTRUCTION": (232, 82, 94),
    "UNKNOWN": (154, 164, 180),
}
RATES = (0.25, 0.5, 1, 2, 4, 8)
V2V_MODES = ("masqué", "utile reçu", "tous liés à l'ego")


class ExperimentView(TopDownView):
    """Vue locale reçue, indépendante de la référence globale d'inspection."""

    def __init__(self, *, window=True, output=None, context=None, screen_size=None):
        import pygame

        self.frame = None
        self.selected = None
        self.manual_selection = False
        self.v2v_mode = 0
        self.active_view = "traffic"
        self.v2v_presentation = "list"
        self.inspected_message = None
        self.raw_packet = False
        self.message_buttons = {}
        self.show_graph = False
        self.show_resources = False
        self.show_details = True
        self.diagram_page = 0
        self.graph_nodes = {}
        self.resource_aliases = {}
        self.message_aliases = {}
        self.inspected_node = None
        self.info_snapshot = dict(nodes={}, edges=[], units=[])
        self.d01_history = defaultdict(lambda: deque(maxlen=301))
        self.label_font_size = 17
        self.zoom = 1.0
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
        self.recent_messages = OrderedDict()
        self.displayed_links = []
        size = screen_size or (available_size() if window else (1200, 720))
        self.scene_rect, self.panel_rect, self.footer_rect = rectangles(size)
        super().__init__(
            window=window,
            camera_position=(20, 0),
            scaling=4.5,
            screen_size=size,
            film_size=(1600, 1600),
        )
        self.clock = pygame.time.Clock()
        self.deadline = time.monotonic()
        pygame.font.init()
        self.panel_font = pygame.font.Font(None, 21)
        self.small_font = pygame.font.Font(None, 18)
        self._layout()
        if not self.no_window:
            self._screen_canvas = pygame.display.set_mode(size, pygame.RESIZABLE)
            pygame.display.set_caption("CAV Recovery — inspection MetaDrive")
            pygame.key.set_repeat()

    @property
    def show_v2v(self):
        return self.v2v_mode != 0

    @property
    def frames_per_second(self):
        return 5 * RATES[self.rate_index]

    @property
    def playback_rate(self):
        return RATES[self.rate_index]

    def header_text(self):
        at = self.frame["at_s"] if self.frame else 0
        count = len(self.frame["diagnostics"]) if self.frame else 0
        return (
            f"{self.context.get('scenario', 'MetaDrive')} · graine "
            f"{self.context.get('seed', '—')} · {at:.1f} s · {count} CAV actifs"
        )

    def _layout(self):
        width, _ = self.screen_canvas.get_size()
        lines = max(
            1, len(list(wrapped(self.header_text(), self.panel_font, width - 24)))
        )
        self.navigation_y = 12 + lines * 22
        self.scene_rect, self.panel_rect, self.footer_rect = rectangles(
            self.screen_canvas.get_size(),
            title_lines=lines,
            details=self.show_details,
            diagnostic=self.active_view != "traffic" or self.show_help,
        )
        import pygame

        if self.active_view == "v2v" and self.inspected_message is not None:
            self.graph_rect = pygame.Rect(
                self.panel_rect.x, self.panel_rect.y, self.panel_rect.width, 0
            )
            self.details_rect = self.panel_rect.copy()
        elif self.active_view == "v2v" and self.v2v_presentation == "list":
            self.graph_rect = self.panel_rect.copy()
            self.details_rect = pygame.Rect(
                self.panel_rect.x, self.panel_rect.bottom, self.panel_rect.width, 0
            )
        elif self.active_view == "traffic" and not self.show_help:
            self.graph_rect = pygame.Rect(
                self.panel_rect.x, self.panel_rect.y, self.panel_rect.width, 0
            )
            self.details_rect = self.panel_rect.copy()
        else:
            split = (
                int(self.panel_rect.height * 0.70)
                if self.show_details
                else self.panel_rect.height
            )
            self.graph_rect = pygame.Rect(
                self.panel_rect.x, self.panel_rect.y, self.panel_rect.width, split
            )
            self.details_rect = pygame.Rect(
                self.panel_rect.x,
                self.graph_rect.bottom,
                self.panel_rect.width,
                self.panel_rect.height - split,
            )
        self._fit_map()
        self.dirty = True

    def _fit_map(self):
        xmin, xmax, ymin, ymax = self.map.road_network.get_bounding_box()
        self.position = ((xmin + xmax) / 2, (ymin + ymax) / 2)
        self.scaling = self.zoom * min(
            self.scene_rect.width / max(1, xmax - xmin + 12),
            self.scene_rect.height / max(1, ymax - ymin + 12),
        )
        for canvas in (self._background_canvas, self._frame_canvas):
            canvas.scaling = self.scaling
            canvas.move_display_window_to(self.position)
        self._draw_roads()

    def _resize(self, size):
        import pygame

        self._screen_size = tuple(size)
        if self.no_window:
            self._screen_canvas = pygame.Surface(size)
        elif self.screen_canvas.get_size() != tuple(size):
            self._screen_canvas = pygame.display.set_mode(size, pygame.RESIZABLE)
        self._layout()

    def _remember_d01(self):
        for key, diagnostic in self.frame["diagnostics"].items():
            history = self.d01_history[key]
            if history and history[-1]["at_s"] >= self.frame["at_s"]:
                continue
            history.append(
                dict(
                    at_s=self.frame["at_s"],
                    status=diagnostic["status"],
                    speed=diagnostic["own_state"]["signed_speed_mps"],
                    requested=diagnostic["own_controls"]["requested_speed_mps"],
                )
            )

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
        self.recent_messages = OrderedDict(
            (key, value)
            for key, value in self.recent_messages.items()
            if now - value["received_at_s"] <= 0.6 + 1e-8
        )
        for event in self.frame["deliveries"]:
            if event["event"] == "emitted":
                sender = event["message"]["sender_id"]
                for recipient in event["delivered_to"]:
                    self.recent_links[sender, recipient] = now
                message = event["message"]
                channel = event.get("channel", "telemetry")
                identity = (channel, message.get("message_id", sender + ":" + str(now)))
                self.recent_messages[identity] = dict(
                    channel=channel,
                    message_id=identity[1],
                    sender=sender,
                    recipients=list(event["delivered_to"]),
                    observed_at_s=message.get("observed_at_s", now),
                    sent_at_s=message.get("sent_at_s", now),
                    received_at_s=now,
                    payload=copy.deepcopy(message),
                )

    def _v2v_snapshot(self):
        from .info_graph import message_snapshot

        pairs = set(self._useful_links(self.frame["graphs"].get(self.selected, {})))
        self.displayed_links = sorted(pairs)
        return message_snapshot(
            self.recent_messages.values(),
            pairs,
            self.message_aliases,
            useful=self.v2v_mode == 1,
        )

    def _open_message(self, key):
        node = self.info_snapshot["nodes"].get(key)
        if node is None or node["kind"] != "message":
            return
        self.inspected_message = copy.deepcopy(node)
        self.raw_packet = False
        self.show_help = False
        self.scroll = 0
        self._layout()
        self._tell(
            "Contenu du message " + node["label"] + " · L pour revenir aux échanges."
        )

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
            links.append((sender, recipient))
        return links

    def _draw(self, *args, **kwargs):
        self.buttons.clear()
        self.message_buttons.clear()
        self.graph_nodes = {}
        if self.active_view != "v2v":
            self.displayed_links = []
        if self.frame:
            for key, diagnostic in self.frame["diagnostics"].items():
                self.labels[key] = label(key)
                self.colors[key] = COLORS[diagnostic["status"]]
        # Le trafic est toujours dessiné, y compris pendant la consultation des preuves.
        super()._draw(*args, **kwargs)
        if self.selected in self.engine.agents:
            import pygame

            pygame.draw.circle(
                self.screen_canvas,
                (242, 247, 255),
                self._pixel(self.engine.agents[self.selected].position),
                12,
                1,
            )
        if not self.frame or self.active_view == "traffic":
            return
        self.screen_canvas.set_clip(self.graph_rect)
        if self.active_view == "v2v" and self.inspected_message is not None:
            pass  # Le paquet occupe le panneau droit ; le trafic reste à gauche.
        elif self.active_view == "v2v" and self.v2v_presentation == "list":
            v2v_info.draw_correspondents(self)
        elif self.active_view in ("graph", "v2v", "d01"):
            diagrams.draw_network(self, communication=self.active_view == "v2v")
        elif self.active_view == "history":
            self.info_snapshot = dict(nodes={}, edges=[], units=[])
            self.visible_info_edges = []
            diagrams.draw_d01(self)
        self.screen_canvas.set_clip(None)
        if getattr(self, "resource_inspection_requested", False):
            self.inspected_node = next(
                (
                    key
                    for key in self.graph_nodes
                    if self.info_snapshot["nodes"][key]["kind"] == "resource"
                ),
                None,
            )
            self.resource_inspection_requested = False

    def _tell(self, message):
        self.notice = message
        self.notice_until = time.monotonic() + 5
        self.dirty = True

    def _information_lines(self):
        if self.active_view == "traffic":
            return self._panel_lines()
        if self.active_view == "v2v":
            if self.inspected_message is not None:
                return list(
                    v2v_info.message_lines(
                        self.inspected_message, self.selected, raw=self.raw_packet
                    )
                )
            return v2v_info.own_lines(self.frame, self.selected) + [
                (
                    "Les mesures propres ne sont pas toutes diffusées. Cliquez sur M pour lire le paquet exact.",
                    False,
                ),
                (
                    "Les destinataires livrés sont montrés par l'instrumentation, sans être ajoutés aux entrées du CAV.",
                    False,
                ),
            ]
        nodes = self.info_snapshot["nodes"]
        node = nodes.get(self.inspected_node)
        if node is not None:
            lines = [("DÉTAILS · " + node["label"], True)]
            if node["kind"] == "resource":
                lines += [("Espace occupé étayé ; identifiant d'affichage.", False)]
                when = node.get("observed_at_s")
                if when is not None:
                    lines.append((f"Mesuré à {when:.1f} s", False))
                for edge in self.info_snapshot["edges"]:
                    if edge["target"] == self.inspected_node:
                        lines.append(
                            (
                                nodes[edge["source"]]["label"] + " demande cet espace",
                                False,
                            )
                        )
                    if edge["source"] == self.inspected_node:
                        lines.append(
                            ("Occupé par " + nodes[edge["target"]]["label"], False)
                        )
            elif node["kind"] == "message":
                lines += [
                    (
                        "Canal : "
                        + (
                            "diagnostic"
                            if node["channel"] == "diagnostic"
                            else "télémétrie"
                        ),
                        False,
                    ),
                    (
                        f"Émis par {label(node['sender'])} à {node['sent_at_s']:.1f} s",
                        False,
                    ),
                    (f"Livré à {node['received_at_s']:.1f} s", False),
                    (
                        "Destinataires : "
                        + ", ".join(label(key) for key in node["recipients"]),
                        False,
                    ),
                ]
            else:
                lines.append(
                    ("Véhicule apparaissant dans les preuves affichées.", False)
                )
                if node["vehicle_id"] == self.selected:
                    diagnostic = self.frame["diagnostics"].get(self.selected)
                    if diagnostic:
                        lines.append((french(diagnostic["status"]), False))
                for edge in self.info_snapshot["edges"]:
                    if self.inspected_node in (edge["source"], edge["target"]):
                        lines.append(
                            (
                                nodes[edge["source"]]["label"]
                                + " "
                                + edge["role"]
                                + " "
                                + nodes[edge["target"]]["label"],
                                False,
                            )
                        )
            return lines
        diagnostic = self.frame["diagnostics"].get(self.selected)
        lines = [("VÉHICULE " + label(self.selected or "—"), True)]
        if diagnostic:
            lines.append((french(diagnostic["status"]), False))
        for edge in getattr(self, "visible_info_edges", []):
            lines.append(
                (
                    nodes[edge["source"]]["label"]
                    + " "
                    + edge["role"]
                    + " "
                    + nodes[edge["target"]]["label"],
                    False,
                )
            )
        return lines

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
                f"Vue V2V : {V2V_MODES[self.v2v_mode]} ({len(self.displayed_links)} liens)",
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
        if self.frame:
            chrome.draw_chrome(self)

    def _legend(self):
        chrome.draw_legend(self)

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
        elif action in ("traffic", "graph", "v2v", "d01", "history"):
            self.active_view = "traffic" if self.active_view == action else action
            self.show_graph = self.active_view == "graph"
            self.show_resources = False
            self.inspected_node = None
            self.inspected_message = None
            self.scroll = 0
            self.v2v_mode = 2 if self.active_view == "v2v" else 0
            self.diagram_page = 0
            title = {
                "traffic": "Circulation",
                "graph": "Graphe C-RDG",
                "v2v": "Échanges V2V",
                "d01": "Détecteur D01",
                "history": "Historique D01",
            }[self.active_view]
            self._layout()
            self._tell("Panneau : " + title)
        elif action == "resources":
            self.inspected_message = None
            if self.active_view not in ("graph", "d01"):
                self.active_view = "graph"
                self.show_graph = True
                self.v2v_mode = 0
            self.show_resources = not self.show_resources
            self.resource_inspection_requested = self.show_resources
            self.inspected_node = None
            self.show_details = True
            self._layout()
            self._tell(
                "Cliquez sur un rectangle R pour consulter la preuve d'occupation."
            )
        elif action == "v2v_filter" and self.active_view == "v2v":
            self.v2v_mode = 2 if self.v2v_mode == 1 else 1
            self._tell("V2V : " + V2V_MODES[self.v2v_mode])
        elif action == "v2v_presentation" and self.active_view == "v2v":
            if self.inspected_message is not None:
                self.inspected_message = None
            else:
                self.v2v_presentation = (
                    "graph" if self.v2v_presentation == "list" else "list"
                )
            self.scroll = 0
            self.diagram_page = 0
            self._layout()
            self._tell(
                "V2V : "
                + (
                    "liste des correspondants"
                    if self.v2v_presentation == "list"
                    else "graphe des messages"
                )
            )
        elif action == "raw_packet" and self.inspected_message is not None:
            self.raw_packet = not self.raw_packet
            self.scroll = 0
        elif action in ("diagram_prev", "diagram_next"):
            self.diagram_page += -1 if action == "diagram_prev" else 1
            self.inspected_node = None
            self.scroll = 0
        elif action == "details":
            self.show_details = not self.show_details
            self._layout()
        elif action in ("reference", "help"):
            field = {
                "reference": "show_reference",
                "help": "show_help",
            }[action]
            setattr(self, field, not getattr(self, field))
            self._layout()
            title = {
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
                self.inspected_node = None
                self.inspected_message = None
                self.diagram_page = 0
                self._layout()
                self._tell("Véhicule sélectionné : " + label(self.selected))
        self.dirty = True

    def _handle_event(self):
        if self.no_window or self.closed:
            return
        import pygame

        events = pygame.event.get()
        sizes = [event.size for event in events if event.type == pygame.VIDEORESIZE]
        if sizes:
            self._resize(sizes[-1])
        for event in events:
            if event.type == pygame.QUIT:
                self._action("close")
                return
            if event.type == pygame.VIDEORESIZE:
                continue
            if event.type in (pygame.WINDOWEXPOSED, pygame.WINDOWRESTORED):
                self.dirty = True
            elif event.type == pygame.MOUSEWHEEL:
                if self.scene_rect.collidepoint(pygame.mouse.get_pos()):
                    self.zoom = max(0.75, min(3, self.zoom * 1.15**event.y))
                    self._fit_map()
                else:
                    self.scroll = max(
                        0, min(self.max_scroll, self.scroll - event.y * 60)
                    )
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
                    if button in self.message_buttons:
                        self._open_message(self.message_buttons[button])
                    else:
                        self._action(button)
                elif self.frame and self.panel_rect.collidepoint(event.pos):
                    node = next(
                        (
                            key
                            for key, rect in self.graph_nodes.items()
                            if rect.collidepoint(event.pos)
                        ),
                        None,
                    )
                    if node is not None:
                        if self.info_snapshot["nodes"][node]["kind"] == "message":
                            self._open_message(node)
                        else:
                            self.inspected_node = node
                            self.scroll = 0
                            self.dirty = True
                elif self.frame and self.scene_rect.collidepoint(event.pos):
                    candidates = [
                        (math.dist(event.pos, self._pixel(vehicle.position)), key)
                        for key, vehicle in self.engine.agents.items()
                    ]
                    if candidates and min(candidates)[0] < 35:
                        self.selected = min(candidates)[1]
                        self.manual_selection = True
                        self.scroll = 0
                        self.inspected_node = None
                        self.inspected_message = None
                        self.diagram_page = 0
                        self._layout()
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
                        pygame.K_d: "d01",
                        pygame.K_h: "history",
                        pygame.K_t: "traffic",
                        pygame.K_p: "details",
                        pygame.K_u: "v2v_filter",
                        pygame.K_l: "v2v_presentation",
                        pygame.K_j: "raw_packet",
                        pygame.K_LEFT: "diagram_prev",
                        pygame.K_RIGHT: "diagram_next",
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
            view=self.active_view,
            v2v_presentation=self.v2v_presentation,
            inspected_message_id=self.inspected_message["message_id"]
            if self.inspected_message
            else None,
            window_size=self.screen_canvas.get_size(),
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
        previous_selected = self.selected
        if not self.manual_selection:
            holders = [key for key, view in frame["graphs"].items() if view["groups"]]
            if holders:
                self.selected = holders[0]
        if self.selected not in frame["diagnostics"] and frame["diagnostics"]:
            self.selected = next(iter(frame["diagnostics"]))
        if self.closed:
            return False
        if self.selected != previous_selected:
            self.inspected_message = None
            self.inspected_node = None
            self.scroll = 0
            self.diagram_page = 0
            self._layout()
        self._remember_deliveries()
        self._remember_d01()
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
