"""Inspection dans le rendu 2D natif MetaDrive, sans fenêtre externe de diagnostic."""

import math
from pathlib import Path

from shapely import wkt

from ..metadrive.view import TopDownView

COLORS = {
    "NO_ALERT": (67, 190, 130),
    "SUSPECTED": (246, 183, 58),
    "CONFIRMED_OBSTRUCTION": (232, 82, 94),
    "UNKNOWN": (154, 164, 180),
}


def label(key):
    return "V" + key.removeprefix("agent") if key.startswith("agent") else key


class ExperimentView(TopDownView):
    """Les flèches V2V correspondent aux livraisons et le graphe à l'ego choisi."""

    def __init__(self, *, window=True, output=None):
        self.frame = None
        self.selected = None
        self.manual_selection = False
        self.show_v2v = True
        self.show_graph = True
        self.show_resources = True
        self.paused = False
        self.frames_per_second = 20
        self.output = Path(output) if output is not None else None
        self.panel_font = None
        super().__init__(
            window=window, camera_position=(20, 0), scaling=4.5, screen_size=(1440, 900)
        )
        import pygame

        self.clock = pygame.time.Clock()

    def _pixel(self, point):
        width, height = self.screen_canvas.get_size()
        cx, cy = self._frame_canvas.pos2pix(*self.position)
        px, py = self._frame_canvas.pos2pix(*point)
        return px - cx + width / 2, py - cy + height / 2

    def _arrow(self, start, end, color, width=2):
        import pygame

        pygame.draw.line(self.screen_canvas, color, start, end, width)
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        left = (
            end[0] - 10 * math.cos(angle - 0.45),
            end[1] - 10 * math.sin(angle - 0.45),
        )
        right = (
            end[0] - 10 * math.cos(angle + 0.45),
            end[1] - 10 * math.sin(angle + 0.45),
        )
        pygame.draw.polygon(self.screen_canvas, color, [end, left, right])

    def _draw(self, *args, **kwargs):
        if self.frame:
            for key, diagnostic in self.frame["diagnostics"].items():
                self.labels[key] = label(key)
                self.colors[key] = COLORS[diagnostic["status"]]
        super()._draw(*args, **kwargs)
        if not self.frame:
            return
        import pygame

        view = self.frame["graphs"].get(self.selected, {})
        own_packet = self.frame["measurements"].get(self.selected, {})
        plan = own_packet.get("own_controls", {}).get("planned_path_xyh_m", [])
        if len(plan) >= 2:
            pygame.draw.lines(
                self.screen_canvas,
                (116, 222, 238),
                False,
                [self._pixel(pose[:2]) for pose in plan],
                2,
            )
        if self.show_v2v:
            for event in self.frame["deliveries"]:
                if event["event"] != "emitted":
                    continue
                sender = event["message"]["sender_id"]
                for recipient in event["delivered_to"]:
                    if self.selected not in (sender, recipient):
                        continue
                    if sender in self.engine.agents and recipient in self.engine.agents:
                        self._arrow(
                            self._pixel(self.engine.agents[sender].position),
                            self._pixel(self.engine.agents[recipient].position),
                            (67, 149, 245),
                            1,
                        )
        for edge in view.get("edges", []):
            if edge["state"] != "CURRENT":
                continue
            color = (232, 82, 94) if edge["level"] == "CONFIRMED" else (246, 183, 58)
            if self.show_resources:
                geometry = wkt.loads(edge["resource_wkt"])
                polygons = (
                    list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
                )
                for polygon in polygons:
                    if polygon.geom_type != "Polygon":
                        continue
                    pygame.draw.polygon(
                        self.screen_canvas,
                        color,
                        [self._pixel(point) for point in polygon.exterior.coords],
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

    def _add_text(self, text):
        if not self.frame:
            return
        import pygame

        if self.panel_font is None:
            self.panel_font = pygame.font.Font(None, 20)
        panel = pygame.Rect(1080, 0, 360, 900)
        pygame.draw.rect(self.screen_canvas, (23, 32, 48), panel)
        diagnostic = self.frame["diagnostics"].get(self.selected)
        graph = self.frame["graphs"].get(self.selected, {})
        lines = [
            "CAV Recovery | MetaDrive",
            f"Temps : {self.frame['at_s']:.1f} s",
            f"CAV actifs : {len(self.frame['diagnostics'])}",
            "",
            "Vue locale sélectionnée : " + label(self.selected or "—"),
        ]
        if diagnostic:
            lines += [
                diagnostic["status"],
                "Motif : " + diagnostic["reason"],
                "Destination : " + diagnostic["destination"],
                f"Vitesse : {diagnostic['own_state']['signed_speed_mps']:.2f} m/s",
                f"Demandée / acceptée : {diagnostic['own_controls']['requested_speed_mps']:.1f} / {diagnostic['own_controls']['accepted_speed_mps']:.1f} m/s",
                f"Début preuve : {diagnostic['coverage']['series_start_s']}",
                f"Début effet : {diagnostic['coverage']['effect_start_s']}",
                f"Rendez-vous : {len(diagnostic['coverage']['proof_times_s'])}",
            ]
            if diagnostic["events"]:
                event = diagnostic["events"][-1]
                lines.append(f"{event['kind']} : {event['available_at_s']:.1f} s")
        lines += [
            "",
            "C-RDG : vue partielle reçue",
            f"Groupes confirmés : {len(graph.get('groups', []))}",
        ]
        affected = {
            key
            for group in graph.get("groups", [])
            for key in group["affected_requesters"]
        }
        lines.append(f"Demandeurs étayés : {len(affected)}")
        lines.append(
            f"Relations courantes : {sum(edge['state'] == 'CURRENT' for edge in graph.get('edges', []))}"
        )
        costs = self.frame["costs_ms"]
        total_ms = sum(
            costs.get(key, 0.0)
            for key in ("adaptation", "detector", "diagnostic_communication", "crdg")
        )
        lines.append(f"D01 + rapports + graphe : {total_ms:.1f} ms")
        independent = self.frame.get("independent_reference")
        if independent is not None:
            lines.append(
                f"Réf. native (préfixe) : {len(independent['affected_requesters'])}"
            )
        for group in graph.get("groups", [])[:2]:
            lines.append(
                "Affectés : "
                + ", ".join(label(key) for key in group["affected_requesters"][:6])
                + (
                    f" (+{len(group['affected_requesters']) - 6})"
                    if len(group["affected_requesters"]) > 6
                    else ""
                )
            )
        lines += ["", "Messages V2V réellement reçus :"]
        inbox = self.frame["measurements"].get(self.selected, {}).get("v2v")
        for entry in (inbox or {}).get("received", [])[:3]:
            message = entry["message"]
            lines.append(
                f"{label(message['sender_id'])} : âge {self.frame['at_s'] - message['observed_at_s']:.1f} s"
            )
        lines += [
            "",
            "Vert : pas d'alerte | jaune : suspicion",
            "Rouge : confirmé | gris : inconnu",
            "Bleu : rapport V2V livré",
            "Flèche : demandeur vers bloqueur",
            "Cyan : mouvement propre demandé",
            "Contour : espace occupé prouvé",
            "",
            "Clic / Tab : choisir un véhicule",
            "V : V2V | G : graphe | R : ressources",
            "Espace : pause | + / - : vitesse",
            "S : capture | Échap : fermer la vue",
        ]
        for index, line in enumerate(lines):
            rendered = self.panel_font.render(line, True, (226, 234, 246))
            self.screen_canvas.blit(rendered, (1092, 16 + index * 19))

    def _handle_event(self):
        if self.no_window:
            return
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.user_closed = True
            elif event.type == pygame.MOUSEBUTTONDOWN and self.frame:
                candidates = [
                    (math.dist(event.pos, self._pixel(vehicle.position)), key)
                    for key, vehicle in self.engine.agents.items()
                ]
                if candidates and min(candidates)[0] < 35:
                    self.selected = min(candidates)[1]
                    self.manual_selection = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.user_closed = True
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_v:
                    self.show_v2v = not self.show_v2v
                elif event.key == pygame.K_g:
                    self.show_graph = not self.show_graph
                elif event.key == pygame.K_r:
                    self.show_resources = not self.show_resources
                elif event.key == pygame.K_TAB and self.frame:
                    self.manual_selection = True
                    ids = sorted(self.frame["diagnostics"])
                    if ids:
                        self.selected = (
                            ids[(ids.index(self.selected) + 1) % len(ids)]
                            if self.selected in ids
                            else ids[0]
                        )
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    self.frames_per_second = min(100, self.frames_per_second + 5)
                elif event.key == pygame.K_MINUS:
                    self.frames_per_second = max(1, self.frames_per_second - 5)
                elif event.key == pygame.K_s and self.output is not None:
                    pygame.image.save(
                        self.screen_canvas,
                        str(self.output / f"inspection-{self.frame['at_s']:.1f}.png"),
                    )

    def inspect(self, frame):
        self.frame = frame
        if not self.manual_selection:
            holders = [key for key, view in frame["graphs"].items() if view["groups"]]
            if holders:
                self.selected = holders[0]
        if self.selected not in frame["diagnostics"] and frame["diagnostics"]:
            self.selected = next(iter(frame["diagnostics"]))
        if self.user_closed:
            return
        self.render({}, to_image=False)
        if not self.no_window:
            self.clock.tick(self.frames_per_second)
            while self.paused and not self.user_closed:
                self.render({}, to_image=False)
                self.clock.tick(20)
