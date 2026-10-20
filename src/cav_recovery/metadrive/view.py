"""Voitures 2D à l'échelle des corps physiques ; inspection sans action sur le trafic."""

import math

from metadrive.constants import MetaDriveType
from metadrive.engine.top_down_renderer import TopDownRenderer

from .observation import body


class TopDownView(TopDownRenderer):
    def __init__(
        self, *, window=True, labels=None, camera_position=(85, -10), scaling=6
    ):
        self.user_closed = False
        super().__init__(
            window=window,
            screen_size=(1000, 760),
            camera_position=camera_position,
            scaling=scaling,
            show_agent_name=False,
            num_stack=1,
            draw_target_vehicle_trajectory=False,
        )
        self.labels = (
            dict(labels)
            if labels is not None
            else {
                key: f"V{index}" for index, key in enumerate(sorted(self.engine.agents))
            }
        )
        self.palette = ((65, 161, 235), (244, 170, 61), (181, 117, 223), (81, 191, 153))
        self.colors = {
            key: self.palette[index % len(self.palette)]
            for index, key in enumerate(self.labels)
        }
        self._vehicle_font = None
        self._header_font = None
        self._draw_roads()

    def _draw_roads(self):
        """Les surfaces et marquages proviennent de la carte native, sans décor inventé."""
        import pygame

        canvas = self._background_canvas
        canvas.fill((222, 230, 223))
        features = self.map.get_map_features(1)
        for feature in features.values():
            if MetaDriveType.is_lane(feature["type"]):
                pygame.draw.polygon(
                    canvas,
                    (66, 74, 82),
                    [canvas.pos2pix(*xy[:2]) for xy in feature["polygon"]],
                )
        for feature in features.values():
            kind = feature["type"]
            if not (
                MetaDriveType.is_road_line(kind)
                or MetaDriveType.is_road_boundary_line(kind)
            ):
                continue
            color = (
                (239, 197, 90)
                if MetaDriveType.is_yellow_line(kind)
                else (224, 230, 234)
            )
            points = feature["polyline"]
            distance = 0.0
            for a, b in zip(points, points[1:]):
                # Les pointillés suivent leur longueur métrique, pas le nombre de sommets.
                if not MetaDriveType.is_broken_line(kind) or distance % 5 < 2.5:
                    pygame.draw.line(
                        canvas, color, canvas.pos2pix(*a[:2]), canvas.pos2pix(*b[:2]), 1
                    )
                distance += math.dist(a[:2], b[:2])

    def vehicle_pixels(self, vehicle):
        """Projette l'empreinte physique complète, sans arrondir ses dimensions d'abord."""
        return [
            self._frame_canvas.pos2pix(*xy) for xy in body(vehicle).exterior.coords[:-1]
        ]

    def _draw_car(self, key, vehicle):
        import pygame

        canvas = self._frame_canvas
        outline = self.vehicle_pixels(vehicle)
        color = self.colors[key]
        # Le contour rectangulaire reste la limite du corps, y compris les coins de la silhouette.
        pygame.draw.polygon(canvas, (148, 162, 173), outline, 1)

        def local(points):
            c, s = math.cos(vehicle.heading_theta), math.sin(vehicle.heading_theta)
            x, y = vehicle.position
            return [
                canvas.pos2pix(
                    x + a * vehicle.LENGTH * c - b * vehicle.WIDTH * s,
                    y + a * vehicle.LENGTH * s + b * vehicle.WIDTH * c,
                )
                for a, b in points
            ]

        silhouette = local(
            (
                (0.50, 0.30),
                (0.42, 0.50),
                (-0.42, 0.50),
                (-0.50, 0.34),
                (-0.50, -0.34),
                (-0.42, -0.50),
                (0.42, -0.50),
                (0.50, -0.30),
            )
        )
        pygame.draw.polygon(canvas, color, silhouette)
        pygame.draw.polygon(canvas, (22, 31, 39), silhouette, 1)
        # Pare-brise à l'avant (+longueur), lunette arrière et toit laissent le cap lisible.
        pygame.draw.polygon(
            canvas,
            (28, 51, 68),
            local(((0.25, 0.34), (0.25, -0.34), (0.08, -0.39), (0.08, 0.39))),
        )
        pygame.draw.polygon(
            canvas,
            (39, 61, 76),
            local(((-0.23, 0.34), (-0.23, -0.34), (-0.34, -0.29), (-0.34, 0.29))),
        )
        pygame.draw.polygon(
            canvas,
            tuple(min(255, c + 28) for c in color),
            local(((0.04, 0.32), (0.04, -0.32), (-0.19, -0.32), (-0.19, 0.32))),
        )
        for side in (-1, 1):
            pygame.draw.line(
                canvas,
                (255, 246, 193),
                *local(((0.40, side * 0.29), (0.40, side * 0.42))),
                2,
            )
            pygame.draw.line(
                canvas,
                (204, 53, 64),
                *local(((-0.41, side * 0.28), (-0.41, side * 0.41))),
                2,
            )
            # Les roues sont à l'intérieur de l'empreinte : aucune largeur ajoutée au dessin.
            for axle in (-0.27, 0.28):
                pygame.draw.line(
                    canvas,
                    (22, 31, 39),
                    *local(((axle - 0.06, side * 0.43), (axle + 0.06, side * 0.43))),
                    1,
                )

    def _draw(self, *args, **kwargs):
        import pygame

        footprints = {}
        for key, vehicle in sorted(self.engine.agents.items()):
            if key not in self.labels:
                self.labels[key] = f"V{len(self.labels)}"
                self.colors[key] = self.palette[
                    (len(self.labels) - 1) % len(self.palette)
                ]
            self._draw_car(key, vehicle)
            footprints[key] = self.vehicle_pixels(vehicle)
        width, height = self.screen_canvas.get_size()
        cx, cy = self._frame_canvas.pos2pix(*self.position)
        offset = (cx - width / 2, cy - height / 2)
        self.screen_canvas.blit(self._frame_canvas, (0, 0), (*offset, width, height))
        if self._vehicle_font is None:
            if not pygame.font.get_init():
                pygame.font.init()
            self._vehicle_font = pygame.font.Font(None, 21)
        placed = []
        bodies = {
            key: pygame.Rect(
                min(x for x, y in points) - offset[0],
                min(y for x, y in points) - offset[1],
                max(x for x, y in points) - min(x for x, y in points) + 1,
                max(y for x, y in points) - min(y for x, y in points) + 1,
            )
            for key, points in footprints.items()
        }
        for key, vehicle in sorted(self.engine.agents.items()):
            x, y = self._frame_canvas.pos2pix(*vehicle.position)
            x, y = x - offset[0], y - offset[1]
            label = self._vehicle_font.render(
                self.labels[key], True, (24, 33, 41), (248, 250, 249)
            )
            for dx, dy in ((18, -25), (18, 15), (-38, -25), (-38, 15), (18, -43)):
                rect = label.get_rect(
                    topleft=(
                        min(max(0, x + dx), width - label.get_width()),
                        min(max(0, y + dy), height - label.get_height()),
                    )
                )
                if not any(
                    rect.colliderect(other) for other in placed + list(bodies.values())
                ):
                    break
            placed.append(rect)
            clipped = bodies[key].clipline((x, y), rect.center)
            start = clipped[1] if clipped else (x, y)
            pygame.draw.line(self.screen_canvas, (224, 230, 234), start, rect.center, 1)
            self.screen_canvas.blit(label, rect)

    def _add_text(self, text):
        import pygame

        if self._header_font is None:
            if not pygame.font.get_init():
                pygame.font.init()
            self._header_font = pygame.font.Font(None, 24)
        for index, (key, value) in enumerate((text or {}).items()):
            self.screen_canvas.blit(
                self._header_font.render(f"{key} : {value}", True, (24, 33, 41)),
                (24, 24 + index * 22),
            )

    def _handle_event(self):
        if self.no_window:
            return
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                self.user_closed = True
