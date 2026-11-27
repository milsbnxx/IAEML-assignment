"""Off-screen movie renderer, using the ray overlay from the course frontend."""
import math

import numpy as np
import pygame

from utils.renderer import PygameFrontend


BACKGROUND = "#10202e"
PLOT = "#142939"
WALL = "#526979"
TRAFFIC = "#6faad9"
TAXI = "#ffbd70"
GOAL = "#94e3a6"
RAYS = (111, 205, 204)
TEXT = "#edf4f5"
MUTED = "#a7bec8"


class TaxiRenderer(PygameFrontend):
    def __init__(self, params, title, map_id, seed):
        pygame.font.init()
        self.title = title
        self.context = f"Map {map_id}    /    Seed {seed}"
        self.h, self.w = params.map_height_width
        self.cell = 15
        board_size = ((self.w + 4) * self.cell, (self.h + 4) * self.cell)
        self.board = pygame.Surface(board_size)
        self.screen = pygame.Surface((board_size[0] + 64, board_size[1] + 160))
        self.title_font = pygame.font.Font(None, 38)
        self.data_font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        self.trail = []

    def pixel(self, point):
        return (round((float(point[0]) + 2) * self.cell),
                round((self.h + 2 - float(point[1])) * self.cell))

    def text(self, surface, content, pos, font, color=TEXT):
        surface.blit(font.render(content, True, color), pos)

    def frame(self, state, obs, params, status):
        pos = np.asarray(state.agent_pos)
        centre = self.pixel(pos)
        self.trail.append(centre)
        self.screen.fill(BACKGROUND)
        self.board.fill(PLOT)
        for x in range(0, self.w + 1, 3):
            pygame.draw.line(self.board, "#21394a", self.pixel((x, 0)), self.pixel((x, self.h)))
        for y in range(0, self.h + 1, 3):
            pygame.draw.line(self.board, "#21394a", self.pixel((0, y)), self.pixel((self.w, y)))
        for point in np.asarray(state.static_obstacles):
            pygame.draw.circle(self.board, WALL, self.pixel(point), self.cell)
        for point, velocity in zip(np.asarray(state.kinematic_obstacles),
                                   np.asarray(state.kinematic_obst_velocities)):
            pygame.draw.circle(self.board, TRAFFIC, self.pixel(point), self.cell)
            pygame.draw.line(self.board, BACKGROUND, self.pixel(point), self.pixel(point + 0.65 * velocity), 3)
        goal = self.pixel(state.goal_pos)
        pygame.draw.circle(self.board, GOAL, goal, self.cell, 3)
        pygame.draw.circle(self.board, GOAL, goal, 4)
        if len(self.trail) > 1:
            pygame.draw.lines(self.board, TAXI, False, self.trail, 3)

        # PygameFrontend expects ray vectors as (screen row, screen column).
        lengths = np.asarray(obs.collision_rays)
        row_col = np.column_stack((-lengths[:, 1], lengths[:, 0]))
        origin = (self.h + 1.5 - pos[1], pos[0] + 1.5)
        self.draw_rays_transparent(self.board, origin, row_col, self.cell, alpha=115, color=RAYS)

        # The outline is the actual one-metre collision circle. The body shows the heading.
        pygame.draw.circle(self.board, TAXI, centre, self.cell, 1)
        forward = np.asarray(state.agent_forward_dir)
        left = np.array((-forward[1], forward[0]))

        def body_point(longitudinal, lateral):
            return self.pixel(pos + longitudinal * forward + lateral * left)

        corners = [(0.91, 0.0), (0.56, 0.55), (-0.73, 0.55), (-0.73, -0.55), (0.56, -0.55)]
        pygame.draw.polygon(self.board, TAXI, [body_point(*corner) for corner in corners])
        pygame.draw.line(self.board, BACKGROUND, body_point(0.28, 0.38), body_point(0.28, -0.38), 3)

        self.text(self.screen, self.title, (32, 19), self.title_font)
        context = self.small_font.render(self.context, True, MUTED)
        self.screen.blit(context, (self.screen.get_width() - 32 - context.get_width(), 29))
        self.screen.blit(self.board, (32, 67))
        pygame.draw.rect(self.screen, "#405868", pygame.Rect((32, 67), self.board.get_size()), 1)
        base = 67 + self.board.get_height()
        labels = ((WALL, "Static obstacle"), (TRAFFIC, "Moving obstacle"), (GOAL, "Goal"),
                  (TAXI, "Taxi / driven path"), (RAYS, "Sensor ray"))
        for x, (color, label) in zip((32, 213, 416, 520, 740), labels):
            pygame.draw.circle(self.screen, color, (x + 5, base + 22), 5)
            self.text(self.screen, label, (x + 16, base + 14), self.small_font, MUTED)
        distance = float(np.linalg.norm(pos - np.asarray(state.goal_pos)))
        telemetry = (f"{int(state.time) / params.fps:4.1f} s       "
                     f"Speed {float(state.speed):.1f} m/s       "
                     f"Steering {math.degrees(float(state.steering)):+.0f} deg       "
                     f"Goal {distance:.1f} m")
        self.text(self.screen, telemetry, (32, base + 53), self.data_font)
        badge = self.data_font.render(status, True, GOAL if status == "Goal reached" else TEXT)
        self.screen.blit(badge, (self.screen.get_width() - 32 - badge.get_width(), base + 53))
        return pygame.image.tobytes(self.screen, "RGB")
