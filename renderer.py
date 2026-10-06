import pygame

from config import (
    BIRD_RADIUS,
    COLORS,
    GROUND_Y,
    PIPE_GAP,
    PIPE_WIDTH,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)


class Renderer:
    def __init__(self, screen):
        self.screen = screen
        self.bird_surface = pygame.Surface(
            (SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA
        )

    def draw(self, game, best_idx):
        self.screen.fill(COLORS["sky"])
        self._draw_pipes(game.pipes)
        self._draw_ground()
        self._draw_birds(game.birds, best_idx)

    def _draw_pipes(self, pipes):
        for pipe in pipes:
            top_bottom = pipe.gap_center_y - PIPE_GAP / 2
            bottom_top = pipe.gap_center_y + PIPE_GAP / 2

            cap_h = 6

            if top_bottom > 0:
                body_rect = pygame.Rect(pipe.x + 2, 0, PIPE_WIDTH - 4, max(0, top_bottom - cap_h))
                pygame.draw.rect(self.screen, COLORS["pipe"], body_rect)
                pygame.draw.rect(self.screen, COLORS["pipe_border"], body_rect, 2)

                cap_rect = pygame.Rect(pipe.x, max(0, top_bottom - cap_h), PIPE_WIDTH, cap_h)
                pygame.draw.rect(self.screen, COLORS["pipe_cap"], cap_rect)
                pygame.draw.rect(self.screen, COLORS["pipe_border"], cap_rect, 2)

            if bottom_top < GROUND_Y:
                cap_rect = pygame.Rect(pipe.x, bottom_top, PIPE_WIDTH, cap_h)
                pygame.draw.rect(self.screen, COLORS["pipe_cap"], cap_rect)
                pygame.draw.rect(self.screen, COLORS["pipe_border"], cap_rect, 2)

                body_rect = pygame.Rect(pipe.x + 2, bottom_top + cap_h, PIPE_WIDTH - 4, GROUND_Y - bottom_top - cap_h)
                pygame.draw.rect(self.screen, COLORS["pipe"], body_rect)
                pygame.draw.rect(self.screen, COLORS["pipe_border"], body_rect, 2)

    def _draw_ground(self):
        pygame.draw.rect(
            self.screen, COLORS["ground"],
            (0, GROUND_Y, SCREEN_WIDTH, SCREEN_HEIGHT - GROUND_Y)
        )
        pygame.draw.line(
            self.screen, COLORS["ground_line"],
            (0, GROUND_Y), (SCREEN_WIDTH, GROUND_Y), 2
        )

        stripe_spacing = 30
        for x in range(0, SCREEN_WIDTH, stripe_spacing):
            pygame.draw.line(
                self.screen, COLORS["ground_line"],
                (x, GROUND_Y + 4), (x + 10, SCREEN_HEIGHT), 1
            )

    def _draw_birds(self, birds, best_idx):
        self.bird_surface.fill((0, 0, 0, 0))

        for i, bird in enumerate(birds):
            if not bird.alive:
                continue
            if i == best_idx:
                continue
            color = (*COLORS["bird_alive"], 70)
            pygame.draw.circle(
                self.bird_surface, color,
                (int(bird.x), int(bird.y)), BIRD_RADIUS
            )

        if best_idx < len(birds) and birds[best_idx].alive:
            bird = birds[best_idx]
            color = (*COLORS["bird_best"], 220)
            pygame.draw.circle(
                self.bird_surface, color,
                (int(bird.x), int(bird.y)), BIRD_RADIUS + 2
            )
            outline = (*COLORS["bird_best"][:3], 255)
            pygame.draw.circle(
                self.bird_surface, outline,
                (int(bird.x), int(bird.y)), BIRD_RADIUS + 2, 2
            )

        self.screen.blit(self.bird_surface, (0, 0))
