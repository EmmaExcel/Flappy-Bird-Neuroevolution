import pygame

from config import COLORS, SCREEN_WIDTH, SCREEN_HEIGHT, GROUND_Y


class HUD:
    def __init__(self, screen):
        self.screen = screen
        self.font_large = None
        self.font_small = None
        self._init_fonts()

    def _init_fonts(self):
        pygame.font.init()
        try:
            self.font_large = pygame.font.SysFont("menlo", 16, bold=True)
            self.font_small = pygame.font.SysFont("menlo", 13)
        except Exception:
            self.font_large = pygame.font.Font(None, 20)
            self.font_small = pygame.font.Font(None, 16)

    def draw(self, generation, alive, best_score, wind, gravity,
             best_fitness_history, avg_fitness_history, speed_mult):
        self._draw_stats_panel(generation, alive, best_score, wind, gravity, speed_mult)
        self._draw_fitness_graph(best_fitness_history, avg_fitness_history)

    def _draw_stats_panel(self, generation, alive, best_score, wind, gravity, speed_mult):
        panel_w = 200
        panel_h = 140
        panel_x = 10
        panel_y = 10

        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill(COLORS["hud_bg"])
        self.screen.blit(panel, (panel_x, panel_y))

        lines = [
            (f"GEN  {generation}", COLORS["hud_accent"]),
            (f"ALIVE  {alive}/100", COLORS["hud_text"] if alive > 10 else COLORS["hud_warn"]),
            (f"BEST   {best_score} pipes", COLORS["hud_text"]),
            (f"WIND   {wind:+.2f}", COLORS["hud_text"]),
            (f"GRAV   {gravity:.2f}", COLORS["hud_text"]),
            (f"SPEED  {speed_mult}x", COLORS["hud_text"]),
        ]

        y_offset = panel_y + 8
        for text, color in lines:
            surf = self.font_large.render(text, True, color[:3])
            self.screen.blit(surf, (panel_x + 12, y_offset))
            y_offset += 20

    def _draw_fitness_graph(self, best_history, avg_history):
        if len(best_history) < 2:
            return

        graph_w = SCREEN_WIDTH - 20
        graph_h = 60
        graph_x = 10
        graph_y = GROUND_Y + 8

        bg = pygame.Surface((graph_w, graph_h), pygame.SRCALPHA)
        bg.fill(COLORS["hud_bg"])
        self.screen.blit(bg, (graph_x, graph_y))

        label = self.font_small.render("FITNESS", True, COLORS["hud_text"][:3])
        self.screen.blit(label, (graph_x + 4, graph_y + 2))

        max_val = max(max(best_history), max(avg_history), 1)
        n = len(best_history)
        visible = min(n, graph_w - 20)
        start = n - visible

        def to_points(history):
            points = []
            for i in range(visible):
                idx = start + i
                px = graph_x + 10 + int(i * (graph_w - 20) / max(visible - 1, 1))
                py = graph_y + graph_h - 6 - int((history[idx] / max_val) * (graph_h - 16))
                points.append((px, py))
            return points

        avg_points = to_points(avg_history)
        best_points = to_points(best_history)

        if len(avg_points) >= 2:
            pygame.draw.lines(self.screen, COLORS["graph_avg"][:3], False, avg_points, 1)
        if len(best_points) >= 2:
            pygame.draw.lines(self.screen, COLORS["graph_line"][:3], False, best_points, 2)

        gen_label = self.font_small.render(f"gen {n}", True, COLORS["hud_text"][:3])
        self.screen.blit(gen_label, (graph_x + graph_w - 60, graph_y + 2))

        val_label = self.font_small.render(f"{int(best_history[-1])}", True, COLORS["graph_line"][:3])
        self.screen.blit(val_label, (graph_x + graph_w - 60, graph_y + graph_h - 16))
