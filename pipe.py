import random
import math

from config import (
    GROUND_Y,
    MOVING_PIPE_AMPLITUDE,
    MOVING_PIPE_FREQUENCY,
    PIPE_GAP,
    PIPE_GAP_Y_MAX,
    PIPE_GAP_Y_MIN,
    PIPE_WIDTH,
    BIRD_RADIUS,
)


class Pipe:
    _next_id = 0

    def __init__(self, x, gap_center_y=None, moving=False):
        self.x = float(x)
        if gap_center_y is None:
            gap_center_y = random.randint(PIPE_GAP_Y_MIN, PIPE_GAP_Y_MAX)
        self.gap_center_y = float(gap_center_y)
        self.base_gap_y = self.gap_center_y
        self.moving = moving
        self.phase = random.uniform(0, 2 * math.pi)
        self.pipe_id = Pipe._next_id
        Pipe._next_id += 1

    def update(self, speed, frame):
        self.x -= speed
        if self.moving:
            offset = math.sin(frame * MOVING_PIPE_FREQUENCY + self.phase) * MOVING_PIPE_AMPLITUDE
            self.gap_center_y = self.base_gap_y + offset
            self.gap_center_y = max(PIPE_GAP_Y_MIN, min(PIPE_GAP_Y_MAX, self.gap_center_y))

    @property
    def top_rect(self):
        top_bottom = self.gap_center_y - PIPE_GAP / 2
        return (self.x, 0, PIPE_WIDTH, top_bottom)

    @property
    def bottom_rect(self):
        bottom_top = self.gap_center_y + PIPE_GAP / 2
        return (self.x, bottom_top, PIPE_WIDTH, GROUND_Y - bottom_top)

    def collides(self, bird_x, bird_y):
        for rx, ry, rw, rh in [self.top_rect, self.bottom_rect]:
            if rh <= 0:
                continue
            closest_x = max(rx, min(bird_x, rx + rw))
            closest_y = max(ry, min(bird_y, ry + rh))
            dx = bird_x - closest_x
            dy = bird_y - closest_y
            if dx * dx + dy * dy < BIRD_RADIUS * BIRD_RADIUS:
                return True
        return False

    @property
    def off_screen(self):
        return self.x + PIPE_WIDTH < 0
