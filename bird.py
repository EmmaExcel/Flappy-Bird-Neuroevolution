from config import (
    BIRD_RADIUS,
    BIRD_X,
    FLAP_STRENGTH,
    GRAVITY_RANGE,
    MAX_FALL_SPEED,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    WIND_STRENGTH_RANGE,
)


class Bird:
    def __init__(self, network):
        self.x = BIRD_X
        self.y = SCREEN_HEIGHT / 2.0
        self.velocity = 0.0
        self.alive = True
        self.score = 0
        self.distance = 0
        self.network = network
        self.scored_pipes = set()
        self.max_pipe_progress = 0.0

    def flap(self):
        self.velocity = FLAP_STRENGTH

    def update(self, gravity, wind):
        self.velocity += gravity
        self.velocity = min(self.velocity, MAX_FALL_SPEED)
        self.y += self.velocity
        self.x += wind
        self.x = max(BIRD_RADIUS, min(SCREEN_WIDTH - BIRD_RADIUS, self.x))
        self.distance += 1

    def get_inputs(self, next_pipe, wind, gravity):
        wind_max = max(abs(WIND_STRENGTH_RANGE[0]), abs(WIND_STRENGTH_RANGE[1]), 0.01)
        gravity_max = max(GRAVITY_RANGE[1], 0.01)
        return [
            self.y / SCREEN_HEIGHT,
            self.velocity / MAX_FALL_SPEED,
            next_pipe.gap_center_y / SCREEN_HEIGHT,
            (next_pipe.x - self.x) / SCREEN_WIDTH,
            wind / wind_max,
            gravity / gravity_max,
        ]

    def decide(self, inputs):
        output = self.network.forward(inputs)
        if output[0] > 0.5:
            self.flap()

    def is_out_of_bounds(self, ground_y):
        return self.y - BIRD_RADIUS < 0 or self.y + BIRD_RADIUS > ground_y
