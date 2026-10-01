import random

from config import (
    GRAVITY,
    GRAVITY_CHANGE_INTERVAL,
    GRAVITY_RANGE,
    WIND_CHANGE_INTERVAL,
    WIND_STRENGTH_RANGE,
)


class Environment:
    def __init__(self, enable_wind=True, enable_variable_gravity=True):
        self.enable_wind = enable_wind
        self.enable_variable_gravity = enable_variable_gravity
        self.wind = 0.0
        self.gravity = GRAVITY
        self.frame = 0

    def update(self):
        self.frame += 1
        if self.enable_wind and self.frame % WIND_CHANGE_INTERVAL == 0:
            self.wind = random.uniform(*WIND_STRENGTH_RANGE)
        if self.enable_variable_gravity and self.frame % GRAVITY_CHANGE_INTERVAL == 0:
            self.gravity = random.uniform(*GRAVITY_RANGE)

    def reset(self):
        self.wind = 0.0
        self.gravity = GRAVITY
        self.frame = 0
