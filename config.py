SCREEN_WIDTH = 600
SCREEN_HEIGHT = 800
FPS = 60
GROUND_Y = 720

BIRD_RADIUS = 12
BIRD_X = 120
GRAVITY = 0.5
FLAP_STRENGTH = -8.0
MAX_FALL_SPEED = 12.0

PIPE_WIDTH = 60
PIPE_GAP = 180
PIPE_SPEED = 3.0
PIPE_SPAWN_INTERVAL = 90
PIPE_GAP_Y_MIN = 150
PIPE_GAP_Y_MAX = GROUND_Y - 150

MOVING_PIPE_AMPLITUDE = 40
MOVING_PIPE_FREQUENCY = 0.02

WIND_STRENGTH_RANGE = (-0.15, 0.15)
WIND_CHANGE_INTERVAL = 120

GRAVITY_RANGE = (0.35, 0.65)
GRAVITY_CHANGE_INTERVAL = 180

POPULATION_SIZE = 100
MUTATION_RATE = 0.15
MUTATION_STRENGTH = 0.3
ELITISM_COUNT = 8
TOURNAMENT_K = 5
NETWORK_TOPOLOGY = [6, 12, 1]

COLORS = {
    "sky": (30, 30, 46),
    "ground": (49, 50, 68),
    "ground_line": (88, 91, 112),
    "pipe": (137, 180, 130),
    "pipe_border": (108, 153, 99),
    "pipe_cap": (116, 163, 108),
    "bird_alive": (250, 179, 135),
    "bird_best": (245, 224, 130),
    "bird_dead": (88, 91, 112),
    "hud_bg": (24, 24, 37, 200),
    "hud_text": (205, 214, 244),
    "hud_accent": (137, 180, 130),
    "hud_warn": (243, 139, 168),
    "graph_line": (137, 180, 130),
    "graph_avg": (180, 190, 254),
}
