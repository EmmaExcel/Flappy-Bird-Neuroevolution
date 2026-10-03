from bird import Bird
from pipe import Pipe
from environment import Environment
from config import (
    BIRD_X,
    GROUND_Y,
    PIPE_SPEED,
    PIPE_SPAWN_INTERVAL,
    PIPE_WIDTH,
    SCREEN_WIDTH,
)


class Game:
    def __init__(self, networks, enable_wind=True, enable_variable_gravity=True,
                 enable_moving_pipes=True):
        self.birds = [Bird(net) for net in networks]
        self.pipes = []
        self.environment = Environment(enable_wind, enable_variable_gravity)
        self.enable_moving_pipes = enable_moving_pipes
        self.frame = 0
        self.spawn_timer = PIPE_SPAWN_INTERVAL - 30
        self._pipe_id_counter = 0

    def step(self):
        self.frame += 1
        self.environment.update()

        self._spawn_pipes()
        self._update_pipes()
        self._remove_off_screen_pipes()

        next_pipe = self._get_next_pipe()
        if next_pipe is None:
            next_pipe = self._make_dummy_pipe()

        alive_count = 0
        for bird in self.birds:
            if not bird.alive:
                continue

            inputs = bird.get_inputs(
                next_pipe, self.environment.wind, self.environment.gravity
            )
            bird.decide(inputs)
            bird.update(self.environment.gravity, self.environment.wind)

            if bird.is_out_of_bounds(GROUND_Y):
                bird.alive = False
                continue

            for pipe in self.pipes:
                if pipe.collides(bird.x, bird.y):
                    bird.alive = False
                    break

            if not bird.alive:
                continue

            for pipe in self.pipes:
                if pipe.pipe_id not in bird.scored_pipes and pipe.x + PIPE_WIDTH < bird.x:
                    bird.scored_pipes.add(pipe.pipe_id)
                    bird.score += 1

            dx = next_pipe.x - bird.x
            if dx > 0:
                progress = 1.0 - (dx / SCREEN_WIDTH)
            else:
                progress = 1.0
            total_progress = bird.score + progress
            if total_progress > bird.max_pipe_progress:
                bird.max_pipe_progress = total_progress

            alive_count += 1

        return alive_count > 0

    def _spawn_pipes(self):
        self.spawn_timer += 1
        if self.spawn_timer >= PIPE_SPAWN_INTERVAL:
            self.spawn_timer = 0
            pipe = Pipe(SCREEN_WIDTH + 20, moving=self.enable_moving_pipes)
            pipe.pipe_id = self._pipe_id_counter
            self._pipe_id_counter += 1
            self.pipes.append(pipe)

    def _update_pipes(self):
        speed = PIPE_SPEED + self.environment.wind
        speed = max(0.5, speed)
        for pipe in self.pipes:
            pipe.update(speed, self.frame)

    def _remove_off_screen_pipes(self):
        self.pipes = [p for p in self.pipes if not p.off_screen]

    def _get_next_pipe(self):
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH > BIRD_X:
                return pipe
        return None

    def _make_dummy_pipe(self):
        dummy = Pipe(SCREEN_WIDTH, gap_center_y=GROUND_Y / 2, moving=False)
        dummy.pipe_id = -1
        return dummy

    def get_best_bird_index(self):
        best_idx = 0
        best_val = -1
        for i, bird in enumerate(self.birds):
            if bird.alive:
                val = bird.score * 1000 + bird.distance
                if val > best_val:
                    best_val = val
                    best_idx = i
        return best_idx

    def get_alive_count(self):
        return sum(1 for b in self.birds if b.alive)

    def get_best_score(self):
        return max(b.score for b in self.birds)
