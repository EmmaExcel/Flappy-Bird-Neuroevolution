import asyncio
import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
classic_dir = os.path.join(root_dir, "flappy_classic")

sys.path.insert(0, root_dir)
sys.path.insert(0, classic_dir)

from storage import load_model_json

os.chdir(classic_dir)

import pygame
from src.flappy import Flappy
from src.entities import PlayerMode


class AIClassicFlappy(Flappy):
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(root_dir, "best_bird.json")
        elif not os.path.isabs(model_path):
            candidate_root = os.path.join(root_dir, model_path)
            if os.path.exists(candidate_root):
                model_path = candidate_root
            else:
                model_path = os.path.abspath(model_path)

        super().__init__()
        self.network, self.model_info = load_model_json(model_path)
        print(f"Loaded Champion: Gen {self.model_info.get('generation')}, Score {self.model_info.get('score')}")

    async def splash(self):
        self.player.set_mode(PlayerMode.SHM)
        self.background.tick()
        self.floor.tick()
        self.player.tick()
        self.welcome_message.tick()
        pygame.display.update()
        await asyncio.sleep(0.1)
        return

    def get_ai_decision(self):
        next_pipe = None
        for pipe in self.pipes.upper:
            if pipe.x + pipe.w > self.player.x:
                next_pipe = pipe
                break

        if next_pipe is None:
            gap_center_y = self.config.window.viewport_height * 0.5
            dist_x = self.config.window.width
        else:
            gap_y = next_pipe.y + next_pipe.h
            gap_center_y = gap_y + (self.pipes.pipe_gap / 2.0)
            dist_x = next_pipe.x - self.player.x

        vh = self.config.window.viewport_height
        vw = self.config.window.width

        inputs = [
            self.player.y / vh,
            self.player.vel_y / 10.0,
            gap_center_y / vh,
            dist_x / vw,
            0.0,
            0.77,
        ]

        output = self.network.forward(inputs)
        return output[0] > 0.5

    async def play(self):
        self.score.reset()
        self.player.set_mode(PlayerMode.NORMAL)

        while True:
            if self.player.collided(self.pipes, self.floor):
                return

            for pipe in self.pipes.upper:
                if self.player.crossed(pipe):
                    self.score.add()

            for event in pygame.event.get():
                self.check_quit_event(event)

            if self.get_ai_decision():
                self.player.flap()

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

    async def game_over(self):
        self.player.set_mode(PlayerMode.CRASH)
        self.pipes.stop()
        self.floor.stop()
        for _ in range(15):
            for event in pygame.event.get():
                self.check_quit_event(event)
            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()
            self.game_over_message.tick()
            self.config.tick()
            pygame.display.update()
            await asyncio.sleep(0)
        return


if __name__ == "__main__":
    model_arg = sys.argv[1] if len(sys.argv) > 1 else None
    game = AIClassicFlappy(model_arg)
    asyncio.run(game.start())
