import sys
import argparse
import json
import os

import pygame

from config import (
    COLORS,
    FPS,
    POPULATION_SIZE,
    NETWORK_TOPOLOGY,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from game import Game
from hud import HUD
from population import Population
from renderer import Renderer
from neuroevolution import run_comparison
from storage import Database, save_model_json, load_model_json


def parse_args():
    parser = argparse.ArgumentParser(description="Flappy Bird Neuroevolution")
    parser.add_argument("--no-wind", action="store_true", help="Disable wind")
    parser.add_argument("--no-moving", action="store_true", help="Disable moving pipes")
    parser.add_argument("--no-vargrav", action="store_true", help="Disable variable gravity")
    parser.add_argument("--speed", type=int, default=1, help="Simulation speed multiplier")
    parser.add_argument(
        "--topology", type=int, nargs="+", default=None,
        help="Network topology"
    )
    parser.add_argument(
        "--compare-topologies", action="store_true",
        help="Run topology benchmark"
    )
    parser.add_argument(
        "--compare-gens", type=int, default=50,
        help="Generations per topology in benchmark"
    )
    parser.add_argument(
        "--play", type=str, default=None,
        help="Model json file path to load"
    )
    parser.add_argument(
        "--play-champion", action="store_true",
        help="Load champion from database"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Seed initial population from json file"
    )
    parser.add_argument(
        "--db", type=str, default="evolution.db",
        help="Database file path"
    )
    return parser.parse_args()


def run_play(network, info, args):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird")
    clock = pygame.time.Clock()

    renderer = Renderer(screen)
    hud = HUD(screen)

    enable_wind = not args.no_wind
    enable_vargrav = not args.no_vargrav
    enable_moving = not args.no_moving
    speed_mult = max(1, min(20, args.speed))

    high_score = 0
    attempts = 0
    running = True

    while running:
        attempts += 1
        game = Game(
            [network.clone()],
            enable_wind=enable_wind,
            enable_variable_gravity=enable_vargrav,
            enable_moving_pipes=enable_moving,
        )

        game_running = True
        while game_running and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    if event.key == pygame.K_UP:
                        speed_mult = min(20, speed_mult + 1)
                    if event.key == pygame.K_DOWN:
                        speed_mult = max(1, speed_mult - 1)

            if not running:
                break

            for _ in range(speed_mult):
                if not game.step():
                    game_running = False
                    break

            current_score = game.get_best_score()
            if current_score > high_score:
                high_score = current_score

            renderer.draw(game, 0)
            hud.draw(
                generation=attempts,
                alive=game.get_alive_count(),
                best_score=current_score,
                wind=game.environment.wind,
                gravity=game.environment.gravity,
                best_fitness_history=[high_score * 5000],
                avg_fitness_history=[current_score * 5000],
                speed_mult=speed_mult,
            )

            pygame.display.flip()
            clock.tick(FPS)

    pygame.quit()


def run_visual(args):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird Neuroevolution")
    clock = pygame.time.Clock()

    topology = args.topology if args.topology else NETWORK_TOPOLOGY
    enable_wind = not args.no_wind
    enable_vargrav = not args.no_vargrav
    enable_moving = not args.no_moving
    speed_mult = max(1, min(20, args.speed))

    db = Database(args.db)
    population = Population(POPULATION_SIZE, topology)

    if args.resume and os.path.exists(args.resume):
        seed_net, _ = load_model_json(args.resume)
        population.networks[0] = seed_net

    renderer = Renderer(screen)
    hud = HUD(screen)

    running = True

    while running:
        game = Game(
            population.networks,
            enable_wind=enable_wind,
            enable_variable_gravity=enable_vargrav,
            enable_moving_pipes=enable_moving,
        )

        generation_running = True

        while generation_running and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    if event.key == pygame.K_UP:
                        speed_mult = min(20, speed_mult + 1)
                    if event.key == pygame.K_DOWN:
                        speed_mult = max(1, speed_mult - 1)

            if not running:
                break

            for _ in range(speed_mult):
                if not game.step():
                    generation_running = False
                    break

            best_idx = game.get_best_bird_index()
            renderer.draw(game, best_idx)
            hud.draw(
                generation=population.generation + 1,
                alive=game.get_alive_count(),
                best_score=game.get_best_score(),
                wind=game.environment.wind,
                gravity=game.environment.gravity,
                best_fitness_history=population.best_fitness_history,
                avg_fitness_history=population.avg_fitness_history,
                speed_mult=speed_mult,
            )

            pygame.display.flip()
            clock.tick(FPS)

        if running:
            fitnesses = population.evaluate_fitness(game.birds)
            best_score = game.get_best_score()
            population.best_score_history.append(best_score)
            best_fit = max(fitnesses)
            avg_fit = sum(fitnesses) / len(fitnesses)
            best_idx = game.get_best_bird_index()
            best_bird = game.birds[best_idx]

            db.record_generation(
                population.generation + 1,
                best_score,
                best_fit,
                avg_fit,
                best_bird.network
            )
            save_model_json(
                "best_bird.json",
                best_bird.network,
                generation=population.generation + 1,
                score=best_score,
                fitness=best_fit
            )

            print(f"Gen {population.generation + 1:4d} | Best: {best_score:2d} | Fit: {best_fit:8.0f} | Avg: {avg_fit:8.0f}")
            population.evolve(fitnesses)

    pygame.quit()


def main():
    args = parse_args()

    if args.play:
        if not os.path.exists(args.play):
            print(f"File not found: {args.play}")
            sys.exit(1)
        net, info = load_model_json(args.play)
        print(f"Loaded {args.play} (Gen {info.get('generation', 0)}, Score {info.get('score', 0)})")
        run_play(net, info, args)
        return

    if args.play_champion:
        db = Database(args.db)
        net, info = db.get_champion()
        if not net:
            print(f"No champion found in {args.db}")
            sys.exit(1)
        print(f"Loaded champion from {args.db} (Gen {info['generation']}, Score {info['score']})")
        run_play(net, info, args)
        return

    if args.compare_topologies:
        topologies = [
            [6, 4, 1],
            [6, 8, 1],
            [6, 12, 1],
            [6, 16, 1],
        ]
        results = run_comparison(
            topologies,
            generations_per_topology=args.compare_gens,
            enable_wind=not args.no_wind,
            enable_variable_gravity=not args.no_vargrav,
            enable_moving_pipes=not args.no_moving,
        )
        with open("results.json", "w") as f:
            json.dump(
                {k: {kk: vv for kk, vv in v.items()
                     if kk not in ("best_fitness_history", "avg_fitness_history", "best_score_history")}
                 for k, v in results.items()},
                f, indent=2,
            )
        print("Saved to results.json")
    else:
        run_visual(args)


if __name__ == "__main__":
    main()
