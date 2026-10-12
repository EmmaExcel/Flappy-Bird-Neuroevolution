import os
import sys
import argparse
import time

os.environ["SDL_VIDEODRIVER"] = "dummy"

from config import (
    NETWORK_TOPOLOGY,
    POPULATION_SIZE,
)
from game import Game
from population import Population
from storage import Database, save_model_json, load_model_json


def parse_args():
    parser = argparse.ArgumentParser(description="Headless training runner")
    parser.add_argument("--gens", type=int, default=0, help="Max generations (0 for unlimited)")
    parser.add_argument("--pop-size", type=int, default=POPULATION_SIZE, help="Population size")
    parser.add_argument("--db", type=str, default="evolution.db", help="Database file path")
    parser.add_argument("--json-out", type=str, default="best_bird.json", help="Output path for best model")
    parser.add_argument("--resume-json", type=str, default=None, help="Input model path to seed population")
    parser.add_argument("--no-wind", action="store_true", help="Disable wind")
    parser.add_argument("--no-moving", action="store_true", help="Disable moving pipes")
    parser.add_argument("--no-vargrav", action="store_true", help="Disable variable gravity")
    return parser.parse_args()


def train():
    args = parse_args()
    db = Database(args.db)

    pop = Population(args.pop_size, NETWORK_TOPOLOGY)

    if args.resume_json and os.path.exists(args.resume_json):
        seed_net, info = load_model_json(args.resume_json)
        pop.networks[0] = seed_net
        print(f"Resumed from {args.resume_json} (Score: {info.get('score', 0)})")
    else:
        champion_net, info = db.get_champion()
        if champion_net:
            pop.networks[0] = champion_net
            print(f"Loaded champion from {args.db} (Gen {info['generation']}, Score {info['score']})")

    gen = 0
    all_time_best_score = 0
    all_time_best_fitness = 0.0
    start_time = time.time()

    print(f"Database: {args.db} | Output: {args.json_out}")

    try:
        while True:
            gen += 1
            game = Game(
                pop.networks,
                enable_wind=not args.no_wind,
                enable_variable_gravity=not args.no_vargrav,
                enable_moving_pipes=not args.no_moving,
            )

            while game.step():
                pass

            fitnesses = pop.evaluate_fitness(game.birds)
            best_idx = 0
            best_val = -1.0
            for i, f in enumerate(fitnesses):
                if f > best_val:
                    best_val = f
                    best_idx = i

            best_bird = game.birds[best_idx]
            best_score = game.get_best_score()
            best_fitness = float(max(fitnesses))
            avg_fitness = float(sum(fitnesses) / len(fitnesses))

            pop.best_score_history.append(best_score)
            db.record_generation(gen, best_score, best_fitness, avg_fitness, best_bird.network)

            if best_fitness > all_time_best_fitness or best_score > all_time_best_score:
                if best_score > all_time_best_score:
                    all_time_best_score = best_score
                if best_fitness > all_time_best_fitness:
                    all_time_best_fitness = best_fitness
                save_model_json(args.json_out, best_bird.network, generation=gen, score=best_score, fitness=best_fitness)
                record_flag = " *"
            else:
                record_flag = ""

            elapsed = time.time() - start_time
            gens_per_sec = gen / max(elapsed, 0.001)

            print(f"Gen {gen:5d} | Best: {best_score:3d} | Fit: {best_fitness:8.0f} | Avg: {avg_fitness:7.0f} | {gens_per_sec:4.1f} gen/s{record_flag}")

            pop.evolve(fitnesses)

            if args.gens > 0 and gen >= args.gens:
                break

    except KeyboardInterrupt:
        print("\nStopped.")

    print(f"Done. Saved to {args.json_out} and {args.db}")


if __name__ == "__main__":
    train()
