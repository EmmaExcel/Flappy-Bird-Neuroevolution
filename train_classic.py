import argparse
import json
import os
import random
import sys
import time
import numpy as np

from network import Network
from storage import save_model_json


def parse_args():
    parser = argparse.ArgumentParser(description="High-speed classic Flappy Bird trainer")
    parser.add_argument("--gens", type=int, default=300, help="Generations to train")
    parser.add_argument("--pop-size", type=int, default=100, help="Population size")
    parser.add_argument("--max-pipes", type=int, default=100, help="Max pipes per generation cap")
    parser.add_argument("--json-out", type=str, default="best_classic.json", help="Output model path")
    return parser.parse_args()


class ClassicPopulation:
    def __init__(self, size=100, topology=None):
        if topology is None:
            topology = [4, 8, 1]
        self.size = size
        self.topology = list(topology)
        self.networks = [Network(self.topology) for _ in range(size)]

    def evolve(self, fitnesses):
        sorted_indices = np.argsort(fitnesses)[::-1]
        new_networks = []

        elitism_count = max(4, int(self.size * 0.1))
        for i in range(elitism_count):
            new_networks.append(self.networks[sorted_indices[i]].clone())

        while len(new_networks) < self.size:
            p1 = self._tournament(fitnesses)
            p2 = self._tournament(fitnesses)
            child = self._crossover(p1, p2)
            self._mutate(child)
            new_networks.append(child)

        self.networks = new_networks

    def _tournament(self, fitnesses, k=5):
        sample = random.sample(range(len(fitnesses)), min(k, len(fitnesses)))
        winner = max(sample, key=lambda i: fitnesses[i])
        return self.networks[winner]

    def _crossover(self, parent1, parent2):
        child = Network(self.topology)
        p1 = parent1.get_params()
        p2 = parent2.get_params()
        mask = np.random.random(len(p1)) < 0.5
        child.set_params(np.where(mask, p1, p2))
        return child

    def _mutate(self, net, rate=0.15, scale=0.3):
        params = net.get_params()
        mask = np.random.random(len(params)) < rate
        n = int(mask.sum())
        if n > 0:
            params[mask] += np.random.randn(n) * scale
        net.set_params(params)


def run_training():
    args = parse_args()
    pop = ClassicPopulation(size=args.pop_size, topology=[4, 8, 1])

    all_time_best_score = 0
    all_time_best_fit = 0.0
    start_time = time.time()

    print(f"Target: {args.json_out} | Population: {args.pop_size} | Max Pipes Cap: {args.max_pipes}")
    print("-" * 68)

    for gen in range(1, args.gens + 1):
        pipes = [{"x": 298.0, "gap_y": random.randint(80, 202), "id": 0}]
        next_id = 1

        birds_y = np.full(args.pop_size, 244.0)
        birds_vel = np.zeros(args.pop_size)
        alive = np.ones(args.pop_size, dtype=bool)
        scores = np.zeros(args.pop_size, dtype=int)
        distance = np.zeros(args.pop_size, dtype=int)
        closest_gap_dev = np.zeros(args.pop_size, dtype=float)

        frame = 0
        max_frames = args.max_pipes * 45 + 200

        while np.any(alive) and frame < max_frames:
            frame += 1

            next_pipe = None
            for p in pipes:
                if p["x"] + 52.0 > 57.0:
                    next_pipe = p
                    break

            if next_pipe is None:
                gap_center = 202.0
                dist_x = 288.0
            else:
                gap_center = next_pipe["gap_y"] + 60.0
                dist_x = next_pipe["x"] - 57.0

            active = np.where(alive)[0]
            if len(active) == 0:
                break

            in_mat = np.column_stack([
                birds_y[active] / 404.0,
                birds_vel[active] / 10.0,
                np.full(len(active), gap_center / 404.0),
                np.full(len(active), dist_x / 288.0),
            ])

            for idx_local, bird_idx in enumerate(active):
                out = pop.networks[bird_idx].forward(in_mat[idx_local])
                if out[0] > 0.5:
                    birds_vel[bird_idx] = -9.0

            birds_vel[active] = np.minimum(10.0, birds_vel[active] + 1.0)
            birds_y[active] += birds_vel[active]
            distance[active] += 1

            gap_dev = np.abs(birds_y[active] - gap_center) / 404.0
            closest_gap_dev[active] += gap_dev

            for p in pipes:
                p["x"] -= 5.0

            last = pipes[-1]
            if 288.0 - (last["x"] + 52.0) > 130.0:
                pipes.append({"x": 288.0 + 10.0, "gap_y": random.randint(80, 202), "id": next_id})
                next_id += 1

            for p in pipes:
                if not p.get("scored") and p["x"] + 52.0 < 57.0:
                    p["scored"] = True
                    scores[active] += 1

            pipes = [p for p in pipes if p["x"] > -60]

            dead_floor = (birds_y[active] + 22.0 >= 404.0) | (birds_y[active] <= -20.0)
            for idx_local, is_dead in enumerate(dead_floor):
                if is_dead:
                    alive[active[idx_local]] = False

            active = np.where(alive)[0]
            if len(active) > 0 and next_pipe is not None:
                px = next_pipe["x"]
                py = next_pipe["gap_y"]
                if 57.0 + 32.0 > px + 2.0 and 57.0 + 2.0 < px + 50.0:
                    dead_pipe = (birds_y[active] + 2.0 < py) | (birds_y[active] + 22.0 > py + 120.0)
                    for idx_local, is_dead in enumerate(dead_pipe):
                        if is_dead:
                            alive[active[idx_local]] = False

            if np.max(scores) >= args.max_pipes:
                break

        fitnesses = np.zeros(args.pop_size, dtype=float)
        for i in range(args.pop_size):
            fit = scores[i] * 10000.0 + distance[i] * 2.0
            avg_dev = closest_gap_dev[i] / max(distance[i], 1)
            fit += max(0.0, (1.0 - avg_dev) * 500.0)
            fitnesses[i] = fit

        best_score = int(np.max(scores))
        best_fit = float(np.max(fitnesses))
        avg_fit = float(np.mean(fitnesses))
        best_idx = int(np.argmax(fitnesses))
        best_net = pop.networks[best_idx]

        record_flag = ""
        if best_score > all_time_best_score or (best_score == all_time_best_score and best_fit > all_time_best_fit):
            all_time_best_score = best_score
            all_time_best_fit = best_fit
            save_model_json(args.json_out, best_net, generation=gen, score=best_score, fitness=best_fit)
            record_flag = " *"

        elapsed = time.time() - start_time
        gen_rate = gen / max(elapsed, 0.001)

        print(f"Gen {gen:4d} | Best: {best_score:3d} pipes | Fit: {best_fit:9.0f} | Avg: {avg_fit:8.0f} | {gen_rate:4.1f} gen/s{record_flag}")

        pop.evolve(fitnesses)

        if best_score >= args.max_pipes:
            print(f"\nTarget of {args.max_pipes} pipes reached at generation {gen}!")
            break

    print(f"\nTraining completed in {time.time() - start_time:.1f}s. Saved to {args.json_out}")


if __name__ == "__main__":
    run_training()
