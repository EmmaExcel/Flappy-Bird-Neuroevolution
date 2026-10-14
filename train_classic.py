import argparse
import json
import os
import random
import sys
import time
import numpy as np

from network import Network
from storage import save_model_json, load_model_json


def parse_args():
    parser = argparse.ArgumentParser(description="Vectorized classic Flappy Bird trainer")
    parser.add_argument("--gens", type=int, default=50000, help="Generations to train")
    parser.add_argument("--time-limit", type=int, default=1800, help="Time limit in seconds (1800 = 30 mins)")
    parser.add_argument("--pop-size", type=int, default=60, help="Population size")
    parser.add_argument("--eval-pipes", type=int, default=30, help="Pipes per generation evaluation cap")
    parser.add_argument("--log-interval", type=int, default=250, help="Print progress every N generations")
    parser.add_argument("--json-out", type=str, default="best_classic.json", help="Output model path")
    parser.add_argument("--resume", type=str, default="best_classic.json", help="Model file to seed population")
    return parser.parse_args()


class VectorizedPopulation:
    def __init__(self, size=60, seed_model=None):
        self.size = size
        self.topology = [4, 8, 1]
        self.w1 = np.random.randn(size, 4, 8) * np.sqrt(2.0 / 4)
        self.b1 = np.zeros((size, 1, 8))
        self.w2 = np.random.randn(size, 8, 1) * np.sqrt(2.0 / 8)
        self.b2 = np.zeros((size, 1, 1))

        if seed_model and os.path.exists(seed_model):
            try:
                seed_net, _ = load_model_json(seed_model)
                if seed_net.topology == self.topology:
                    p = seed_net.get_params()
                    self.set_individual_params(0, p)
                    for i in range(1, int(size * 0.35)):
                        mutated = p.copy()
                        mask = np.random.random(len(p)) < 0.2
                        mutated[mask] += np.random.randn(mask.sum()) * 0.2
                        self.set_individual_params(i, mutated)
                    print(f"Seeded population from {seed_model}")
            except Exception:
                pass

    def get_individual_params(self, i):
        return np.concatenate([
            self.w1[i].flatten(),
            self.b1[i].flatten(),
            self.w2[i].flatten(),
            self.b2[i].flatten()
        ])

    def set_individual_params(self, i, params):
        self.w1[i] = params[0:32].reshape(4, 8)
        self.b1[i] = params[32:40].reshape(1, 8)
        self.w2[i] = params[40:48].reshape(8, 1)
        self.b2[i] = params[48:49].reshape(1, 1)

    def forward_batch(self, active, X):
        W1 = self.w1[active]
        B1 = self.b1[active]
        W2 = self.w2[active]
        B2 = self.b2[active]
        H = np.maximum(0, np.matmul(X, W1) + B1)
        O = 1.0 / (1.0 + np.exp(-np.clip(np.matmul(H, W2) + B2, -10, 10)))
        return O.reshape(-1)

    def evolve(self, fitnesses):
        sorted_indices = np.argsort(fitnesses)[::-1]
        all_params = [self.get_individual_params(i) for i in range(self.size)]

        new_params = []
        elitism_count = max(4, int(self.size * 0.12))
        for i in range(elitism_count):
            new_params.append(all_params[sorted_indices[i]].copy())

        while len(new_params) < self.size:
            sample1 = random.sample(range(self.size), 5)
            p1_idx = max(sample1, key=lambda i: fitnesses[i])
            sample2 = random.sample(range(self.size), 5)
            p2_idx = max(sample2, key=lambda i: fitnesses[i])

            p1 = all_params[p1_idx]
            p2 = all_params[p2_idx]
            mask = np.random.random(len(p1)) < 0.5
            child = np.where(mask, p1, p2)

            mut_mask = np.random.random(len(child)) < 0.15
            n_mut = int(mut_mask.sum())
            if n_mut > 0:
                child[mut_mask] += np.random.randn(n_mut) * 0.25

            new_params.append(child)

        for i in range(self.size):
            self.set_individual_params(i, new_params[i])

    def export_champion(self, best_idx):
        net = Network(self.topology)
        net.set_params(self.get_individual_params(best_idx))
        return net


def format_duration(seconds):
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins}m {secs:02d}s"


def run_training():
    args = parse_args()
    pop = VectorizedPopulation(
        size=args.pop_size,
        seed_model=args.resume if args.resume else None
    )

    all_time_best_score = 0
    all_time_best_fit = 0.0
    start_time = time.time()

    print(f"Goal: {args.gens} generations | Max Time: {format_duration(args.time_limit)} | Target File: {args.json_out}")
    print(f"Population: {args.pop_size} | Evaluation Cap: {args.eval_pipes} pipes/gen | Log Interval: Every {args.log_interval} gens")
    print("-" * 88)

    gen = 0
    while True:
        gen += 1
        pipes = [{"x": 298.0, "gap_y": random.randint(80, 202), "id": 0}]
        next_id = 1

        birds_y = np.full(args.pop_size, 244.0)
        birds_vel = np.zeros(args.pop_size)
        alive = np.ones(args.pop_size, dtype=bool)
        scores = np.zeros(args.pop_size, dtype=int)
        distance = np.zeros(args.pop_size, dtype=int)
        gap_deviations = np.zeros(args.pop_size, dtype=float)

        frame = 0
        max_frames = args.eval_pipes * 45 + 100

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

            X = np.empty((len(active), 1, 4), dtype=np.float64)
            X[:, 0, 0] = birds_y[active] / 404.0
            X[:, 0, 1] = birds_vel[active] / 10.0
            X[:, 0, 2] = gap_center / 404.0
            X[:, 0, 3] = dist_x / 288.0

            outputs = pop.forward_batch(active, X)
            flap_mask = outputs > 0.5
            birds_vel[active[flap_mask]] = -9.0

            birds_vel[active] = np.minimum(10.0, birds_vel[active] + 1.0)
            birds_y[active] += birds_vel[active]
            distance[active] += 1
            gap_deviations[active] += np.abs(birds_y[active] - gap_center) / 404.0

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

            if np.max(scores) >= args.eval_pipes:
                break

        fitnesses = np.zeros(args.pop_size, dtype=float)
        for i in range(args.pop_size):
            fit = scores[i] * 10000.0 + distance[i] * 2.0
            avg_dev = gap_deviations[i] / max(distance[i], 1)
            fit += max(0.0, (1.0 - avg_dev) * 500.0)
            fitnesses[i] = fit

        best_score = int(np.max(scores))
        best_fit = float(np.max(fitnesses))
        avg_fit = float(np.mean(fitnesses))
        best_idx = int(np.argmax(fitnesses))

        record_flag = ""
        if best_fit > all_time_best_fit or best_score > all_time_best_score:
            all_time_best_score = max(all_time_best_score, best_score)
            all_time_best_fit = best_fit
            best_net = pop.export_champion(best_idx)
            save_model_json(args.json_out, best_net, generation=gen, score=all_time_best_score, fitness=best_fit)
            record_flag = " *"

        elapsed = time.time() - start_time
        gen_rate = gen / max(elapsed, 0.001)

        if gen % args.log_interval == 0 or record_flag or gen == 1:
            remaining_gens = max(0, args.gens - gen)
            eta_seconds = remaining_gens / max(gen_rate, 0.001)
            pct = min(100.0, (gen / args.gens) * 100.0)

            elapsed_str = format_duration(elapsed)
            eta_str = format_duration(eta_seconds)

            print(f"Gen {gen:5d}/{args.gens} ({pct:4.1f}%) | Best: {best_score:2d}/{args.eval_pipes} | Fit: {best_fit:8.0f} | Avg: {avg_fit:8.0f} | {gen_rate:4.1f} gen/s | Elapsed: {elapsed_str} | ETA: {eta_str}{record_flag}")

        pop.evolve(fitnesses)

        if gen >= args.gens:
            print(f"\nCompleted all {args.gens} generations!")
            break

        if args.time_limit > 0 and elapsed >= args.time_limit:
            print(f"\nTime limit of {format_duration(args.time_limit)} reached after {gen} generations!")
            break

    total_time = time.time() - start_time
    print(f"Total training time: {format_duration(total_time)} ({gen / total_time:.1f} gen/s). Champion saved to {args.json_out}")


if __name__ == "__main__":
    run_training()
