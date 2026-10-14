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
    parser = argparse.ArgumentParser(description="Master neuroevolution trainer for Flappy Bird")
    parser.add_argument("--gens", type=int, default=1500, help="Generations to train")
    parser.add_argument("--pop-size", type=int, default=120, help="Population size")
    parser.add_argument("--eval-pipes", type=int, default=40, help="Pipes per generation course cap")
    parser.add_argument("--log-interval", type=int, default=25, help="Print progress every N generations")
    parser.add_argument("--json-out", type=str, default="best_classic.json", help="Output model path")
    parser.add_argument("--time-limit", type=int, default=0, help="Time limit in seconds (0 for unlimited)")
    return parser.parse_args()


class MasterPopulation:
    def __init__(self, size=120, hidden_dim=16):
        self.size = size
        self.topology = [4, hidden_dim, 1]
        self.hidden_dim = hidden_dim

        self.w1 = np.random.randn(size, 4, hidden_dim) * np.sqrt(2.0 / 4)
        self.b1 = np.zeros((size, 1, hidden_dim))
        self.w2 = np.random.randn(size, hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((size, 1, 1))

    def get_individual_params(self, i):
        return np.concatenate([
            self.w1[i].flatten(),
            self.w2[i].flatten(),
            self.b1[i].flatten(),
            self.b2[i].flatten()
        ])

    def set_individual_params(self, i, params):
        p0 = 4 * self.hidden_dim
        p1 = p0 + self.hidden_dim * 1
        p2 = p1 + self.hidden_dim
        p3 = p2 + 1

        self.w1[i] = params[0:p0].reshape(4, self.hidden_dim)
        self.w2[i] = params[p0:p1].reshape(self.hidden_dim, 1)
        self.b1[i] = params[p1:p2].reshape(1, self.hidden_dim)
        self.b2[i] = params[p2:p3].reshape(1, 1)

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
        elitism_count = max(8, int(self.size * 0.08))
        for i in range(elitism_count):
            new_params.append(all_params[sorted_indices[i]].copy())

        while len(new_params) < self.size - 4:
            sample1 = random.sample(range(self.size), 5)
            p1_idx = max(sample1, key=lambda i: fitnesses[i])
            sample2 = random.sample(range(self.size), 5)
            p2_idx = max(sample2, key=lambda i: fitnesses[i])

            p1 = all_params[p1_idx]
            p2 = all_params[p2_idx]

            mask = np.random.random(len(p1)) < 0.5
            child = np.where(mask, p1, p2)

            alpha = np.random.uniform(-0.1, 0.1, size=len(p1))
            child += alpha * (p1 - p2)

            mut_chance = np.random.random()
            if mut_chance < 0.80:
                mut_mask = np.random.random(len(child)) < 0.10
                n_mut = int(mut_mask.sum())
                if n_mut > 0:
                    child[mut_mask] += np.random.randn(n_mut) * 0.06
            else:
                mut_mask = np.random.random(len(child)) < 0.20
                n_mut = int(mut_mask.sum())
                if n_mut > 0:
                    child[mut_mask] += np.random.randn(n_mut) * 0.25

            new_params.append(child)

        while len(new_params) < self.size:
            rand_child = np.random.randn(len(all_params[0])) * 0.5
            new_params.append(rand_child)

        for i in range(self.size):
            self.set_individual_params(i, new_params[i])

    def export_network(self, idx):
        net = Network(self.topology)
        net.set_params(self.get_individual_params(idx))
        return net


def simulate_generation(pop, eval_pipes, course_mode=0):
    pipes = [
        {"x": 444.0, "gap_y": _get_gap_height(0, course_mode), "id": 0},
        {"x": 626.0, "gap_y": _get_gap_height(1, course_mode), "id": 1}
    ]
    next_id = 2

    birds_y = np.full(pop.size, 244.0)
    birds_vel = np.full(pop.size, -9.0)
    alive = np.ones(pop.size, dtype=bool)
    scores = np.zeros(pop.size, dtype=int)
    distance = np.zeros(pop.size, dtype=int)
    gap_deviations = np.zeros(pop.size, dtype=float)

    frame = 0
    max_frames = eval_pipes * 45 + 180

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
        non_flaps = active[~flap_mask]
        birds_vel[non_flaps] = np.minimum(10.0, birds_vel[non_flaps] + 1.0)

        birds_y[active] += birds_vel[active]
        distance[active] += 1

        norm_dev = np.abs(birds_y[active] - gap_center) / 404.0
        gap_deviations[active] += norm_dev

        for p in pipes:
            p["x"] -= 5.0

        last = pipes[-1]
        if 288.0 - (last["x"] + 52.0) > 130.0:
            pipes.append({"x": 298.0, "gap_y": _get_gap_height(next_id, course_mode), "id": next_id})
            next_id += 1

        for p in pipes:
            if not p.get("scored") and p["x"] + 52.0 < 57.0:
                p["scored"] = True
                scores[active] += 1

        pipes = [p for p in pipes if p["x"] > -60]

        dead_floor = (birds_y[active] + 24.0 >= 404.0) | (birds_y[active] <= -40.0)
        for idx_local, is_dead in enumerate(dead_floor):
            if is_dead:
                alive[active[idx_local]] = False

        active = np.where(alive)[0]
        if len(active) > 0 and next_pipe is not None:
            px = next_pipe["x"]
            py = next_pipe["gap_y"]
            if 57.0 + 31.0 > px and 57.0 + 3.0 < px + 52.0:
                dead_pipe = (birds_y[active] + 3.0 < py) | (birds_y[active] + 21.0 > py + 120.0)
                for idx_local, is_dead in enumerate(dead_pipe):
                    if is_dead:
                        alive[active[idx_local]] = False

        if np.max(scores) >= eval_pipes:
            break

    fitnesses = np.zeros(pop.size, dtype=float)
    for i in range(pop.size):
        fit = scores[i] * 20000.0 + distance[i] * 2.0
        avg_dev = gap_deviations[i] / max(distance[i], 1)
        fit += max(0.0, (1.0 - (avg_dev ** 1.5)) * 1000.0)
        fitnesses[i] = fit

    return scores, fitnesses


def _get_gap_height(pipe_id, course_mode):
    if course_mode == 0:
        return random.randint(80, 201)
    elif course_mode == 1:
        return 80 if pipe_id % 2 == 1 else 201
    elif course_mode == 2:
        return random.randint(80, 120)
    elif course_mode == 3:
        return random.randint(165, 201)
    else:
        return 80 + (pipe_id * 25) % 120


def run_benchmark_validation(net, num_pipes_per_track=100):
    def test_single_track(gap_fn):
        pipes = [
            {"x": 444.0, "gap_y": gap_fn(0)},
            {"x": 626.0, "gap_y": gap_fn(1)}
        ]
        next_id = 2
        y = 244.0
        vel_y = -9.0
        score = 0
        devs = []

        max_frames = num_pipes_per_track * 45 + 200
        for _ in range(max_frames):
            next_p = None
            for p in pipes:
                if p["x"] + 52.0 > 57.0:
                    next_p = p
                    break

            if next_p is None:
                gap_c = 202.0
                dx = 288.0
            else:
                gap_c = next_p["gap_y"] + 60.0
                dx = next_p["x"] - 57.0

            out = net.forward([y / 404.0, vel_y / 10.0, gap_c / 404.0, dx / 288.0])[0]
            if out > 0.5:
                vel_y = -9.0
            else:
                vel_y = min(10.0, vel_y + 1.0)
            y += vel_y
            devs.append(abs(y - gap_c))

            for p in pipes:
                p["x"] -= 5.0

            if 288.0 - (pipes[-1]["x"] + 52.0) > 130.0:
                pipes.append({"x": 298.0, "gap_y": gap_fn(next_id)})
                next_id += 1

            for p in pipes:
                if not p.get("scored") and p["x"] + 52.0 < 57.0:
                    p["scored"] = True
                    score += 1

            pipes = [p for p in pipes if p["x"] > -60]

            if y + 24.0 >= 404.0 or y <= -40.0:
                return score, float(np.mean(devs))
            if next_p is not None and 57.0 + 31.0 > next_p["x"] and 57.0 + 3.0 < next_p["x"] + 52.0:
                if y + 3.0 < next_p["gap_y"] or y + 21.0 > next_p["gap_y"] + 120.0:
                    return score, float(np.mean(devs))

            if score >= num_pipes_per_track:
                break

        return score, float(np.mean(devs))

    rng = random.Random(1337)
    random_heights = [rng.randint(80, 201) for _ in range(num_pipes_per_track + 10)]

    s1, d1 = test_single_track(lambda i: random_heights[i % len(random_heights)])
    s2, d2 = test_single_track(lambda i: 80 if i % 2 == 1 else 201)
    s3, d3 = test_single_track(lambda i: 80 + (i * 17) % 40)
    s4, d4 = test_single_track(lambda i: 165 + (i * 13) % 36)
    s5, d5 = test_single_track(lambda i: 80 if (i // 3) % 2 == 0 else 201)

    total_score = s1 + s2 + s3 + s4 + s5
    avg_dev = (d1 + d2 + d3 + d4 + d5) / 5.0
    return total_score, avg_dev, [s1, s2, s3, s4, s5]


def verify_live_flappy_classic(net, num_pipes=1000):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    classic_dir = os.path.join(root_dir, "flappy_classic")
    current_cwd = os.getcwd()

    try:
        os.chdir(classic_dir)
        sys.path.insert(0, classic_dir)
        sys.path.insert(0, root_dir)

        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

        import pygame
        pygame.init()
        pygame.display.set_mode((288, 512))

        from src.flappy import Flappy
        from src.entities import Background, Floor, GameOver, Pipes, Player, Score, WelcomeMessage, PlayerMode

        flappy = Flappy()
        flappy.background = Background(flappy.config)
        flappy.floor = Floor(flappy.config)
        flappy.player = Player(flappy.config)
        flappy.welcome_message = WelcomeMessage(flappy.config)
        flappy.game_over_message = GameOver(flappy.config)
        flappy.pipes = Pipes(flappy.config)
        flappy.score = Score(flappy.config)
        flappy.player.set_mode(PlayerMode.NORMAL)
        flappy.config.tick = lambda: None

        passed = 0
        max_steps = num_pipes * 45 + 500

        for _ in range(max_steps):
            if flappy.player.collided(flappy.pipes, flappy.floor):
                return False, passed

            for p in flappy.pipes.upper:
                if flappy.player.crossed(p):
                    passed += 1

            if passed >= num_pipes:
                return True, passed

            next_pipe = None
            for pipe in flappy.pipes.upper:
                if pipe.x + pipe.w > flappy.player.x:
                    next_pipe = pipe
                    break

            if next_pipe is None:
                gap_center_y = 202.0
                dist_x = 288.0
            else:
                gap_y = next_pipe.y + next_pipe.h
                gap_center_y = gap_y + 60.0
                dist_x = next_pipe.x - flappy.player.x

            inputs = [
                flappy.player.y / 404.0,
                flappy.player.vel_y / 10.0,
                gap_center_y / 404.0,
                dist_x / 288.0,
            ]
            out = net.forward(inputs)[0]
            if out > 0.5:
                flappy.player.flap()

            flappy.floor.tick()
            flappy.pipes.tick()
            flappy.player.tick()

        return passed >= num_pipes, passed
    finally:
        os.chdir(current_cwd)


def format_duration(seconds):
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins}m {secs:02d}s"


def train_master():
    args = parse_args()
    pop = MasterPopulation(size=args.pop_size, hidden_dim=16)

    best_benchmark_score = 0
    best_benchmark_dev = 999.0
    start_time = time.time()

    print(f"Master Curriculum Training | Population: {args.pop_size} | Architecture: [4, 16, 1]")
    print(f"Goal: {args.gens} generations | 5-Track Benchmark Validation Every {args.log_interval} gens")
    print("-" * 88)

    champion_saved = False
    for gen in range(1, args.gens + 1):
        course_mode = (gen % 5)
        scores, fitnesses = simulate_generation(pop, args.eval_pipes, course_mode=course_mode)

        best_score = int(np.max(scores))
        best_idx = int(np.argmax(fitnesses))

        elapsed = time.time() - start_time
        gen_rate = gen / max(elapsed, 0.001)

        if gen % args.log_interval == 0 or gen == 1:
            candidate_net = pop.export_network(best_idx)
            bench_score, bench_dev, track_scores = run_benchmark_validation(candidate_net, num_pipes_per_track=100)

            record_flag = ""
            if bench_score > best_benchmark_score or (bench_score == 500 and bench_dev < best_benchmark_dev):
                best_benchmark_score = bench_score
                best_benchmark_dev = bench_dev

                if bench_score == 500:
                    verified, live_score = verify_live_flappy_classic(candidate_net, num_pipes=1000)
                    if verified:
                        save_model_json(
                            args.json_out,
                            candidate_net,
                            generation=gen,
                            score=500,
                            fitness=round(1000000.0 - bench_dev * 1000.0, 1)
                        )
                        record_flag = f" [CERTIFIED MASTER CHAMPION: {live_score} PIPES]"
                        champion_saved = True
                    else:
                        record_flag = f" [BENCHMARK 500 | LIVE FLAPPY: {live_score}]"
                else:
                    save_model_json(
                        args.json_out,
                        candidate_net,
                        generation=gen,
                        score=bench_score,
                        fitness=round(bench_score * 1000.0 - bench_dev * 10.0, 1)
                    )
                    record_flag = " [NEW BENCHMARK LEADER]"

            pct = min(100.0, (gen / args.gens) * 100.0)

            print(
                f"Gen {gen:4d}/{args.gens} ({pct:4.1f}%) | "
                f"Curriculum Score: {best_score:2d}/{args.eval_pipes} | "
                f"Benchmark: {bench_score}/500 pipes (dev: {bench_dev:.1f}px) | "
                f"{gen_rate:4.1f} gen/s | Elapsed: {format_duration(elapsed)}{record_flag}"
            )

        pop.evolve(fitnesses)

        if args.time_limit > 0 and elapsed >= args.time_limit:
            print(f"\nTime limit reached after {gen} generations.")
            break

        if champion_saved and gen >= 300:
            print(f"\nTarget achieved! Master Champion certified with 1000-pipe clearance in live game.")
            break

    final_net, final_info = load_model_json(args.json_out)
    print("\n" + "=" * 88)
    print("MASTER TRAINING COMPLETE")
    print(f"Champion Saved to: {args.json_out}")
    print(f"Certified Benchmark Score: {final_info.get('score')}/500 across all 5 obstacle courses")
    print(f"Total Generations: {gen} | Total Time: {format_duration(time.time() - start_time)}")
    print("=" * 88)


if __name__ == "__main__":
    train_master()
