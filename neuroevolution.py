from population import Population
from game import Game


def run_comparison(topologies, generations_per_topology=50, population_size=100,
                   enable_wind=True, enable_variable_gravity=True,
                   enable_moving_pipes=True):
    results = {}

    for topo in topologies:
        topo_key = "x".join(str(t) for t in topo)
        print(f"\nTopology: {topo} ({_count_params(topo)} params)")

        pop = Population(population_size, topo)

        for gen in range(generations_per_topology):
            game = Game(
                pop.networks,
                enable_wind=enable_wind,
                enable_variable_gravity=enable_variable_gravity,
                enable_moving_pipes=enable_moving_pipes,
            )

            while game.step():
                pass

            fitnesses = pop.evaluate_fitness(game.birds)
            best_score = game.get_best_score()
            best_fit = max(fitnesses)
            avg_fit = sum(fitnesses) / len(fitnesses)
            pop.best_score_history.append(best_score)
            pop.evolve(fitnesses)

            if (gen + 1) % 10 == 0 or gen == 0:
                print(f"Gen {gen+1:3d} | Best Fit: {best_fit:8.0f} | Avg Fit: {avg_fit:8.0f} | Best Score: {best_score}")

        results[topo_key] = {
            "topology": topo,
            "params": _count_params(topo),
            "best_fitness_history": pop.best_fitness_history,
            "avg_fitness_history": pop.avg_fitness_history,
            "best_score_history": pop.best_score_history,
            "final_best_fitness": pop.best_fitness_history[-1],
            "final_avg_fitness": pop.avg_fitness_history[-1],
            "peak_score": max(pop.best_score_history) if pop.best_score_history else 0,
        }

    print(f"\n{'Topology':<20} {'Params':>8} {'Peak Score':>12} {'Final Fit':>12} {'Final Avg':>12}")
    print("-" * 68)
    for key, data in results.items():
        topo_str = " -> ".join(str(t) for t in data["topology"])
        print(f"{topo_str:<20} {data['params']:>8} {data['peak_score']:>12} {data['final_best_fitness']:>12.0f} {data['final_avg_fitness']:>12.0f}")

    return results


def _count_params(topology):
    total = 0
    for i in range(len(topology) - 1):
        total += topology[i] * topology[i + 1] + topology[i + 1]
    return total
