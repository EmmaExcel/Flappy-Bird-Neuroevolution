import random

import numpy as np

from config import (
    ELITISM_COUNT,
    MUTATION_RATE,
    MUTATION_STRENGTH,
    TOURNAMENT_K,
)
from network import Network


class Population:
    def __init__(self, size, topology):
        self.size = size
        self.topology = list(topology)
        self.networks = [Network(topology) for _ in range(size)]
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_score_history = []

    def evaluate_fitness(self, birds):
        fitnesses = []
        for b in birds:
            pipe_reward = b.max_pipe_progress * 5000
            survival_reward = b.distance
            fitnesses.append(pipe_reward + survival_reward)
        return fitnesses

    def evolve(self, fitnesses):
        self.generation += 1
        sorted_indices = np.argsort(fitnesses)[::-1]

        new_networks = []

        for i in range(min(ELITISM_COUNT, len(sorted_indices))):
            new_networks.append(self.networks[sorted_indices[i]].clone())

        while len(new_networks) < self.size:
            p1 = self._tournament_select(fitnesses)
            p2 = self._tournament_select(fitnesses)
            child = self._crossover(p1, p2)
            self._mutate(child)
            new_networks.append(child)

        best = float(max(fitnesses))
        avg = float(sum(fitnesses) / len(fitnesses))
        self.best_fitness_history.append(best)
        self.avg_fitness_history.append(avg)

        self.networks = new_networks

    def _tournament_select(self, fitnesses):
        k = min(TOURNAMENT_K, len(fitnesses))
        indices = random.sample(range(len(fitnesses)), k)
        winner = max(indices, key=lambda i: fitnesses[i])
        return self.networks[winner]

    def _crossover(self, parent1, parent2):
        child = Network(self.topology)
        p1 = parent1.get_params()
        p2 = parent2.get_params()
        mask = np.random.random(len(p1)) < 0.5
        child.set_params(np.where(mask, p1, p2))
        return child

    def _mutate(self, network):
        params = network.get_params()
        mask = np.random.random(len(params)) < MUTATION_RATE
        num_mutations = int(mask.sum())
        if num_mutations > 0:
            params[mask] += np.random.randn(num_mutations) * MUTATION_STRENGTH
        network.set_params(params)
