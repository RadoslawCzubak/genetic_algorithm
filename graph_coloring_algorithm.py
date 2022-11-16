import random
from typing import Optional, List

import numpy as np

from base_genetic_algorithm import Population, PipelineGeneticAlgorithm


class GraphVertexColoringPopulation(Population, object):

    def __init__(self, population_count: int, adjacency_matrix: np.matrix,
                 color_diversity_penalty_multiplier: float = 1.0,
                 same_color_penalty: Optional[int] = None):
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("adjacency_matrix must be square!")

        self.adjacency_matrix = adjacency_matrix
        self.population_count = population_count
        self.color_weight = self.get_max_colors()
        self.color_diversity_penalty_multiplier = color_diversity_penalty_multiplier
        if same_color_penalty is None:
            self.same_color_penalty = self.color_weight
        else:
            self.same_color_penalty = same_color_penalty
        gene_values = set(range(self.get_max_colors()))
        super().__init__(population_count, [gene_values for i in range(adjacency_matrix.shape[0])])

        self.generate_initial_population()

    def get_fitness_for_individual(self, individual):
        k = len(list(set(individual)))
        penalty = 0 + k * self.color_diversity_penalty_multiplier
        connected_vertex = np.where(self.adjacency_matrix == 1)
        for idx in range(len(connected_vertex[0])):
            i, j = connected_vertex[0][idx], connected_vertex[1][idx]
            if i != j and individual[i] == individual[j]:
                penalty += self.same_color_penalty
        return -float(penalty)

    def generate_initial_population(self):
        am = np.matrix(self.adjacency_matrix)
        vertex_count = am.shape[0]
        individuals = []
        for i in range(self.population_count):
            individuals.append([random.choice(list(self.gene_values_sets[i])) for i in range(vertex_count)])
        self.individuals = individuals

    def get_max_colors(self):
        return self.adjacency_matrix.sum(0).max() + 1


class GraphColoringAlorithm(PipelineGeneticAlgorithm, object):

    def __init__(self, population: Population, pipeline: List, no_update_iters: Optional[int] = None,
                 verbose: bool = False):
        super().__init__(population, pipeline, verbose)
        self.best = None
        self.last_best_update = 0
        self.no_update_iters = no_update_iters

    def run_iteration(self):
        super(GraphColoringAlorithm, self).run_iteration()
        best_in_iter = self.population.get_fitness_for_individual(self.population.get_best_individual())
        if self.best is None:
            self.best = best_in_iter
        if best_in_iter > self.best:
            self.best = best_in_iter
            self.last_best_update = 0
        else:
            self.last_best_update += 1

    def stop_condition(self):
        if self.no_update_iters is None:
            return False
        else:
            return self.last_best_update > self.no_update_iters
