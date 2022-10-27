import random

import numpy as np

from base_genetic_algorithm import Population


class GraphVertexColoringPopulation(Population):

    def __init__(self, population_count: int, adjacency_matrix: np.matrix):
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("adjacency_matrix must be square!")

        self.adjacency_matrix = adjacency_matrix
        self.population_count = population_count
        gene_values = set(range(self.get_max_colors()))
        super().__init__(population_count, [gene_values for i in range(adjacency_matrix.shape[0])])

        self.generate_initial_population()

    def get_fitness_for_individual(self, individual):
        k = len(list(set(individual)))
        penalty = 0 + k
        vertex_count = len(individual)
        for i in range(vertex_count):
            for j in range(i, vertex_count):
                if self.adjacency_matrix[i, j] == 1 and individual[i] == individual[j] and i != j:
                    penalty += 1
        return -float(penalty)

    def generate_initial_population(self):
        am = np.matrix(self.adjacency_matrix)
        vertex_count = am.shape[0]
        population = []
        for i in range(self.population_count):
            population.append([random.choice(list(self.gene_values_sets[i])) for i in range(vertex_count)])
        self.population = population

    def get_max_colors(self):
        return self.adjacency_matrix.sum(0).max() + 1
