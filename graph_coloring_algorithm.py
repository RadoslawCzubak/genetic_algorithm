import random

import numpy as np

from base_genetic_algorithm import BaseGeneticAlgorithm, MutationMethod, SelectionMethod, CrossOverMethod


class OnePlaceMutation(MutationMethod):

    def __init__(self, probability):
        self.probability = probability

    def mutate(self, population):
        print("mutate")


class OnePlaceCrossover(CrossOverMethod):

    def __init__(self, probability):
        self.probability = probability

    def crossover(self, population):
        print("crossover")


class RouletteSelection(SelectionMethod):

    def select(self, population):
        print("select")


class GraphColoringAlgorithm(BaseGeneticAlgorithm):

    def __init__(self, population_count: int, adjacency_matrix: np.matrix, crossover, mutation, selection):
        super().__init__(crossover, mutation, selection)
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError

        if population_count % 2 == 1:
            raise ValueError("Population should be even")
        self.adjacency_matrix = adjacency_matrix
        self.generate_initial_population(population_count)

    def generate_initial_population(self, pop_count):
        am = np.matrix(self.adjacency_matrix)
        vertex_count = am.shape[0]
        max_vertex_degree = am.sum(0).max()
        max_color_count = max_vertex_degree + 1
        population = []
        for i in range(pop_count):
            population.append([random.randrange(max_color_count) for i in range(vertex_count)])
        self.population = population

    def get_fitness(self, chromosome) -> float:
        # k = len(list(set(chromosome)))
        penalty = 0
        for i in range(self.get_vertex_count()):
            for j in range(i, self.get_vertex_count()):
                if self.adjacency_matrix[i, j] == 1 and chromosome[i] == chromosome[j] and i != j:
                    penalty += 1
        return -float(penalty)

    def get_vertex_count(self):
        return self.adjacency_matrix.shape[0]
