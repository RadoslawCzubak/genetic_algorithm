import copy
import random
from typing import List

import numpy as np

from base_genetic_algorithm import BaseGeneticAlgorithm, MutationMethod, SelectionMethod, CrossOverMethod


class OnePlaceMutation(MutationMethod):

    def __init__(self, probability):
        self.probability = probability

    def mutate(self, population, possible_values: list):
        mutation_prob = [random.random() for i in population]
        for index, individual in enumerate(population):
            if mutation_prob[index] < self.probability:
                individual[random.randint(0, len(individual))] = copy.deepcopy(random.choice(possible_values))
        return population


class OnePlaceCrossover(CrossOverMethod):

    def __init__(self, probability):
        self.probability = probability

    def crossover(self, population: List[List[int]]):
        cross_prob = [random.random() for _ in range(len(population))]
        to_cross = []
        children = []
        for index, individual in enumerate(population):
            if cross_prob[index] < self.probability:
                to_cross.append(individual)
            else:
                children.append(individual)

        if len(to_cross) % 2 == 1:
            children.append(to_cross.pop(random.randint(0, len(to_cross) - 1)))

        for i in range(0, len(to_cross), 2):
            cross_field = random.randint(1, len(to_cross[i]) - 1)
            parent1 = to_cross[i]
            parent2 = to_cross[i + 1]
            child1 = parent1[:cross_field] + parent2[cross_field:]
            child2 = parent2[:cross_field] + parent1[cross_field:]
            children.extend([child2, child1])
        return children


class RouletteSelection(SelectionMethod):

    def select(self, population_with_fitness, k):
        choices = random.choices([individual for individual, fitness in population_with_fitness],
                                 [1 / abs(fitness) for individual, fitness in population_with_fitness], k=k)
        return [copy.deepcopy(item) for item in choices]


class GraphColoringAlgorithm(BaseGeneticAlgorithm):

    def __init__(self, population_count: int, adjacency_matrix: np.matrix, crossover, mutation, selection):
        super().__init__(crossover, mutation, selection)
        self.gene_values = []
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
        self.gene_values = max_color_count

    def get_fitness(self, chromosome) -> float:
        k = len(list(set(chromosome))) / 10
        penalty = 0 + k
        for i in range(self.get_vertex_count()):
            for j in range(i, self.get_vertex_count()):
                if self.adjacency_matrix[i, j] == 1 and chromosome[i] == chromosome[j] and i != j:
                    penalty += 1
        return -float(penalty)

    def get_vertex_count(self):
        return self.adjacency_matrix.shape[0]
