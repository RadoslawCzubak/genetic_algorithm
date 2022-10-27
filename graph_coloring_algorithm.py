import copy
import random

import numpy as np

from base_genetic_algorithm import MutationMethod, SelectionMethod, CrossOverMethod, Population


class OnePlaceMutation(MutationMethod):

    def __init__(self, probability):
        self.probability = probability

    def mutate(self, population: Population):
        mutation_prob = [random.random() for i in population.population]
        for index, individual in enumerate(population.population):
            if mutation_prob[index] < self.probability:
                mutation_index = random.randint(0, len(individual)-1)
                individual[mutation_index] = copy.deepcopy(
                    random.choice(list(population.gene_values_sets[mutation_index]))
                )
        return population


class OnePlaceCrossover(CrossOverMethod):

    def __init__(self, probability):
        self.probability = probability

    def crossover(self, population: Population):
        cross_prob = [random.random() for _ in range(len(population.population))]
        to_cross = []
        children = []
        for index, individual in enumerate(population.population):
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

    def select(self, population: Population, k):
        choices = random.choices(population.population,
                                 [1 / abs(fitness) for fitness in population.get_fitness_for_all()], k=k)
        return [copy.deepcopy(item) for item in choices]


class GraphVertexColoringPopulation(Population):

    def __init__(self, population_count: int, adjacency_matrix: np.matrix):
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError

        if population_count % 2 == 1:
            raise ValueError("Population should be even")
        self.adjacency_matrix = adjacency_matrix
        self.population_count = population_count
        gene_values = set(range(self.get_max_colors()))
        super().__init__(population_count, [gene_values for i in range(adjacency_matrix.shape[0])])

        self.generate_initial_population()

    def get_fitness_for_individual(self, individual):
        k = len(list(set(individual))) / 5
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
