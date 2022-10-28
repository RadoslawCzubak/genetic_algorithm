import copy
import random

from base_genetic_algorithm import MutationMethod, Population


class OnePlaceMutation(MutationMethod):

    def __init__(self, probability):
        self.probability = probability

    def mutate(self, population: Population):
        mutation_prob = [random.random() for i in population.population]
        for index, individual in enumerate(population.population):
            if mutation_prob[index] < self.probability:
                mutation_index = random.randint(0, len(individual) - 1)
                individual[mutation_index] = copy.deepcopy(
                    random.choice(list(population.gene_values_sets[mutation_index]))
                )
        return population


class Inversion(MutationMethod):

    def __init__(self, probability):
        self.probability = probability

    def mutate(self, population: Population) -> Population:
        mutation_prob = [random.random() for i in population.population]
        for index, individual in enumerate(population.population):
            if mutation_prob[index] < self.probability:
                start = random.randint(1, len(population.gene_values_sets) - 1)
                end = random.randint(1, len(population.gene_values_sets) - 1)
                indices = [start, end]
                indices.sort()
                inverted = individual[indices[0]: indices[1]]
                inverted.reverse()
                individual[indices[0]: indices[1]] = inverted
        return population
