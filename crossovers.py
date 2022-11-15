import copy
import random
from random import shuffle

from base_genetic_algorithm import CrossOverMethod, Population


class OnePlaceCrossover(CrossOverMethod):

    def __init__(self, probability):
        self.probability = probability

    def crossover(self, population: Population) -> Population:
        new_pop = copy.deepcopy(population)
        children = []
        shuffle(population.individuals)
        if len(population.individuals) % 2 == 1:
            children.append(population.individuals[-1])
        parents_pairs = zip(population.individuals[::2], population.individuals[1::2])
        crossover_probability = [random.random() for i in parents_pairs]

        for idx, (parent1, parent2) in enumerate(parents_pairs):
            if crossover_probability[idx] < self.probability:
                cross_field = random.randint(1, len(population.gene_values_sets) - 1)
                child1 = parent1[:cross_field] + parent2[cross_field:]
                child2 = parent2[:cross_field] + parent1[cross_field:]
                children = children + [child1, child2]
            else:
                children = children + [parent1, parent2]
        new_pop.individuals = children
        return new_pop
