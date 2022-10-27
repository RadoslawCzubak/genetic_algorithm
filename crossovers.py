import copy
import random

from base_genetic_algorithm import CrossOverMethod, Population


class OnePlaceCrossover(CrossOverMethod):

    def __init__(self, probability):
        self.probability = probability

    def crossover(self, population: Population) -> Population:
        new_pop = copy.deepcopy(population)
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
            new_pop.population = children
        return new_pop
