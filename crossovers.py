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
        parents_pairs = [(p1, p2) for p1, p2 in zip(population.individuals[::2], population.individuals[1::2])]
        crossover_probability = [random.random() for i in parents_pairs]

        for idx, (parent1, parent2) in enumerate(parents_pairs):
            if crossover_probability[idx] < self.probability:
                cross_field = random.randint(1, len(population.gene_values_sets) - 1)
                child1 = parent1[:cross_field] + parent2[cross_field:]
                child2 = parent2[:cross_field] + parent1[cross_field:]
                children.extend([child2, child1])
            else:
                children.extend([copy.deepcopy(parent1), copy.deepcopy(parent2)])
        new_pop.individuals = children
        return new_pop


class MultiplePlaceCrossover(CrossOverMethod):

    def __init__(self, n_crossover_places: int, probability):
        self.probability: float = probability
        self.n_crossover_places: int = n_crossover_places

    def crossover(self, population: Population) -> Population:
        new_pop = copy.deepcopy(population)
        n_genes = len(new_pop.gene_values_sets)

        if (n_genes - 1) < self.n_crossover_places:
            raise ValueError("N crossover places number has to be lower than genes number")

        children = []
        shuffle(population.individuals)
        if len(population.individuals) % 2 == 1:
            children.append(population.individuals[-1])
        parents_pairs = [(p1, p2) for p1, p2 in zip(population.individuals[::2], population.individuals[1::2])]
        crossover_probability = [random.random() for i in parents_pairs]

        for idx, (parent1, parent2) in enumerate(parents_pairs):
            if crossover_probability[idx] < self.probability:
                cross_fields = random.sample(range(1, self.n_crossover_places), self.n_crossover_places)
                cross_fields.sort()
                child1, child2 = self._crossover_multiple_places()
                children.extend([child2, child1])
            else:
                children.extend([copy.deepcopy(parent1), copy.deepcopy(parent2)])
        new_pop.individuals = children
        return new_pop

    def _crossover_multiple_places(self, cross_fields, parent1, parent2):
        child1 = []
        child2 = []

        for idx, cross_field in enumerate(cross_fields):
            if idx == 0:  # First cross place
                start_pos = 0
                end_pos = cross_field
            else:
                start_pos = cross_fields[idx - 1]
                end_pos = cross_field

            if idx % 2 == 0:
                child1.extend(parent1[start_pos:end_pos])
                child2.extend(parent2[start_pos:end_pos])
            else:
                child1.extend(parent2[start_pos:end_pos])
                child2.extend(parent1[start_pos:end_pos])

            # is last cross place?
            if idx == len(cross_fields) - 1:
                if (idx + 1) % 2 == 0:
                    child1.extend(parent1[end_pos:])
                    child2.extend(parent2[end_pos:])
                else:
                    child1.extend(parent2[end_pos:])
                    child2.extend(parent1[end_pos:])
        return child1, child2


class UniformCrossover(MultiplePlaceCrossover):

    def __init__(self, probability: float):
        super().__init__(n_crossover_places=1, probability=probability)

    def crossover(self, population: Population) -> Population:
        self.n_crossover_places = len(population.gene_values_sets) - 1
        return super().crossover(population)
