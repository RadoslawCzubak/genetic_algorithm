import copy
import random
from typing import List

from base_genetic_algorithm import SelectionMethod, Population


class RouletteSelection(SelectionMethod):

    def select(self, population: Population, k) -> Population:
        new_pop = copy.deepcopy(population)
        choices = random.choices(population.individuals,
                                 self.get_distribution_function(population), k=k)
        new_pop.individuals = [copy.deepcopy(item) for item in choices]
        return new_pop

    @staticmethod
    def get_distribution_function(population: Population):
        fitness_scores = population.get_fitness_for_all()
        try:
            standardized = [(i - min(fitness_scores)) / (max(fitness_scores) - min(fitness_scores)) for i in
                            fitness_scores]
        except ZeroDivisionError:
            standardized = [1 for i in fitness_scores]

        min_non_zero_value = min([i for i in standardized if i > 0])
        for idx, value in enumerate(standardized):
            if value > 0:
                pass
            else:
                standardized[idx] = min_non_zero_value / 2
        return standardized


class TournamentSelection(SelectionMethod):

    def __init__(self, n_elements_in_group):
        self.n_elements_in_group = n_elements_in_group

    def select(self, population: Population, k) -> Population:
        new_population = copy.deepcopy(population)

        groups = self._group_individuals(population.individuals, k)

        def key_fun(individual: List):
            return population.get_fitness_for_individual(individual)

        selected = []
        for group in groups:
            group.sort(key=key_fun)
            best = group[-1]
            selected.append(copy.deepcopy(best))
        new_population.individuals = selected
        return new_population

    def _group_individuals(self, individuals, k):
        groups = [random.sample(individuals, self.n_elements_in_group) for i in range(k)]
        return groups


class RankingSelection(SelectionMethod):
    def select(self, population: Population, k) -> Population:
        def key_fun(individual):
            return population.get_fitness_for_individual(individual)

        new_pop = copy.deepcopy(population)
        sorted_individuals = sorted(population.individuals, key=key_fun)
        selected = random.sample(sorted_individuals, k, counts=range(1, len(sorted_individuals) + 1))
        new_pop.individuals = selected
        return new_pop
