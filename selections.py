import copy
import random

from base_genetic_algorithm import SelectionMethod, Population


class RouletteSelection(SelectionMethod):

    def select(self, population: Population, k):
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
