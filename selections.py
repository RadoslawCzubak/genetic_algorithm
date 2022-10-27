import copy
import random

from base_genetic_algorithm import SelectionMethod, Population


class RouletteSelection(SelectionMethod):

    def select(self, population: Population, k):
        new_pop = copy.deepcopy(population)
        choices = random.choices(population.population,
                                 [1 / abs(fitness) for fitness in population.get_fitness_for_all()], k=k)
        new_pop.population = [copy.deepcopy(item) for item in choices]
        return new_pop
