import copy
from abc import ABC, abstractmethod
from typing import List, Set

from utils import is_list_of_unique_objects


class Population(ABC):

    def __init__(self, population_count: int, gene_values_sets: List[Set]):
        self.gene_values_sets = gene_values_sets
        self.population = None
        self.population_count = population_count

    def get_fitness_for_all(self):
        fitness_values = []
        for individual in self.population:
            fitness_values.append(self.get_fitness_for_individual(individual))
        return fitness_values

    @abstractmethod
    def generate_initial_population(self):
        pass

    @abstractmethod
    def get_fitness_for_individual(self, individual):
        pass

    def get_best_individual(self):
        def sort_func(x):
            return self.get_fitness_for_individual(x)
        pop_sorted = copy.deepcopy(self.population)
        pop_sorted.sort(key=sort_func)
        return pop_sorted[-1]


class MutationMethod(ABC):

    @abstractmethod
    def mutate(self, population: Population) -> Population:
        pass


class SelectionMethod(ABC):

    @abstractmethod
    def select(self, population: Population, k):
        pass


class CrossOverMethod(ABC):

    @abstractmethod
    def crossover(self, population: Population):
        pass


class BaseGeneticAlgorithm(ABC):

    def __init__(self, population: Population, crossover: CrossOverMethod, mutation: MutationMethod,
                 selection: SelectionMethod, verbose: bool = False):
        self.population = population
        self.mutation = mutation
        self.cross_over = crossover
        self.selection = selection
        self.gene_values = []
        self.verbose = verbose

    def run(self, generations):
        for i in range(0, generations):
            selected = self.selection.select(self.population, len(self.population.population))
            self.population.population = selected
            if self.verbose:
                print(is_list_of_unique_objects(self.population.population))
            children = self.cross_over.crossover(self.population)
            self.population.population = children
            if self.verbose:
                print(is_list_of_unique_objects(self.population.population))
            children = self.mutation.mutate(self.population)
            self.population = children
            if self.verbose:
                print(is_list_of_unique_objects(self.population.population))
                print(f"Generation {i + 1}")
                for child in self.population.population:
                    print(f"{child} - {self.population.get_fitness_for_individual(child)}")

                print(is_list_of_unique_objects(self.population.population))
                print("\n\n")
