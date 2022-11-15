import copy
from abc import ABC, abstractmethod
from typing import List, Set, Optional

from tqdm import tqdm

from utils import is_list_of_unique_objects


class Population(ABC):

    def __init__(self, population_count: int, gene_values_sets: List[Set]):
        self.gene_values_sets: List[Set] = gene_values_sets
        self.individuals: List[List] = []
        self.population_count: int = population_count

    def get_fitness_for_all(self):
        fitness_values = []
        for individual in self.individuals:
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

        pop_sorted = copy.deepcopy(self.individuals)
        pop_sorted.sort(key=sort_func)
        return pop_sorted[-1]

    def get_best_individual_fitness_score(self):
        return max(self.get_fitness_for_all())


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

    def __init__(self, population: Population, crossover: Optional[CrossOverMethod], mutation: Optional[MutationMethod],
                 selection: Optional[SelectionMethod], verbose: bool = False):
        self.population = population
        self.mutation = mutation
        self.cross_over = crossover
        self.selection = selection
        self.gene_values = []
        self.history = []
        self.verbose = verbose

    def run(self, generations):
        for i in tqdm(range(0, generations)):
            self.run_iteration()
            self.history.append(copy.deepcopy(self.population))
            if self.verbose:
                print(f"Generation {i + 1}")
                for child in self.population.individuals:
                    print(f"{child} - {self.population.get_fitness_for_individual(child)}")

                print(is_list_of_unique_objects(self.population.individuals))
                print("\n\n")
            if self.stop_condition():
                break

    @abstractmethod
    def run_iteration(self):
        pass

    def get_best_individual_with_iter(self):
        best = self.population.get_best_individual()
        idx_best = 0
        for idx, population in enumerate(self.history):
            best_in_population = population.get_best_individual()
            if self.population.get_fitness_for_individual(
                    best_in_population) > self.population.get_fitness_for_individual(best):
                best = best_in_population
                idx_best = idx
        return idx_best + 1, best, self.population.get_fitness_for_individual(best)

    @abstractmethod
    def stop_condition(self) -> bool:
        pass


class PipelineGeneticAlgorithm(BaseGeneticAlgorithm, ABC):

    def __init__(self, population: Population, pipeline: List, verbose: bool = False):
        super().__init__(population, None, None, None, verbose)
        self.pipeline = pipeline

    def run_iteration(self):
        for operator in self.pipeline:
            if isinstance(operator, MutationMethod):
                self.population = operator.mutate(self.population)
            elif isinstance(operator, SelectionMethod):
                self.population = operator.select(self.population, self.population.population_count)
            elif isinstance(operator, CrossOverMethod):
                self.population = operator.crossover(self.population)
