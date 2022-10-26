from abc import ABC, abstractmethod


class MutationMethod(ABC):

    @abstractmethod
    def mutate(self, population):
        pass


class SelectionMethod(ABC):

    @abstractmethod
    def select(self, population):
        pass


class CrossOverMethod(ABC):

    @abstractmethod
    def crossover(self, population):
        pass


class BaseGeneticAlgorithm(ABC):

    def __init__(self, crossover: CrossOverMethod, mutation: MutationMethod, selection: SelectionMethod):
        self.population = None
        self.mutation = mutation
        self.cross_over = crossover
        self.selection = selection

    @abstractmethod
    def get_fitness(self, chromosome) -> float:
        pass

    def run(self, generations):
        for i in range(0, generations):
            children = self.cross_over.crossover(self.population)
            children = self.mutation.mutate(children)
            self.population = self.selection.select(children)
