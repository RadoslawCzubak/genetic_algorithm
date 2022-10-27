from abc import ABC, abstractmethod


class MutationMethod(ABC):

    @abstractmethod
    def mutate(self, population, valid_gene_values):
        pass


class SelectionMethod(ABC):

    @abstractmethod
    def select(self, population_with_fitness, k):
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
            selected = self.selection.select([(i, self.get_fitness(i)) for i in self.population], len(self.population))
            children = self.cross_over.crossover(selected)
            children = self.mutation.mutate(children, self.gene_values)
            self.population = children
            print(f"Generation {i + 1}")
            for child in children:
                print(f"{child} - {self.get_fitness(child)}")
            print("\n\n")


