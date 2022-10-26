import numpy as np

from graph_coloring_algorithm import GraphColoringAlgorithm, OnePlaceCrossover, OnePlaceMutation, RouletteSelection


def main():
    graph = np.matrix(
        [[0, 1, 0, 0, 1],
         [1, 0, 1, 1, 1],
         [0, 1, 0, 1, 0],
         [0, 1, 1, 0, 1],
         [1, 1, 0, 1, 0]]
    )
    gca = GraphColoringAlgorithm(
        population_count=10,
        adjacency_matrix=graph,
        crossover=OnePlaceCrossover(probability=0.8),
        mutation=OnePlaceMutation(probability=0.05),
        selection=RouletteSelection()
    )
    gca.get_fitness([1, 1, 1, 1, 1])


if __name__ == '__main__':
    main()
