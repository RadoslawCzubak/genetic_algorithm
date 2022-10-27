import numpy as np

from base_genetic_algorithm import BaseGeneticAlgorithm
from graph_coloring_algorithm import OnePlaceCrossover, OnePlaceMutation, RouletteSelection, \
    GraphVertexColoringPopulation


def to_adj_matrix(string: str):
    lines = string.split('\n')
    adj_mat = []
    for line in lines:
        line = line.strip()
        line = [val.strip() for val in line.split(",")]
        clear_line = []
        for v in line:
            if len(v) > 0:
                clear_line.append(int(v))
        adj_mat.append(clear_line)
    return np.matrix(adj_mat)


def main():
    graph = """0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 
0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 
1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 
1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 
0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 
0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 
1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 
0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 
0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 
0, 0, 0, 1, 0, 1, 0, 0, 1, 0,  """
    graph = to_adj_matrix(graph)
    # graph = np.matrix(
    #     [[0, 1, 0, 0, 1, 0, 0, 0],
    #      [1, 0, 1, 1, 1, 0, 1, 0],
    #      [0, 1, 0, 1, 0, 1, 1, 0],
    #      [0, 1, 1, 0, 1, 0, 0, 1],
    #      [1, 1, 0, 1, 0, 0, 0, 0],
    #      [0, 0, 1, 0, 0, 0, 0, 0],
    #      [0, 1, 1, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 1, 0, 0, 0, 0]]
    # )

    gen_alg = BaseGeneticAlgorithm(
        population=GraphVertexColoringPopulation(10, graph),
        crossover=OnePlaceCrossover(probability=0.8),
        mutation=OnePlaceMutation(probability=0.05),
        selection=RouletteSelection()
    )
    gen_alg.run(10000)
    print(gen_alg.population.get_best_individual())
    print(len(set(gen_alg.population.get_best_individual())))


if __name__ == '__main__':
    main()
