import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from base_genetic_algorithm import BaseGeneticAlgorithm
from crossovers import OnePlaceCrossover
from graph_coloring_algorithm import GraphVertexColoringPopulation
from mutations import OnePlaceMutation
from selections import RouletteSelection


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
    gen_alg.run(100)
    color_solution = gen_alg.population.get_best_individual()
    print(color_solution)
    print(gen_alg.population.get_fitness_for_individual(color_solution))
    print(len(set(color_solution)))

    gr = nx.Graph(graph)
    pos = nx.shell_layout(gr)
    color_map = plt.cm.get_cmap('Spectral')

    nx.draw(gr,
            pos,
            node_color=[color_map(color_solution[idx] / max(color_solution)) for idx, node in enumerate(gr)],
            with_labels=True)
    plt.show()


if __name__ == '__main__':
    main()
