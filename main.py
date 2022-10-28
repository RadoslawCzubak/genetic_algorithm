import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from test_graphs import graph1, graph2, graph3
from base_genetic_algorithm import BaseGeneticAlgorithm
from crossovers import OnePlaceCrossover
from graph_coloring_algorithm import GraphVertexColoringPopulation
from mutations import OnePlaceMutation
from selections import RouletteSelection
from tqdm import tqdm


def show_graph_with_colors(graph, vertex_colors):
    pos = nx.shell_layout(graph)
    color_map = plt.cm.get_cmap('Spectral')

    nx.draw(
        graph,
        pos,
        node_color=[color_map(vertex_colors[idx] / max(vertex_colors)) for idx, node in enumerate(graph)],
        with_labels=True
    )
    plt.show()


def plot_fitness_in_time(algorithm_history, mean=False):
    if not mean:
        plt.plot([abs(pop.get_best_individual_fitness_score()) for pop in tqdm(algorithm_history)])
    else:
        plt.plot([abs(np.array(pop.get_fitness_for_all()).mean()) for pop in tqdm(algorithm_history)])
    plt.show()


def main():
    graph = graph3

    G = nx.Graph(graph)

    gen_alg = BaseGeneticAlgorithm(
        population=GraphVertexColoringPopulation(10, graph),
        crossover=OnePlaceCrossover(probability=0.8),
        mutation=OnePlaceMutation(probability=0.1),
        selection=RouletteSelection()
    )
    gen_alg.run(10000)
    color_solution = gen_alg.population.get_best_individual()
    print(color_solution)
    print(gen_alg.population.get_fitness_for_individual(color_solution))
    print(len(set(color_solution)))

    show_graph_with_colors(G, color_solution)
    plot_fitness_in_time(gen_alg.history, True)


if __name__ == '__main__':
    main()
