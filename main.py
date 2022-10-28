import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

from base_genetic_algorithm import PipelineGeneticAlgorithm
from crossovers import OnePlaceCrossover
from graph_coloring_algorithm import GraphVertexColoringPopulation
from mutations import OnePlaceMutation
from selections import RouletteSelection
from test_graphs import graph3


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


def plot_fitness_in_time(algorithm_history, mean=False, samples=None):
    if samples is None:
        step = 1
    else:
        step = int(len(algorithm_history) / samples)
    if not mean:
        plt.plot([abs(algorithm_history[idx].get_best_individual_fitness_score()) for idx in
                  tqdm(range(0, len(algorithm_history), step))])
    else:
        plt.plot(
            [abs(np.array(algorithm_history[idx].get_fitness_for_all()).mean()) for idx in
             tqdm(range(0, len(algorithm_history), step))])
    plt.show()


def main():
    graph = graph3

    G = nx.Graph(graph)

    # gen_alg = BaseGeneticAlgorithm(
    #     population=GraphVertexColoringPopulation(10, graph),
    #     crossover=OnePlaceCrossover(probability=0.8),
    #     mutation=OnePlaceMutation(probability=0.1),
    #     selection=RouletteSelection()
    # )

    gen_alg = PipelineGeneticAlgorithm(
        population=GraphVertexColoringPopulation(10, graph),
        pipeline=[
            RouletteSelection(),
            OnePlaceCrossover(probability=0.8),
            OnePlaceMutation(probability=0.8)
        ]
    )

    gen_alg.run(10000)
    color_solution = gen_alg.population.get_best_individual()
    print(color_solution)
    print(gen_alg.population.get_fitness_for_individual(color_solution))
    print(len(set(color_solution)))

    show_graph_with_colors(G, color_solution)
    plot_fitness_in_time(gen_alg.history, True, samples=500)


if __name__ == '__main__':
    main()
