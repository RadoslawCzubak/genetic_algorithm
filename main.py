from copy import deepcopy

import jsonpickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

from crossovers import OnePlaceCrossover
from graph_coloring_algorithm import GraphVertexColoringPopulation, GraphColoringAlorithm
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
    save_results = False

    graph = graph3

    G = nx.Graph(graph)

    test_population = GraphVertexColoringPopulation(10, graph, same_color_penalty=1,
                                                    color_diversity_penalty_multiplier=0.33)
    cross_probabilities = [0.7]  # [0.2, 0.5, 0.7, 0.99]
    mutation_probabilities = [0.05]  # [0.001, 0.01, 0.05, 0.1, 0.5]

    results = []
    for idx_cross, cross_probability in enumerate(cross_probabilities):
        for idx_mut, mutation_probability in enumerate(mutation_probabilities):
            gen_alg = GraphColoringAlorithm(
                population=deepcopy(test_population),
                pipeline=[
                    RouletteSelection(),
                    OnePlaceCrossover(probability=cross_probability),
                    OnePlaceMutation(probability=mutation_probability)
                ],
                no_update_iters=None
            )

            gen_alg.run(20000)
            color_solution = gen_alg.population.get_best_individual()
            best_color_solution_found = gen_alg.get_best_individual_with_iter()
            print(color_solution)
            print(f"best: {best_color_solution_found}")
            print(gen_alg.population.get_fitness_for_individual(color_solution))
            print(len(set(color_solution)))

            results.append({"cross_prob": cross_probability, "mut_prob": mutation_probability,
                            "instance": gen_alg})
            show_graph_with_colors(G, color_solution)
            plot_fitness_in_time(gen_alg.history, samples=500, mean=False)

    if save_results:
        with open("results.txt", "w") as f:
            f.write(jsonpickle.encode(results))
            f.close()


if __name__ == '__main__':
    main()
