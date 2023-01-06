import copy
from copy import deepcopy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

import crossovers
import mutations
import selections
import utils
from graph_coloring_algorithm import GraphVertexColoringPopulation, GraphColoringAlorithm
from test_graphs import graph3


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
    cross_probabilities = [0.8]  # [0.2, 0.5, 0.7, 0.99]

    selection_mthds = [
        ("roulette", selections.RouletteSelection()),
        ("tournament2", selections.TournamentSelection(2)),
        ("tournament3", selections.TournamentSelection(3)),
        ("ranking", selections.RankingSelection()),
    ]

    crossover_mthds = [
        ("oneplace", crossovers.OnePlaceCrossover(probability=1)),
        ("multiplace2", crossovers.MultiplePlaceCrossover(probability=1, n_crossover_places=2)),
        ("multiplace4", crossovers.MultiplePlaceCrossover(probability=1, n_crossover_places=4)),
        ("uniform", crossovers.UniformCrossover(probability=1))
    ]

    mutation_probabilities = [0.0, 0.05, 0.1]  # [0.0, 0.01, 0.05, 0.1, 0.5]
    inversion_probabilities = [0.0]  # [0.0, 0.01, 0.05, 0.1, 0.5]

    results = []
    for idx_cross, cross_probability in tqdm(enumerate(cross_probabilities), position=0):
        for idx_mut, mutation_probability in tqdm(enumerate(mutation_probabilities), position=1, leave=False):
            for idx_inv, inversion_probability in tqdm(enumerate(inversion_probabilities), position=2, leave=False):
                for selection_name, selection_mthd in tqdm(selection_mthds, position=3, leave=False):
                    for cross_name, cross_method in tqdm(crossover_mthds, position=4, leave=False):
                        for evaluation_idx in tqdm(range(10), position=5, leave=False):
                            c = copy.deepcopy(cross_method)
                            c.probability = cross_probability

                            gen_alg = GraphColoringAlorithm(
                                population=deepcopy(test_population),
                                pipeline=[
                                    selection_mthd,
                                    c,
                                    mutations.SortMutation(probability=mutation_probability),
                                    mutations.Inversion(probability=inversion_probability)
                                ],
                                no_update_iters=None,
                                verbose=False
                            )

                            gen_alg.run(100)
                            # color_solution = gen_alg.population.get_best_individual()
                            # best_color_solution_found = gen_alg.get_best_individual_with_iter()
                            # print(color_solution)
                            # print(f"best: {best_color_solution_found}")
                            # print(gen_alg.population.get_fitness_for_individual(color_solution))
                            # print(len(set(color_solution)))

                            # results.append({"cross_prob": cross_probability, "mut_prob": mutation_probability,
                            #                 "instance": gen_alg})
                            # show_graph_with_colors(G, color_solution)
                            # plot_fitness_in_time(gen_alg.history, samples=500, mean=False)
                            utils.save_algorithm_history(gen_alg, "sort-mutation-res",
                                                         f"{cross_probability}_{mutation_probability}_{inversion_probability}_{selection_name}_{cross_name}_{evaluation_idx}")

    # if save_results:
    #     with open("results.txt", "w") as f:
    #         f.write(jsonpickle.encode(results))
    #         f.close()


if __name__ == '__main__':
    main()
