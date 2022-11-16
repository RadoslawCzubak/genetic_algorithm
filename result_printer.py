import copy
import json
import time

import jsonpickle
import matplotlib.pyplot as plt
from tqdm import tqdm

from graph_coloring_algorithm import GraphColoringAlorithm


def open_results_file():
    with open("results.txt", "r") as f:
        result_json = f.read()
        results = jsonpickle.decode(result_json)
    return results


def open_results_saved_file():
    with open("plot_results.txt", "r") as f:
        result_json = f.read()
        results = json.loads(result_json)
    return results


def get_best_fitness(history, samples):
    step = int(len(history) / samples)

    return [abs(history[idx].get_best_individual_fitness_score()) for idx in
            tqdm(range(0, len(history), step))]


def save_computed_results(data):
    data = copy.deepcopy(data)
    for idx, d in enumerate(data):
        data[idx]["instance"] = get_best_fitness(d["instance"].history, 500)
    with open("plot_results.txt", "w") as f:
        f.write(json.dumps(data))
        f.close()


def show_from_results(fig, ax, results):
    for idx, result in enumerate(results):
        alg: GraphColoringAlorithm = result["instance"]
        ax_idx = int(idx / 5)
        best_fitness = get_best_fitness(alg.history, 500)
        if idx % 5 == 0:
            ax[ax_idx].set_title(f'cross_prob: {result["cross_prob"]}')
            plt.plot(best_fitness, 'r', label=result["mut_prob"])
        elif idx % 5 == 1:
            ax[ax_idx].set_title(f'cross_prob: {result["cross_prob"]}')
            plt.plot(best_fitness, 'g', label=result["mut_prob"])
        elif idx % 5 == 2:
            ax[ax_idx].set_title(f'cross_prob: {result["cross_prob"]}')
            plt.plot(best_fitness, 'b', label=result["mut_prob"])
        elif idx % 5 == 3:
            ax[ax_idx].set_title(f'cross_prob: {result["cross_prob"]}')
            plt.plot(best_fitness, 'y', label=result["mut_prob"])
        else:
            ax[ax_idx].set_title(f'cross_prob: {result["cross_prob"]}')
            plt.plot(best_fitness, 'm', label=result["mut_prob"])

    legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')
    legend.set_label('mut_prob')
    plt.show()


def show_from_save(fig, ax):
    with open("plot_results.txt", "r") as f:
        data = json.loads(f.read())

    for idx, result in enumerate(data):
        best_fitness = result["instance"]
        ax_idx = int(idx / 5)
        if idx % 5 == 0:
            ax[ax_idx].set_title(f'cross_prob: {result["cross_prob"]}')
            ax[ax_idx].plot(best_fitness, 'r', label=result["mut_prob"])
        elif idx % 5 == 1:
            ax[ax_idx].set_title(f'cross_prob: {result["cross_prob"]}')
            ax[ax_idx].plot(best_fitness, 'g', label=result["mut_prob"])
        elif idx % 5 == 2:
            ax[ax_idx].set_title(f'cross_prob: {result["cross_prob"]}')
            ax[ax_idx].plot(best_fitness, 'b', label=result["mut_prob"])
        elif idx % 5 == 3:
            ax[ax_idx].set_title(f'cross_prob: {result["cross_prob"]}')
            ax[ax_idx].plot(best_fitness, 'y', label=result["mut_prob"])
        else:
            ax[ax_idx].set_title(f'cross_prob: {result["cross_prob"]}')
            ax[ax_idx].plot(best_fitness, 'm', label=result["mut_prob"])

    legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')
    legend.set_label('mut_prob')
    plt.show()


def main():
    saved = True

    start = time.time()
    if saved:
        results = open_results_saved_file()
    else:
        results = open_results_file()
    end = time.time()
    print(f"file opened: {end - start}")
    fig, ax = plt.subplots(1, 4)
    if not saved:
        save_computed_results(results)
        show_from_results(fig, ax,results)
    else:
        show_from_save(fig, ax)


if __name__ == '__main__':
    main()
