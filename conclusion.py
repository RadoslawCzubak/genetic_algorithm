import itertools
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from test_graphs import graph3

SELECTION_NAME = "sel_name"
CROSS_NAME = "cross_name"
CROSS_PROB = "cross_prob"
INV_PROB = "inv_prob"
MUT_PROB = "mut_prob"
ITER = "iter"

DIRECTORY = "results"


def load_data():
    # {cross_probability}_{mutation_probability}_{inversion_probability}_{selection_name}_{cross_name}_{evaluation_idx}n_iteration

    files = os.listdir(DIRECTORY)
    results = []
    for file in files:
        if not re.match(".*_.*_.*_.*_.*_.*\.npy", file):
            continue
        name = file.replace(".npy", '').split("_")
        name = {
            CROSS_PROB: name[0],
            MUT_PROB: name[1],
            INV_PROB: name[2],
            SELECTION_NAME: name[3],
            CROSS_NAME: name[4],
            ITER: name[5],
        }
        results.append((name, np.load(os.path.join(DIRECTORY, file))))

    return results


color_diversity_penalty_multiplier = 0.33
same_color_penalty = 1
adjacency_matrix = graph3


def get_fitness_for_individual(individual):
    k = len(list(set(individual)))
    penalty = 0 + k * color_diversity_penalty_multiplier
    connected_vertex = np.where(adjacency_matrix == 1)
    for idx in range(len(connected_vertex[0])):
        i, j = connected_vertex[0][idx], connected_vertex[1][idx]
        if i != j and individual[i] == individual[j]:
            penalty += same_color_penalty
    return -float(penalty)


def plot_selection_to_crossovers_no_inv_matrix():
    results = load_data()

    def check_info(info, inv):
        return info[INV_PROB] == inv #and info[MUT_PROB] == "0.0"

    results = [r for r in results if check_info(r[0], inv='0.0')]

    def key_fun(o):
        return o[0][SELECTION_NAME] + "_" + o[0][CROSS_NAME]

    def key_fun2(o):
        return o[0][CROSS_NAME] + "_" + o[0][SELECTION_NAME]

    def cross_key(o):
        return o[0][CROSS_NAME]

    def selection_key(o):
        return o[0][SELECTION_NAME]

    results.sort(key=key_fun)
    idx_s = 0
    idx_c = 0

    for selection, v_s in itertools.groupby(results, key=selection_key):
        v_ss = [i for i in v_s]
        for cross, v_c in itertools.groupby(v_ss, key=cross_key):
            values = [i[1] for i in v_c]
            fitness_scores = np.apply_along_axis(get_fitness_for_individual, 3, values)
            fitness_scores = fitness_scores.mean(axis=2)
            fitness_scores = fitness_scores.mean(axis=0)
            plt.ylim([-10, 0])
            plt.subplot(4, 4, (idx_s * 4) + (idx_c + 1))
            plt.title(f"{selection}_{cross}")
            plt.plot(fitness_scores.tolist())
            idx_c += 1
        idx_s += 1
        idx_c = 0
    plt.show()

    idx_s = 0
    idx_c = 0
    fig, ax = plt.subplots(1, 4)
    for selection, v_s in itertools.groupby(results, key=selection_key):
        v_ss = [i for i in v_s]
        for cross, v_c in itertools.groupby(v_ss, key=cross_key):
            values = [i[1] for i in v_c]
            fitness_scores = np.apply_along_axis(get_fitness_for_individual, 3, values)
            fitness_scores = fitness_scores.mean(axis=2)
            fitness_scores = fitness_scores.mean(axis=0)
            ax_idx = int(idx_c / 4)
            ax[ax_idx].set_ylim([-10, 0])
            if idx_c % 4 == 0:
                ax[ax_idx].set_title(f'{selection}')
                ax[ax_idx].plot(fitness_scores, 'r', label=f"{cross}")
            elif idx_c % 4 == 1:
                ax[ax_idx].set_title(f'{selection}')
                ax[ax_idx].plot(fitness_scores, 'g', label=f"{cross}")
            elif idx_c % 4 == 2:
                ax[ax_idx].set_title(f'{selection}')
                ax[ax_idx].plot(fitness_scores, 'b', label=f"{cross}")
            else:
                ax[ax_idx].set_title(f'{selection}')
                ax[ax_idx].plot(fitness_scores, 'y', label=f"{cross}")
            idx_c += 1
        legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')
        legend.set_label('cross')
    plt.show()

    results.sort(key=key_fun2)
    idx_s = 0
    idx_c = 0
    fig, ax = plt.subplots(1, 4)
    for cross, v_c in itertools.groupby(results, key=cross_key):
        v_cc = [i for i in v_c]
        for selection, v_s in itertools.groupby(v_cc, key=selection_key):
            values = [i[1] for i in v_s]
            fitness_scores = np.apply_along_axis(get_fitness_for_individual, 3, values)
            fitness_scores = fitness_scores.mean(axis=2)
            fitness_scores = fitness_scores.mean(axis=0)

            ax_idx = int(idx_s / 4)
            ax[ax_idx].set_ylim([-10, 0])

            if idx_s % 4 == 0:
                ax[ax_idx].set_title(f'{cross}')
                ax[ax_idx].plot(fitness_scores, 'r', label=f"{selection}")
            elif idx_s % 4 == 1:
                ax[ax_idx].set_title(f'{cross}')
                ax[ax_idx].plot(fitness_scores, 'g', label=f"{selection}")
            elif idx_s % 4 == 2:
                ax[ax_idx].set_title(f'{cross}')
                ax[ax_idx].plot(fitness_scores, 'b', label=f"{selection}")
            else:
                ax[ax_idx].set_title(f'{cross}')
                ax[ax_idx].plot(fitness_scores, 'y', label=f"{selection}")
            idx_s += 1
        legend = plt.legend(loc='lower right', shadow=True, fontsize='x-large')
        legend.set_label('selection')
    plt.show()


def show_inv_vs_non_inv():
    results = load_data()

    def check_info(info, inv):
        return info[INV_PROB] == inv #and info[MUT_PROB] == "0.1"

    results_non_inv = [r for r in results if check_info(r[0], inv='0.0')]
    results_inv1 = [r for r in results if check_info(r[0], inv='0.1')]
    results_inv05 = [r for r in results if check_info(r[0], inv='0.05')]

    def key_fun(o):
        return o[0][SELECTION_NAME] + "_" + o[0][CROSS_NAME]

    def key_fun2(o):
        return o[0][CROSS_NAME] + "_" + o[0][SELECTION_NAME]

    def cross_key(o):
        return o[0][CROSS_NAME]

    def selection_key(o):
        return o[0][SELECTION_NAME]

    fig, ax = plt.subplots(4, 4)
    for res_name, color, results in [("inv_prob=0.0", "r", results_non_inv), ("inv_prob=0.05", "g", results_inv05),
                                     ("inv_prob=0.1", "b", results_inv1)]:
        results.sort(key=key_fun)
        idx_s = 0
        idx_c = 0
        for selection, v_s in itertools.groupby(results, key=selection_key):
            v_ss = [i for i in v_s]
            for cross, v_c in itertools.groupby(v_ss, key=cross_key):
                values = [i[1] for i in v_c]
                fitness_scores = np.apply_along_axis(get_fitness_for_individual, 3, values)
                fitness_scores = fitness_scores.mean(axis=2)
                fitness_scores = fitness_scores.mean(axis=0)
                ax[idx_s, idx_c].set_ylim([-10, 0])
                ax[idx_s, idx_c].set_title(f"{selection}_{cross}")
                ax[idx_s, idx_c].plot(fitness_scores.tolist(), color, label=res_name)
                idx_c += 1
            idx_s += 1
            idx_c = 0
    legend = plt.legend(loc='lower left', shadow=False)
    plt.show()


def show_mutation_vs_non_mutation():
    results = load_data()

    def check_info(info, mut):
        return info[MUT_PROB] == mut and info[INV_PROB] == "0.0"

    results_non_inv = [r for r in results if check_info(r[0], mut='0.0')]
    results_inv1 = [r for r in results if check_info(r[0], mut='0.1')]
    results_inv05 = [r for r in results if check_info(r[0], mut='0.05')]

    def key_fun(o):
        return o[0][SELECTION_NAME] + "_" + o[0][CROSS_NAME]

    def key_fun2(o):
        return o[0][CROSS_NAME] + "_" + o[0][SELECTION_NAME]

    def cross_key(o):
        return o[0][CROSS_NAME]

    def selection_key(o):
        return o[0][SELECTION_NAME]

    fig, ax = plt.subplots(4, 4)
    for res_name, color, results in [("mut_prob=0.0", "r", results_non_inv), ("mut_prob=0.05", "g", results_inv05),
                                     ("mut_prob=0.1", "b", results_inv1)]:
        results.sort(key=key_fun)
        idx_s = 0
        idx_c = 0
        for selection, v_s in itertools.groupby(results, key=selection_key):
            v_ss = [i for i in v_s]
            for cross, v_c in itertools.groupby(v_ss, key=cross_key):
                values = [i[1] for i in v_c]
                fitness_scores = np.apply_along_axis(get_fitness_for_individual, 3, values)
                fitness_scores = fitness_scores.mean(axis=2)
                fitness_scores = fitness_scores.mean(axis=0)
                ax[idx_s, idx_c].set_ylim([-10, 0])
                ax[idx_s, idx_c].set_title(f"{selection}_{cross}")
                ax[idx_s, idx_c].plot(fitness_scores.tolist(), color, label=res_name)
                idx_c += 1
            idx_s += 1
            idx_c = 0
    legend = plt.legend(loc='lower left', shadow=False)
    plt.show()

show_inv_vs_non_inv()