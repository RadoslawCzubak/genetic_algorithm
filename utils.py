import json
from typing import List

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


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


def is_list_of_unique_objects(l: List) -> bool:
    test = []
    for jj, _ in enumerate(l):
        for j, _ in enumerate(l):
            if jj != j:
                test.append(l[jj] is l[j])
    return any(test)


class AlgorithmEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.matrix):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


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


def save_algorithm_history(algorithm, path: str, filename: str):
    history = algorithm.history
    history = np.array([pop.individuals for pop in history])
    np.save(f"{path}/{filename}.npy", history)

def no_adj_matrix_operator(x):
    return (x + 1) % 2