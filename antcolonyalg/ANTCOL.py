import random
from typing import List

import numpy as np

import utils
from test_graphs import graph2, graph3

alpha = 1
beta = 1
vaporization_factor = 0.1
test_graph = graph3
n_ants = 10
max_cycles = 100


class Ant:

    def __init__(self, adj_graph, pheromone_matrix):
        self.pheromone_matrix = pheromone_matrix
        self.adj_graph = adj_graph
        self.start_vertex = random.randint(0, self.adj_graph.shape[0] - 1)

        # To reset
        self.path = []
        self.solution = [-1 for _ in graph2]
        self.colors_num = None
        self.are_visited = [False for _ in adj_graph]

    def reset_state(self):
        self.colors_num = None
        self.are_visited = [False for _ in self.adj_graph]
        self.path = []
        self.solution = [-1 for _ in self.adj_graph]

    def run(self):
        self.reset_state()
        current_vertex_idx = self.start_vertex
        self.are_visited = [False for _ in self.adj_graph]
        self.solution[current_vertex_idx] = 0
        self.path.append(current_vertex_idx)
        self.are_visited[current_vertex_idx] = True
        while not all(self.are_visited):
            current_vertex_idx = self.chooseNextVertex(current_vertex_idx)
            locked_colors = self.get_locked_colors(vertex_id=current_vertex_idx)
            for color in self.get_possible_colors():
                if color not in locked_colors:
                    self.solution[current_vertex_idx] = color
                    break
            self.path.append(current_vertex_idx)
            self.are_visited[current_vertex_idx] = True
        return self.path, self.solution, len(set(self.solution))

    def get_locked_colors(self, vertex_id):
        neighbours = self.get_neighbours(vertex_id)

        locked_colors = []
        for neighbour in neighbours:
            neighbour_color = self.solution[neighbour]
            if neighbour_color != -1:
                locked_colors.append(neighbour_color)
        return locked_colors

    def get_neighbours(self, vertex_id):
        neighbours_vertices = np.where(self.adj_graph[vertex_id] == 1)[1]
        return neighbours_vertices

    def chooseNextVertex(self, current_vertex):  # TODO: Losowanie z prawdopodobieństwami
        not_visited_indices = np.where(np.logical_not(np.array(self.are_visited)))[0]
        if len(not_visited_indices) == 1:
            return not_visited_indices[0]

        weights = []
        candidates = []

        for not_visited_vertex_idx in not_visited_indices:
            candidates.append(not_visited_vertex_idx)
            weights.append(self.get_probability_of_candidate(not_visited_vertex_idx, current_vertex))
        if sum(weights) == 0:
            weights = [1 for _ in weights]
        return random.choices(candidates, weights=weights, k=1)[0]

    def get_probability_of_candidate(self, candidate, current_vertex):
        pheromone_value = self.pheromone_matrix.item((candidate, current_vertex))
        vertex_grade = self.desaturation_degree(vertex=candidate)
        return pheromone_value ** alpha * vertex_grade ** beta

    def desaturation_degree(self, vertex):
        neighbours = self.get_neighbours(vertex)
        neighbours_applied_colors = []
        for n in neighbours:
            neighbours_applied_colors.append(self.solution[n])
        return len(set(neighbours_applied_colors))

    def get_possible_colors(self):
        return [i for i in range(len(self.solution))]

    def get_pheromone_path_matrix(self):
        matrix = np.zeros(self.pheromone_matrix.shape)
        edges = list(zip(self.path, self.path[1:] + self.path[:1]))  # Generate list of edges
        for edge in edges:
            delta_pheromone = 1 / len(set(self.solution))
            matrix[edge[0]][edge[1]] = delta_pheromone
            matrix[edge[1]][edge[0]] = delta_pheromone
        return matrix


class AntColonyAlgorithm:

    def __init__(self, graph_adj_matrix, num_ants, vaporization_factor):
        self.vaporization_factor = vaporization_factor
        self.graph_adj_matrix = graph_adj_matrix
        self.pheromone_matrix = utils.no_adj_matrix_operator(graph_adj_matrix)
        self.lowest_color_num = None
        self.best_solution = None
        self.ant_colony: List[Ant] = []
        self.num_ants = num_ants
        self.history = []

    def run(self):
        self.init_colony()
        for cycle in range(max_cycles):
            delta_pheromones_sum = np.zeros(self.pheromone_matrix.shape)
            for ant in self.ant_colony:
                path, solution, color_num = ant.run()
                delta_pheromones_sum = np.add(ant.get_pheromone_path_matrix(), delta_pheromones_sum)
                self.history.append(solution)
                if self.lowest_color_num is None or self.lowest_color_num > color_num:
                    self.best_solution = solution
                    self.lowest_color_num = color_num
                if self.check_solution(solution):
                    print("Error!")

            self.update_pheromones(delta_pheromones_sum)
        return self.best_solution, self.lowest_color_num

    def init_colony(self):
        self.ant_colony = [Ant(self.graph_adj_matrix, self.pheromone_matrix) for i in range(self.num_ants)]

    def update_colony(self):
        for ant in self.ant_colony:
            ant.pheromone_matrix = self.pheromone_matrix

    def update_pheromones(self, delta_pheromones_sum):
        def vaporization(x):
            return x * (1 - self.vaporization_factor)

        vaporization(self.pheromone_matrix)
        self.pheromone_matrix = np.add(self.pheromone_matrix, delta_pheromones_sum)

    def check_solution(self, solution):
        connected_vertex = np.where(self.graph_adj_matrix == 1)
        for idx in range(len(connected_vertex[0])):
            i, j = connected_vertex[0][idx], connected_vertex[1][idx]
            if i != j and solution[i] == solution[j]:
                return True
        return False


import matplotlib.pyplot as plt

if __name__ == '__main__':
    alg = AntColonyAlgorithm(test_graph, n_ants, vaporization_factor)
    print(alg.run())

    plt.plot([len(set(i)) for i in alg.history])
    plt.ylabel('some numbers')
    plt.show()