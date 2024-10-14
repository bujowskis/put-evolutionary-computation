from pandas import DataFrame, read_csv
from numpy import zeros, ndarray, sqrt, round, ceil
from itertools import pairwise
from typing import List
from time import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


DATA_FOLDER = "data"


class SolutionTSP:
    def __init__(self, tsp: 'TSP', nodes: list):
        self.nodes = nodes
        self.cost = tsp.calculate_total_additional_cost(nodes)
        self.edge_length = tsp.calculate_total_edge_length(nodes)
        self.objective_function = tsp.calculate_total_objective_function(nodes)

    def nodes_in_excel_format(self) -> str:
        return '\n'.join(map(str, self.nodes))

    def __str__(self):
        return '\n'.join((
            '{',
            f'cost: {self.cost}',
            f'edge_length: {self.edge_length}',
            f'objective_function: {self.objective_function}',
            f'nodes: {self.nodes}',
            '}'
        ))

    @staticmethod
    def get_best_solution(solutions) -> 'SolutionTSP':
        return min(solutions)

    def __lt__(self, other: 'SolutionTSP'):
        return self.objective_function < other.objective_function

    def __eq__(self, other: 'SolutionTSP'):
        return self.objective_function == other.objective_function

    def __gt__(self, other: 'SolutionTSP'):
        return self.objective_function > other.objective_function


class TSP:
    def __init__(self, path: str):
        self.raw_data: DataFrame = read_csv(path, delimiter=';', header=None).rename(
            columns={0: 'x', 1: 'y', 2: 'additional_cost'})
        self.nodes: List[int] = [i for i in range(len(self.raw_data))]
        self.additional_costs: ndarray = self.calculate_additional_cost_array()
        # note - distance is the same both sides, so could improve by doing only upper half
        self.distances_matrix: ndarray = self.calculate_distances_matrix()
        self.total_move_costs: ndarray = self.calculate_total_move_costs_matrix()
        # note - distance is the same both sides, so could improve by doing only upper half
        self.insertion_costs: ndarray = self.calculate_insertion_costs()

    def calculate_additional_cost_array(self) -> ndarray:
        return self.raw_data['additional_cost'].to_numpy()

    def calculate_distances_matrix(self) -> ndarray:
        distances_matrix = zeros((len(self.raw_data), len(self.raw_data))).astype(int)
        for i in range(len(self.raw_data)):
            for j in range(i+1, len(self.raw_data)):
                point1 = (self.raw_data.loc[i, 'x'], self.raw_data.loc[i, 'y'])
                point2 = (self.raw_data.loc[j, 'x'], self.raw_data.loc[j, 'y'])
                distance = round(self.calculate_euclidean_distance(point1, point2), 0)
                distances_matrix[i][j] = distance
                distances_matrix[j][i] = distance
        return distances_matrix

    def calculate_total_move_costs_matrix(self) -> ndarray:
        total_move_costs = self.distances_matrix.copy()
        for i in range(len(self.raw_data)):
            for j in range(len(self.raw_data)):
                if i == j:
                    continue
                total_move_costs[i][j] += self.additional_costs[j]
        return total_move_costs

    def calculate_insertion_costs(self) -> ndarray:
        # [edge_start_node][edge_end_node][inserted_node]
        insertion_costs = zeros((len(self.raw_data), len(self.raw_data), len(self.raw_data))).astype(int)
        for start in range(len(self.raw_data)):
            for end in range(len(self.raw_data)):
                for inserted in range(len(self.raw_data)):
                    if inserted == start or inserted == end:
                        continue
                    if start == end:
                        # "inserting node between itself" - i.e. forming cycle by appending 1 node to just 1 node
                        # e.g. in greedy cycle method, at start
                        insertion_costs[start][end][inserted] = (
                                self.distances_matrix[start][inserted] + self.distances_matrix[inserted][end]
                                + self.additional_costs[inserted])
                        # no previous edge cost
                        continue
                    inserted_edge_cost = (self.distances_matrix[start][inserted] + self.distances_matrix[inserted][end]
                                          + self.additional_costs[inserted])
                    previous_edge_cost = self.distances_matrix[start][end]
                    insertion_costs[start][end][inserted] = inserted_edge_cost - previous_edge_cost

        return insertion_costs

    def get_required_number_of_nodes_in_solution(self) -> int:
        return ceil(len(self.raw_data) / 2).astype(int)

    def get_nodes(self, without_nodes: List[int] = None):
        nodes = self.nodes.copy()
        if without_nodes:
            for unwanted_node in without_nodes:
                nodes.remove(unwanted_node)
        return nodes

    def calculate_solution(self, nodes) -> 'SolutionTSP':
        return SolutionTSP(self, nodes)

    def calculate_total_additional_cost(self, nodes: list) -> int:
        return sum([self.additional_costs[node] for node in nodes])

    def calculate_total_edge_length(self, nodes: list) -> int:
        edges = TSP.determine_edges(nodes)
        return sum([self.distances_matrix[start][end] for start, end in edges])

    def calculate_total_objective_function(self, nodes: list):
        edges = TSP.determine_edges(nodes)
        return sum([self.total_move_costs[start][end] for start, end in edges])

    def visualize_solution(self, solution: 'SolutionTSP', method_name: str, path_to_save: str = None):
        nodes = solution.nodes

        x_coords = self.raw_data.loc[nodes, 'x'].to_numpy()
        y_coords = self.raw_data.loc[nodes, 'y'].to_numpy()
        additional_costs = self.raw_data.loc[nodes, 'additional_cost'].to_numpy()

        cmap = plt.cm.viridis

        x_coords_all = self.raw_data['x'].to_numpy()
        y_coords_all = self.raw_data['y'].to_numpy()
        additional_costs_all = self.raw_data['additional_cost'].to_numpy()

        all_additional_costs = np.concatenate((additional_costs_all, additional_costs))
        normalize_costs = mcolors.Normalize(vmin=all_additional_costs.min(), vmax=all_additional_costs.max())

        fig, ax = plt.subplots(figsize=(15, 12))
        for i in range(len(x_coords_all)):
            ax.plot(x_coords_all[i], y_coords_all[i], "o", markersize=10,
                    color=cmap(normalize_costs(additional_costs_all[i])), zorder=1)

        for i in range(len(x_coords)):
            ax.plot(x_coords[i], y_coords[i], "o", markersize=12, color=cmap(normalize_costs(additional_costs[i])),
                    zorder=2)

        for i, j in pairwise(nodes + [nodes[0]]):
            x_start, y_start = self.raw_data.loc[i, 'x'], self.raw_data.loc[i, 'y']
            x_end, y_end = self.raw_data.loc[j, 'x'], self.raw_data.loc[j, 'y']
            ax.plot((x_start, x_end), (y_start, y_end), "-", color='#36454f', zorder=1)

        gradient = np.linspace(0, 1, 256).reshape(-1, 1)
        axins = ax.inset_axes([1.05, 0.1, 0.05, 0.6], transform=ax.transAxes)
        axins.imshow(gradient, aspect='auto', cmap=cmap, origin='lower',
                     extent=[0, 1, additional_costs.min(), additional_costs.max()])
        axins.xaxis.set_visible(False)
        axins.set_ylabel('Cost', fontsize=12)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'TSP Solution: {method_name}')

        if path_to_save:
            plt.savefig(path_to_save)
        else:
            plt.show()

    @staticmethod
    def determine_edges(nodes: list) -> list:
        return pairwise(nodes + [nodes[0]])

    @staticmethod
    def calculate_euclidean_distance(point1, point2):
        return sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    @staticmethod
    def load_tspa(data_folder: str = DATA_FOLDER) -> 'TSP':
        return TSP(f'{data_folder}/TSPA.csv')

    @staticmethod
    def load_tspb(data_folder: str = DATA_FOLDER) -> 'TSP':
        return TSP(f'{data_folder}/TSPB.csv')


if __name__ == "__main__":
    t0 = time()
    tsp = TSP.load_tspa()
    t1 = time()
    print(f'execution_time: {t1 - t0}')
    print(tsp.raw_data.head())
    print(tsp.distances_matrix)
    print(tsp.additional_costs)
    print(tsp.total_move_costs)
    print(tsp.insertion_costs)
