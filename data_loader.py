from pandas import DataFrame, read_csv
from numpy import zeros, ndarray, sqrt, round, ceil
from itertools import pairwise


DATA_FOLDER = "data"


class TSP:
    def __init__(self, path: str):
        self.raw_data: DataFrame = read_csv(path, delimiter=';', header=None).rename(
            columns={0: 'x', 1: 'y', 2: 'additional_cost'})
        self.additional_costs: ndarray = self.calculate_additional_cost_array()
        self.distances_matrix: ndarray = self.calculate_distances_matrix()
        self.total_move_costs: ndarray = self.calculate_total_move_costs_matrix()

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

    def get_required_number_of_nodes_in_solution(self) -> int:
        return ceil(len(self.raw_data) / 2).astype(int)

    def calculate_total_additional_cost(self, nodes: list) -> int:
        return sum([self.additional_costs[node] for node in nodes])

    def calculate_total_edge_length(self, nodes: list) -> int:
        edges = TSP.determine_edges(nodes)
        return sum([self.distances_matrix[start][end] for start, end in edges])

    def calculate_total_objective_function(self, nodes: list):
        edges = TSP.determine_edges(nodes)
        return sum([self.total_move_costs[start][end] for start, end in edges])

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
    tsp = TSP.load_tspa()
    print(tsp.raw_data.head())
    print(tsp.distances_matrix)
    print(tsp.additional_costs)
    print(tsp.total_move_costs)
