from data_loader import TSP
from random import sample


def random_solve(tsp: TSP) -> list:
    required_number_of_nodes = tsp.get_required_number_of_nodes_in_solution()
    chosen_nodes = sample([i for i in range(len(tsp.raw_data))], required_number_of_nodes)
    return chosen_nodes


if __name__ == "__main__":
    tsp = TSP.load_tspa(data_folder='../data')
    chosen_nodes = random_solve(tsp)
    print(chosen_nodes)
    print(f'total additional cost: {tsp.calculate_total_additional_cost(chosen_nodes)}')
    print(f'total edge length: {tsp.calculate_total_edge_length(chosen_nodes)}')
    print(f'total objective function: {tsp.calculate_total_objective_function(chosen_nodes)}')
