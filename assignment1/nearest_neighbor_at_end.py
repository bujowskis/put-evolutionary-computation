from data_loader import TSP, SolutionTSP
from random import choice
from time import time


def nearest_neighbor_at_end_solve(tsp: TSP, starting_node: int = None) -> SolutionTSP:
    if starting_node is None:
        starting_node = choice(tsp.nodes)

    chosen_nodes = [starting_node]
    remaining_nodes = tsp.get_nodes(without_nodes=chosen_nodes)

    required_number_of_nodes = tsp.get_required_number_of_nodes_in_solution()
    while len(chosen_nodes) < required_number_of_nodes and remaining_nodes:
        current_end_node = chosen_nodes[-1]
        remaining_nodes_move_costs = [
            tsp.total_move_costs[current_end_node][node]
            for node in remaining_nodes
        ]

        smallest_move_cost = min(remaining_nodes_move_costs)
        # first occurrence in case of tie
        smallest_move_cost_node_idx = remaining_nodes_move_costs.index(smallest_move_cost)
        next_node = remaining_nodes[smallest_move_cost_node_idx]

        chosen_nodes.append(next_node)
        remaining_nodes.remove(next_node)

    return tsp.calculate_solution(chosen_nodes)


if __name__ == "__main__":
    tsp = TSP.load_tspa(data_folder='../data')
    t0 = time()
    solution = nearest_neighbor_at_end_solve(tsp)
    t1 = time()
    print(f'execution_time: {t1 - t0}')
    print(solution)
    print(solution.nodes_in_excel_format())
